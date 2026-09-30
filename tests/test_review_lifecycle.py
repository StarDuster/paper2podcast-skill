"""Offline behavioral checks: real pipeline/validators/files, injected model boundary only.

No model/TTS requests, credentials or existing episode assets are used. A render
sentinel checks the exact on-disk transcript and deliberately stops before audio.
"""
import io
import json
import socket
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from paper2podcast_lib import cli, input_parse, runtime, script, validation
import script_from_pdf_native as native


ENTRIES = [{'speaker_id': 0, 'dialog': '原始稿件内容'}]
REVISED = [{'speaker_id': 0, 'dialog': '审阅后的内容'}]


@pytest.fixture(autouse=True)
def offline(monkeypatch, tmp_path):
    def blocked(*a, **kw):
        raise AssertionError('Network/model/TTS boundary must not be reached')
    monkeypatch.setattr(socket.socket, 'connect', blocked)
    monkeypatch.setattr(script, 'call_llm', blocked)
    monkeypatch.setattr(script, 'call_gemini', blocked)
    monkeypatch.setattr(cli, '_render_segments', blocked)
    monkeypatch.setattr(cli, 'get_api_runtime', lambda args: None)
    monkeypatch.setattr(cli, 'configure_logging', lambda *a: None)
    for key in ('REVIEW_MODEL_BASE_URL', 'REVIEW_MODEL_API_KEY', 'REVIEW_MODEL_API_KEY_FILE'):
        monkeypatch.delenv(key, raising=False)
    runtime.reset_run_context()
    runtime.create_run_work_dir(str(tmp_path / 'standalone'))


def invoke(monkeypatch, tmp_path, src, draft=None, *extra):
    import uuid
    argv = ['paper2podcast', str(src), '--work-dir', str(tmp_path / ('run-' + uuid.uuid4().hex)),
            '--output', str(tmp_path / 'never-generated.mp3'), '--skip-search']
    if draft:
        argv += ['--script', str(draft)]
    monkeypatch.setattr(sys, 'argv', argv + list(extra))
    rc = cli.main()
    return rc, runtime.get_run_context()


def fixture_files(tmp_path):
    src = tmp_path / 'source.txt'
    src.write_text('全文依据，非摘要。', encoding='utf-8')
    draft = tmp_path / 'draft.json'
    draft.write_text(json.dumps({'podcast_transcripts': ENTRIES}), encoding='utf-8')
    return src, draft


def test_review_failure_closes_gate_and_preserves_draft(monkeypatch, tmp_path):
    src, draft = fixture_files(tmp_path)
    before = draft.read_bytes()
    for response in [TimeoutError('injected offline timeout'), 'invalid JSON', '{"podcast_transcripts": []}']:
        def review(**kw):
            if isinstance(response, Exception):
                raise response
            return response
        monkeypatch.setattr(script, '_call_review_model', review)
        rc, ctx = invoke(monkeypatch, tmp_path, src, draft, '--script-only')
        assert rc == 1 and ctx.failed_stage == 'script-review'
        assert not list(Path(ctx.work_dir).glob('final_script*.json'))
        assert list(Path(ctx.work_dir).glob('draft_*.json'))
        assert draft.read_bytes() == before
    for extra in [('--review-provider', 'gemini'), ('--review-provider', 'deepseek', '--review-model', 'gpt-6-astra')]:
        rc, ctx = invoke(monkeypatch, tmp_path, src, draft, '--script-only', *extra)
        assert rc == 1 and ctx.failed_stage == 'config'


def test_receipt_reuse_mutations_opt_out_and_tts_checkpoint(monkeypatch, tmp_path):
    src, draft = fixture_files(tmp_path)
    before = draft.read_bytes()
    calls = []
    def review(**kw):
        calls.append(kw)
        return json.dumps({'podcast_transcripts': REVISED})
    monkeypatch.setattr(script, '_call_review_model', review)
    rc, ctx = invoke(monkeypatch, tmp_path, src, draft, '--script-only')
    assert rc == 0 and len(calls) == 1
    reviewed_path = Path(ctx.script_path)
    reviewed_bytes = reviewed_path.read_bytes()
    reviewed = json.loads(reviewed_bytes)
    assert reviewed['review']['reviewed'] is True
    rc, ctx = invoke(monkeypatch, tmp_path, src, reviewed_path, '--script-only')
    assert rc == 0 and len(calls) == 1
    assert Path(ctx.script_path).read_bytes() == reviewed_bytes
    assert reviewed_path.read_bytes() == reviewed_bytes
    # Changed source invalidates receipt even though transcript did not change.
    src.write_text('修改后的全文依据', encoding='utf-8')
    rc, ctx = invoke(monkeypatch, tmp_path, src, reviewed_path, '--script-only')
    assert rc == 0 and len(calls) == 2
    # Changed transcript invalidates a current-source receipt.
    changed = json.loads(Path(ctx.script_path).read_text())
    changed['podcast_transcripts'][0]['dialog'] = '人工修改后内容'
    changed_path = tmp_path / 'changed.json'
    changed_path.write_text(json.dumps(changed))
    rc, ctx = invoke(monkeypatch, tmp_path, src, changed_path, '--script-only')
    assert rc == 0 and len(calls) == 3
    def stop_before_tts(args, api, segments, output_dir):
        saved = json.loads(Path(runtime.get_run_context().script_path).read_text())
        assert saved['podcast_transcripts'] == [s[0] for s in segments] == REVISED
        assert saved['review']['reviewed'] is True
        runtime.abort('tts-audio-synthesis', 'offline sentinel: no audio rendered')
    monkeypatch.setattr(cli, '_render_segments', stop_before_tts)
    rc, ctx = invoke(monkeypatch, tmp_path, src, draft)
    assert rc == 1 and ctx.failed_stage == 'tts-audio-synthesis' and len(calls) == 4
    assert draft.read_bytes() == before
    rc, ctx = invoke(monkeypatch, tmp_path, tmp_path / 'no-source-required', reviewed_path,
                     '--script-only', '--no-script-review')
    assert rc == 0 and len(calls) == 4
    assert json.loads(Path(ctx.script_path).read_text())['review'] == {
        'status': 'skipped', 'reviewed': False, 'reason': 'explicit --no-script-review'}


def gemini(payload, finish='STOP'):
    return {'candidates': [{'finishReason': finish, 'content': {'parts': [{'text': json.dumps(payload)}]}}]}


def test_generated_single_multistage_and_fallback_checkpoint(monkeypatch, tmp_path):
    src, _ = fixture_files(tmp_path)
    monkeypatch.setattr(script, '_call_review_model', lambda **kw: json.dumps(REVISED))
    for mode in ('single', 'multi', 'fallback'):
        def generate(api, model, body, **kw):
            if kw['request_label'] == 'outline generation':
                if mode == 'fallback':
                    return gemini({'segments': []})
                return gemini({'segments': [{'title': '介绍', 'key_points': ['事实'], 'word_budget': 10, 'tone': 'neutral'}]})
            return gemini(ENTRIES)
        monkeypatch.setattr(script, 'call_gemini', generate)
        extra = ['--no-multistage'] if mode == 'single' else []
        rc, ctx = invoke(monkeypatch, tmp_path, src, None, '--script-only', *extra)
        assert rc == 0
        saved = json.loads(Path(ctx.script_path).read_text())
        assert saved['podcast_transcripts'] == REVISED and saved['review']['reviewed']
        drafts = list(Path(ctx.work_dir).glob('draft_*.json'))
        assert len(drafts) == 1
        assert json.loads(drafts[0].read_text())['podcast_transcripts'] == ENTRIES
        # The generator's receipt must be reusable by the supplied-script branch.
        def must_not_review(**kw):
            raise AssertionError('already reviewed generated script was reviewed twice')
        with monkeypatch.context() as m:
            m.setattr(script, '_call_review_model', must_not_review)
            rc, _ = invoke(m, tmp_path, src, Path(ctx.script_path), '--script-only')
            assert rc == 0


def test_provider_endpoint_resolution_never_uses_sdk_default(monkeypatch):
    import openai
    observed = []
    response = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content='{}'))])
    def factory(**kw):
        observed.append(kw)
        return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **k: response)), close=lambda: None)
    monkeypatch.setattr(openai, 'OpenAI', factory)
    script._call_review_model(provider='deepseek', model='deepseek-v4-pro', messages=[],
        max_tokens=1, timeout=1, api_key='offline-placeholder', api_key_file=None, base_url=None)
    assert observed[0]['base_url'] == 'https://api.deepseek.com/v1'
    for provider, model, key in [('unknown', 'other', 'offline-placeholder'),
                                 ('openai', 'gpt-6-astra', None),
                                 ('openai-codex', 'gpt-6-astra', 'offline-placeholder')]:
        with pytest.raises(runtime.PipelineError):
            script.validate_review_route(provider, model, key)
    assert script.validate_review_route('openai-codex', 'gpt-6-astra') == ''
    assert len(observed) == 1


def test_native_validation_and_atomic_output(monkeypatch, tmp_path):
    pdf = tmp_path / 'paper.pdf'
    pdf.write_bytes(b'%PDF-1.7\nfixture')
    out = tmp_path / 'nested' / 'draft'  # no .json suffix
    monkeypatch.setattr(native, '_api_key', lambda: 'offline-placeholder')
    for payload, finish, accepted in [(ENTRIES, 'STOP', True),
        ({'podcast_transcripts': ENTRIES}, 'STOP', True),
        (ENTRIES, 'MAX_TOKENS', False), ({'podcast_transcripts': []}, 'STOP', False),
        ([{'speaker_id': 99, 'dialog': '内容'}], 'STOP', False),
        (ENTRIES, None, False), ('wrong shape', 'STOP', False)]:
        prior = out.read_bytes() if out.exists() else None
        def model(key, model, body):
            assert body['contents'][0]['parts'][0]['inline_data']['mime_type'] == 'application/pdf'
            return gemini(payload, finish)
        monkeypatch.setattr(native, 'call', model)
        monkeypatch.setattr(sys, 'argv', ['native', str(pdf), str(out)])
        assert native.main() == (0 if accepted else 1)
        if accepted:
            assert json.loads(out.read_text())['review'] == {'status': 'draft', 'reviewed': False}
        else:
            assert out.read_bytes() == prior
    # Atomic writer failure leaves prior bytes untouched, and removes temp file.
    prior = out.read_bytes()
    with monkeypatch.context() as m:
        def fail_replace(*a, **kw):
            raise OSError('injected replace failure')
        m.setattr(Path, 'replace', fail_replace)
        with pytest.raises(runtime.PipelineError):
            validation.write_json_file(str(out), {'changed': True}, 'file-write', 'test')
    assert out.read_bytes() == prior and not list(out.parent.glob('*.tmp'))


def real_pdf():
    """Minimal real PDF, usable by pdftotext (not a mocked extractor)."""
    text = b'BT /F1 12 Tf 50 750 Td (Full paper body beyond the abstract.) Tj ET'
    objects = [b'<< /Type /Catalog /Pages 2 0 R >>',
               b'<< /Type /Pages /Kids [3 0 R] /Count 1 >>',
               b'<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>',
               b'<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>',
               b'<< /Length ' + str(len(text)).encode() + b' >>\nstream\n' + text + b'\nendstream']
    data = b'%PDF-1.4\n'; offsets = []
    for i, obj in enumerate(objects, 1):
        offsets.append(len(data)); data += f'{i} 0 obj\n'.encode() + obj + b'\nendobj\n'
    xref = len(data)
    data += b'xref\n0 6\n0000000000 65535 f \n'
    data += b''.join(f'{offset:010} 00000 n \n'.encode() for offset in offsets)
    return data + f'trailer\n<< /Size 6 /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n'.encode()


def test_input_magic_short_html_and_arxiv_full_text(monkeypatch, tmp_path):
    class Response(io.BytesIO):
        headers = {'Content-Type': 'application/octet-stream'}
    requested = []
    def fetch(req, **kw):
        requested.append(req.full_url)
        return Response(real_pdf())
    monkeypatch.setattr(input_parse.urllib.request, 'urlopen', fetch)
    assert 'Full paper body beyond the abstract.' in input_parse.extract_text_from_url('https://example.invalid/download?id=1')
    assert 'Full paper body beyond the abstract.' in input_parse.load_input('https://arxiv.org/abs/2609.22978v2')
    assert requested[-1] == 'https://arxiv.org/pdf/2609.22978v2'
    local = tmp_path / 'download.bin'; local.write_bytes(real_pdf())
    assert 'Full paper body beyond the abstract.' in input_parse.load_input(str(local))
    class HtmlResponse(io.BytesIO):
        headers = {'Content-Type': 'text/html'}
    monkeypatch.setattr(input_parse.urllib.request, 'urlopen', lambda *a, **k:
                        HtmlResponse(b'<html><script>NOISE</script><p>short article</p></html>'))
    assert input_parse.extract_text_from_url('https://example.invalid/article') == 'short article'

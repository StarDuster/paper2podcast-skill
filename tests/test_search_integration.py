"""Offline integration regressions; injected boundaries never call a live model."""
import json
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))

from paper2podcast_lib import cli, config, script, search
from paper2podcast_lib.runtime import PipelineError, reset_run_context


def report(status='success'):
    return {'status': status, 'facts': [{'claim': 'Verified background', 'evidence': [
        {'source_id': 'S1', 'url': 'https://example.org/paper', 'quote': 'A verified literal source excerpt.'}]}],
        'sources': [{'id': 'S1', 'url': 'https://example.org/paper', 'text': 'A verified literal source excerpt.', 'coverage': 'extracted'}],
        'unverified': []}


def test_explicit_search_route_and_credentials_override_environment(monkeypatch):
    monkeypatch.setenv("SEARCH_MODEL", "environment-model")
    monkeypatch.setenv("SEARCH_PROVIDER", "environment-provider")
    calls = []

    def runner(command, *, timeout, env):
        calls.append((command, timeout, env))
        assert command[command.index('--model') + 1] == 'models/exact:search-model'
        assert command[command.index('--provider') + 1] == 'search-provider'
        assert 'secret-key' not in ' '.join(command)
        assert env['SEARCH_MODEL_API_KEY'] == 'secret-key'
        assert env['SEARCH_MODEL_BASE_URL'] == 'https://model.example/v1'
        assert env['SEARCH_MODEL_USE_HERMES'] == '1'
        Path(command[command.index('--output') + 1]).write_text(json.dumps(report()))
        return SimpleNamespace(returncode=0)

    text = search.search_context(
        'paper content',
        model='models/exact:search-model',
        provider='search-provider',
        model_api_key='secret-key',
        model_base_url='https://model.example/v1',
        use_hermes=True,
        runner=runner,
    )

    assert 'Verified background' in text
    assert len(calls) == 1
    assert 'https://example.org/paper' in text


def test_unverified_search_report_is_rejected_without_retry():
    ctx = reset_run_context()
    calls = []
    def runner(command, *, timeout):
        calls.append(command)
        data = report()
        data['facts'][0]['evidence'][0]['quote'] = 'Invented evidence not in source.'
        Path(command[command.index('--output') + 1]).write_text(json.dumps(data))
        return SimpleNamespace(returncode=0)
    assert search.search_context('paper', model='m', provider='p', runner=runner) == search.NO_CONTEXT
    assert len(calls) == 1 and ctx.degradations


def test_nonzero_partial_keeps_evidenced_facts_and_marks_gaps():
    ctx = reset_run_context()
    def runner(command, *, timeout):
        data = report('partial')
        data['unverified'] = ['Publication date remains unknown']
        Path(command[command.index('--output') + 1]).write_text(json.dumps(data))
        return SimpleNamespace(returncode=2)
    text = search.search_context('paper', model='m', provider='p', runner=runner)
    assert 'Verified background' in text
    assert 'Publication date remains unknown' in text
    assert ctx.degradations


def test_real_process_deadline_kills_grandchildren(tmp_path):
    marker = tmp_path / 'leaked.txt'
    child = f"import time; from pathlib import Path; time.sleep(0.8); Path({str(marker)!r}).write_text('leaked')"
    parent = f"import subprocess, sys, time; subprocess.Popen([sys.executable, '-c', {child!r}]); time.sleep(30)"
    started = time.monotonic()
    with pytest.raises(subprocess.TimeoutExpired):
        search.run_process([sys.executable, '-c', parent], timeout=0.2)
    assert time.monotonic() - started < 2
    time.sleep(0.9)
    assert not marker.exists(), 'grandchild survived deadline'


def gemini_response(payload):
    return {'candidates': [{'content': {'parts': [{'text': json.dumps(payload)}]}, 'finishReason': 'STOP'}]}


def test_default_generator_uses_native_codex_reference(monkeypatch, tmp_path):
    import codex_grounding
    reset_run_context().work_dir = str(tmp_path)
    monkeypatch.delenv('SEARCH_MODEL', raising=False)
    monkeypatch.delenv('SEARCH_PROVIDER', raising=False)
    calls, prompts = [], []
    def codex(command, **kwargs):
        calls.append(command)
        assert 'web_search="live"' in command
        assert '--ignore-user-config' in command
        assert command[command.index('-m') + 1] == 'gpt-6-astra'
        Path(command[command.index('-o') + 1]).write_text('Default grounding https://example.org')
        return SimpleNamespace(returncode=0, stdout='', stderr='')
    monkeypatch.setattr(codex_grounding.subprocess, 'run', codex)
    def runner(command, *, timeout, env):
        codex_grounding.research(Path(command[command.index('--question-file') + 1]).read_text(),
                                model=command[command.index('--model') + 1], timeout=timeout,
                                output=command[command.index('--output') + 1])
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(search, 'run_process', runner)
    def generate(*args, **kwargs):
        prompts.append(str(args[2]))
        return gemini_response({'podcast_transcripts': [{'speaker_id': 0, 'dialog': '正文'}]})
    monkeypatch.setattr(script, 'call_gemini', generate)
    assert script.generate_script(object(), 'paper', review_enabled=False)['podcast_transcripts']
    assert len(calls) == 1
    assert 'Default grounding https://example.org' in prompts[0]


def test_multistage_fallback_never_repeats_search(monkeypatch):
    context = 'CACHED VERIFIED CONTEXT'
    searched, fallback = [], []
    def search(*args, **kwargs):
        searched.append(kwargs)
        return context
    monkeypatch.setattr(script, 'search_context', search)
    def generate(runtime, model, body, **kwargs):
        if kwargs['request_label'] == 'outline generation':
            raise RuntimeError('planned offline generation failure')
        fallback.append(str(body))
        return gemini_response({'podcast_transcripts': [{'speaker_id': 0, 'dialog': '正文'}]})
    monkeypatch.setattr(script, 'call_gemini', generate)
    script.generate_script_multistage(object(), 'paper', review_enabled=False,
                                     search_model='research', search_provider='provider')
    assert len(searched) == 1
    assert len(fallback) == 1
    assert context in fallback[0]


@pytest.mark.parametrize('backend', ['tavily-agent', 'gemini'])
def test_skip_search_precedes_all_search_authentication(monkeypatch, backend):
    def forbidden(*args, **kwargs):
        pytest.fail('search was initialized despite skip-search')
    monkeypatch.setattr(search, 'run_process', forbidden)
    monkeypatch.setattr(config, 'get_api_runtime', forbidden)
    assert search.search_context('paper', skip_search=True, backend=backend, model='m', provider='p') == search.NO_CONTEXT
    monkeypatch.setattr(script, 'search_context', forbidden)
    assert script._prepare_script_inputs(None, 'paper', 'generation-model', True,
                                         search_backend=backend)[1] == search.NO_CONTEXT


@pytest.mark.parametrize('script_only', [False, True])
def test_existing_script_opt_out_never_searches_or_loads_dummy_input(monkeypatch, tmp_path, script_only):
    existing = tmp_path / 'existing.json'
    existing.write_text(json.dumps({'podcast_transcripts': [{'speaker_id': 0, 'dialog': '已有正文'}]}))
    output = tmp_path / 'podcast.mp3'
    argv = ['paper2podcast', 'nonexistent-dummy-input', '--script', str(existing),
            '--no-script-review',  # dummy input is only valid with explicit review opt-out
            '--work-dir', str(tmp_path / 'run'), '--output', str(output)]
    if script_only:
        argv.append('--script-only')
    monkeypatch.setattr(sys, 'argv', argv)
    def forbidden(*args, **kwargs):
        pytest.fail('existing script initialized search/generation/input loading')
    for name in ['load_input', 'generate_script', 'generate_script_multistage']:
        monkeypatch.setattr(cli, name, forbidden)
    monkeypatch.setattr(script, 'search_context', forbidden)
    monkeypatch.setattr(search, 'run_process', forbidden)
    auth = []
    def runtime(args):
        assert not script_only, 'script-only initialized API authentication'
        auth.append(args)
        return object()
    monkeypatch.setattr(cli, 'get_api_runtime', runtime)
    monkeypatch.setattr(cli, 'configure_logging', lambda *a: None)
    monkeypatch.setattr(cli, '_render_segments', lambda *a: ['offline.mp3'])
    monkeypatch.setattr(cli, 'concat_segments', lambda *a, **k: output.write_bytes(b'offline-render-test'))
    assert cli.main() == 0
    assert len(auth) == (0 if script_only else 1)


def test_legacy_auth_failure_is_search_degradation_not_fatal_pipeline_error(monkeypatch):
    ctx = reset_run_context()
    def unavailable(args):
        ctx.failed_stage = 'config'
        raise PipelineError('config', 'credentials unavailable')
    monkeypatch.setattr(config, 'get_api_runtime', unavailable)
    assert search.search_context('paper', backend='gemini', model='gemini-research', provider='vertex') == search.NO_CONTEXT
    assert ctx.failed_stage is None
    assert ctx.degradations

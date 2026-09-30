"""Offline research failures; quota errors are injected, never live account evidence."""
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import codex_grounding
from paper2podcast_lib import search, script
from paper2podcast_lib.runtime import reset_run_context
from tavily_agent import validate_report
from test_tavily_agent import report, URL, TEXT


def test_quota_fallback_preserves_full_input_and_isolates_credentials(monkeypatch, tmp_path):
    monkeypatch.delenv('SEARCH_MODEL', raising=False)
    monkeypatch.delenv('SEARCH_PROVIDER', raising=False)
    for name in ('OPENAI_API_KEY', 'OPENAI_BASE_URL', 'SEARCH_MODEL_API_KEY',
                 'SEARCH_MODEL_API_KEYS', 'SEARCH_MODEL_USE_HERMES', 'BASE_URL'):
        monkeypatch.setenv(name, 'poison')
    monkeypatch.setenv('DEEPSEEK_API_KEY', 'deepseek-only')
    monkeypatch.setenv('DEEPSEEK_BASE_URL', 'https://api.deepseek.com/v1')
    reset_run_context().work_dir = str(tmp_path)
    monkeypatch.setattr(codex_grounding.subprocess, 'run', lambda *a, **kw:
                        SimpleNamespace(returncode=1, stdout='You have hit your usage limit', stderr=''))
    calls = []

    def runner(command, *, timeout, env):
        question = Path(command[command.index('--question-file') + 1]).read_text()
        calls.append(question)
        assert 'poison' not in env.values()
        assert 'deepseek-only' not in ' '.join(command)
        output = Path(command[command.index('--output') + 1])
        if len(calls) == 1:
            result = codex_grounding.research(question, model='gpt-6-astra', timeout=timeout, output=output)
            assert result['stop_reason'] == 'codex_quota_exhausted'
            return SimpleNamespace(returncode=2)
        assert command[command.index('--model') + 1] == 'deepseek-v4-pro'
        assert command[command.index('--provider') + 1] == 'openai-compatible'
        assert env['SEARCH_MODEL_API_KEY'] == 'deepseek-only'
        assert env['SEARCH_MODEL_BASE_URL'] == 'https://api.deepseek.com/v1'
        assert env['SEARCH_MODEL_USE_HERMES'] == '0'
        draft = {**report(), 'reference_text': '参考文章 [S1]'}
        output.write_text(json.dumps(validate_report(draft, [{'id': 'S1', 'url': URL, 'text': TEXT}])))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(search, 'run_process', runner)
    original = '完整原文\n' * 60000 + 'END'
    prepared, context, _ = script._prepare_script_inputs(None, original, 'generation-model', False)
    assert len(calls) == 2 and calls[0] == calls[1]
    assert calls[0].endswith(original)
    assert len(prepared) < len(original)
    assert '参考文章 [S1]' in context and URL in context
    directory = next(tmp_path.glob('grounding-*'))
    manifest = json.loads((directory / 'manifest.json').read_text())
    assert manifest['attempts'][0]['stop_reason'] == 'codex_quota_exhausted'
    assert manifest['attempts'][1]['usable'] is True
    assert (directory / 'reference.md').read_text() == context


def test_execution_failure_and_failed_fallback_do_not_fabricate_context(monkeypatch, tmp_path):
    monkeypatch.setenv('DEEPSEEK_API_KEY', 'test-key')
    ctx = reset_run_context()
    ctx.work_dir = str(tmp_path)
    calls = []

    def runner(command, **kwargs):
        calls.append(command)
        if len(calls) == 1:
            raise subprocess.TimeoutExpired('secret credential', 1)
        Path(command[command.index('--output') + 1]).write_text(json.dumps({'status': 'failed'}))
        return SimpleNamespace(returncode=2)

    assert search.search_context('original', model='m', provider='openai-codex', runner=runner) == search.NO_CONTEXT
    assert len(calls) == 2
    assert not next(tmp_path.glob('grounding-*')).joinpath('reference.md').exists()
    assert 'secret credential' not in str(ctx.degradations)


def test_cancellation_and_bad_input_do_not_fallback(monkeypatch, tmp_path):
    reset_run_context().work_dir = str(tmp_path)
    calls = []
    def runner(*a, **kw):
        calls.append(1)
        raise KeyboardInterrupt()
    with pytest.raises(KeyboardInterrupt):
        search.search_context('original', model='m', provider='openai-codex', runner=runner)
    with pytest.raises(ValueError):
        search.search_context(' ', model='m', provider='openai-codex', runner=runner)
    monkeypatch.setenv('PODCAST_CODEX_TIMEOUT', 'nan')
    with pytest.raises(ValueError):
        search.search_context('original', model='m', provider='openai-codex', runner=runner)
    assert len(calls) == 1


def test_codex_failed_event_cannot_be_success(monkeypatch, tmp_path):
    def runner(command, **kwargs):
        Path(command[command.index('-o') + 1]).write_text('Some text https://example.org')
        return SimpleNamespace(returncode=0, stdout='{"type":"turn.failed","error":{"message":"request failed"}}\n', stderr='')
    monkeypatch.setattr(codex_grounding.subprocess, 'run', runner)
    result = codex_grounding.research('original', model='gpt-6-astra', timeout=2, output=tmp_path / 'out.json')
    assert result['status'] == 'failed'

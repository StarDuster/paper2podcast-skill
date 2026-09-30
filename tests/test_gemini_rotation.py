import json
import sys
from pathlib import Path
from types import SimpleNamespace

import httpx
import openai
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import search_model

URL = 'https://generativelanguage.googleapis.com/v1beta/openai/'


def setup_client(monkeypatch, statuses):
    calls = []
    def handler(request):
        calls.append(request)
        status = statuses[min(len(calls)-1, len(statuses)-1)]
        if status != 200:
            return httpx.Response(status, json={'error': {'message': 'secret-one secret-two', 'type': 'error'}})
        return httpx.Response(200, json={'choices':[{'message':{'content':'{"ok":true}'}, 'finish_reason':'stop', 'index':0}], 'usage':{}})
    original = openai.OpenAI
    monkeypatch.setattr(openai, 'OpenAI', lambda **kw: original(**kw, http_client=httpx.Client(transport=httpx.MockTransport(handler))))
    return calls


def payload():
    return {'config': {'provider':'portable','model':'gemini-flash-latest','api_key':'secret-one','api_keys':['secret-one','secret-two','secret-one'], 'base_url':URL}, 'messages':[{'role':'user','content':'test'}], 'timeout':10}


def test_environment_pool_and_redaction(monkeypatch):
    monkeypatch.setenv('SEARCH_MODEL_API_KEYS', json.dumps(['secret-one', 'secret-two']))
    model = search_model.Model('gemini-flash-latest', 'portable', api_key='secret-one', base_url=URL)
    assert model.config['api_keys'] == ['secret-one', 'secret-two']
    calls = setup_client(monkeypatch, [400])
    result = search_model.worker({**payload(), 'config': model.config})
    assert len(calls) == 1
    assert 'secret-one' not in json.dumps(result)
    assert 'secret-two' not in json.dumps(result)


@pytest.mark.parametrize('status', [400, 401, 403, 429, 500])
def test_non503_does_not_rotate(monkeypatch, status):
    calls = setup_client(monkeypatch, [status, 200])
    assert search_model.worker(payload())['error'] == 'model_request_failed'
    assert len(calls) == 1


def test_all503_exhausts_unique_pool_without_leaking(monkeypatch):
    calls = setup_client(monkeypatch, [503])
    result = search_model.worker(payload())
    assert len(calls) == 2
    assert result['key_attempts'] == [{'key_index':1,'status':503},{'key_index':2,'status':503}]
    assert 'secret-' not in json.dumps(result)


def test_other_endpoint_does_not_receive_pool(monkeypatch):
    calls = setup_client(monkeypatch, [503, 200])
    p = payload()
    p['config']['base_url'] = 'https://other.example/v1/'
    assert search_model.worker(p)['error'] == 'model_request_failed'
    assert len(calls) == 1


def test_deadline_shared(monkeypatch):
    import time
    clock = iter([0, 0, 11])
    monkeypatch.setattr(time, 'monotonic', lambda: next(clock))
    calls = []
    def fail(config, p):
        calls.append(p['timeout'])
        raise openai.InternalServerError('busy', response=httpx.Response(503, request=httpx.Request('POST', URL)), body=None)
    monkeypatch.setattr(search_model, '_openai_single_complete', fail)
    result = search_model.worker(payload())
    assert result['error_type'] == 'TimeoutError'
    assert calls == [10]


def test_deadline_rejects_late_success(monkeypatch):
    import time
    clock = iter([0, 0, 11])
    monkeypatch.setattr(time, 'monotonic', lambda: next(clock))
    monkeypatch.setattr(search_model, '_openai_single_complete', lambda *a: {'result':{'ok':True}, 'usage':{}})
    assert search_model.worker(payload()).get('error_type') == 'TimeoutError'


def test_503_rotates_once_per_unique_key(monkeypatch):
    calls = setup_client(monkeypatch, [503, 200])
    result = search_model.worker(payload())
    assert result.get('result') == {'ok': True}
    assert [r.headers['authorization'] for r in calls] == ['Bearer secret-one','Bearer secret-two']
    assert result['usage']['key_attempts'] == [{'key_index':1,'status':503},{'key_index':2,'status':200}]

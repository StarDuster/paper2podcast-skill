"""Offline adapter contracts using a local HTTP server, not live research."""
import asyncio
from contextlib import asynccontextmanager
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from aiohttp import web

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))

from search_model import Model
from tavily_agent import Agent, ResearchError, Tavily
from test_tavily_agent import FakeModel, FakeTavily

@asynccontextmanager
async def serve(handler):
    app = web.Application()
    app.router.add_post('/{action:.*}', handler)
    runner = web.AppRunner(app)
    await runner.setup()
    try:
        site = web.TCPSite(runner, '127.0.0.1', 0)
        await site.start()
        yield f'http://127.0.0.1:{site._server.sockets[0].getsockname()[1]}'
    finally:
        await runner.cleanup()


class ModelTests(unittest.IsolatedAsyncioTestCase):
    async def test_non_codex_provider_requires_explicit_hermes(self):
        with self.assertRaisesRegex(ResearchError, 'model_credentials_unavailable .*--use-hermes'):
            await Model('local-model', 'some-hermes-provider').complete([], 3)

    async def test_invalid_model_shape_returns_safe_failure(self):
        result = await Agent(FakeModel([{'queries': None}]), FakeTavily()).run('question')
        self.assertEqual(result['stop_reason'], 'invalid_model_action')
        self.assertEqual(result['status'], 'failed')

    async def test_child_deadline_kills_stalled_provider_resolution(self):
        with tempfile.TemporaryDirectory() as directory:
            package = Path(directory) / 'agent'
            package.mkdir()
            (package / '__init__.py').write_text('')
            (package / 'auxiliary_client.py').write_text('import time\ndef resolve_provider_client(*a, **kw):\n time.sleep(10)\n')
            started = time.monotonic()
            with patch.dict('os.environ', {'PYTHONPATH': directory}):
                with self.assertRaises(TimeoutError):
                    await Model('local-model', 'test-hermes', use_hermes=True).complete([], .15)
            self.assertLess(time.monotonic() - started, 1)

    async def test_portable_model_json_from_openai_compatible_endpoint(self):
        calls = []
        async def handler(request):
            calls.append(await request.json())
            return web.json_response({'choices': [{'message': {'content': '{"queries":["paper"]}'}}], 'usage': {'total_tokens': 12}})
        async with serve(handler) as base_url:
            model = Model('local-model', 'portable', api_key='test', base_url=base_url)
            result, usage = await model.complete([{'role':'user','content':'JSON'}], 3)
            self.assertEqual(result, {'queries':['paper']})
            self.assertEqual(usage['total_tokens'], 12)
            self.assertNotIn('tools', calls[0])

class HTTPTests(unittest.IsolatedAsyncioTestCase):
    async def test_401_once_and_http_deadline(self):
        calls = []
        async def handler(request):
            calls.append(request.path)
            if request.path == '/extract':
                await asyncio.sleep(.2)
                return web.json_response({'results': []})
            return web.Response(status=401, text='secret')
        async with serve(handler) as base_url:
            client = Tavily(api_key='test', base_url=base_url)
            with self.assertRaisesRegex(ResearchError, '^tavily_http_401$'):
                await client.search('question', 1)
            with self.assertRaises(TimeoutError):
                await client.extract(['https://example.org'], .02)
            self.assertEqual(calls, ['/search', '/extract'])

    async def test_tavily_auth_payload_and_no_retries(self):
        calls = []
        async def handler(request):
            payload = await request.json()
            calls.append((request.path, request.headers.get('Authorization'), payload))
            if payload.get('query') == 'bad query':
                return web.Response(status=400)
            if request.path == '/search':
                return web.json_response({'results': [], 'usage': {'credits': 1}})
            return web.Response(status=429, text='secret must not appear')
        async with serve(handler) as base_url:
            client = Tavily(api_key='test-key', base_url=base_url)
            self.assertEqual((await client.search('query', 1))['results'], [])
            with self.assertRaisesRegex(ResearchError, '^tavily_http_429$'):
                await client.extract(['https://example.org'], 1)
            self.assertEqual(len(calls), 2)
            self.assertEqual(calls[0][1], 'Bearer test-key')
            self.assertEqual(calls[0][2]['query'], 'query')
            self.assertEqual(calls[1][2]['urls'], ['https://example.org'])
            skipped = await client.search('bad query', 1)
            self.assertEqual(skipped['results'], [])
            self.assertEqual(skipped['skipped_status'], 400)

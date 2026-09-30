"""OFFLINE tests: injected model and Tavily doubles; never live evidence."""
import asyncio
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))

from tavily_agent import Agent, ResearchError, Tavily

URL = 'https://example.org/paper'
TEXT = 'The cache is shared across layers. ' * 30

class FakeModel:
    def __init__(self, replies):
        self.replies = iter(replies)
        self.inputs = []
    async def complete(self, messages, timeout):
        self.inputs.append(messages)
        return next(self.replies), {'prompt_tokens': 10, 'completion_tokens': 20}

class FakeTavily:
    def __init__(self):
        self.calls = []
    async def search(self, query, timeout):
        self.calls.append(('search', query))
        return {'results': [{'url': URL, 'title': 'Paper', 'content': 'snippet'}], 'usage': {'credits': 2}}
    async def extract(self, urls, timeout):
        self.calls.append(('extract', urls))
        return {'results': [{'url': u, 'raw_content': TEXT} for u in urls], 'usage': {'credits': 1}}

def report():
    return {'facts': [{'claim': 'Cache sharing', 'evidence': [{'source_id': 'S1', 'url': URL, 'quote': 'The cache is shared across layers.'}]}], 'unverified': []}

class AgentTests(unittest.IsolatedAsyncioTestCase):
    async def test_missing_key_fails_without_network(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(ResearchError, 'missing_tavily_key'):
                Tavily()


    async def test_caps_restrict_queries_documents_and_events_record_usage(self):
        model = FakeModel([{'queries': ['one', 'two', 'three', 'four']}, {'extract': [URL]}, {'done': True, **report()}])
        web = FakeTavily()
        result = await Agent(model, web, max_doc_chars=220, max_total_chars=250).run('Question')
        self.assertLessEqual(sum(len(s['text']) for s in result['sources']), 250)
        self.assertLessEqual(len(result['sources'][0]['text']), 220)
        self.assertEqual(result['sources'][0]['coverage'], 'truncated')
        self.assertEqual(len([c for c in web.calls if c[0] == 'search']), 3)
        self.assertEqual(result['usage']['llm_calls'], 3)
        self.assertEqual(result['usage']['tavily_credits'], 7)
        self.assertTrue(any(e['kind'] == 'model_response' for e in result['events']))


    async def test_incomplete_extract_and_arxiv_missing_abstract_are_marked(self):
        for url, text in [(URL, ''), ('https://arxiv.org/abs/2405.05254', 'Download PDF References ' * 30)]:
            class Incomplete(FakeTavily):
                async def search(self, query, timeout):
                    return {'results': [{'url': url, 'title': 'Paper', 'content': 'snippet'}]}
                async def extract(self, urls, timeout):
                    return {'results': [{'url': url, 'raw_content': text}]}
            model = FakeModel([{'queries': ['paper']}, {'extract': [url]}, {'done': True, 'facts': [], 'unverified': []}])
            result = await Agent(model, Incomplete()).run('Question')
            self.assertNotEqual(result['status'], 'success')
            self.assertEqual(result['sources'][0]['coverage'], 'incomplete')
            self.assertTrue(any('extraction' in s for s in result['unverified']))


    async def test_round_limit_forces_honest_report(self):
        model = FakeModel([{'queries': ['paper']}, {'extract': [URL]}, {'done': False, 'gaps': ['unread followup'], 'queries': ['followup'], **report()}])
        web = FakeTavily()
        result = await Agent(model, web, max_rounds=1).run('Question')
        self.assertEqual(result['status'], 'partial')
        self.assertIn('unread followup', result['unverified'])
        self.assertEqual(result['rounds'], 1)
        self.assertEqual(len(result['facts']), 1)


    async def test_api_budget_stops_before_next_paid_call(self):
        model = FakeModel([{'queries': ['paper']}, {'extract': [URL]}])
        web = FakeTavily()
        result = await Agent(model, web, max_api_calls=1).run('Question')
        self.assertNotEqual(result['status'], 'success')
        self.assertEqual(result['stop_reason'], 'api_budget')
        self.assertEqual(web.calls, [('search', 'paper')])
        self.assertEqual(result['usage']['api_calls'], 1)


    async def test_deadline_cancels_model_and_retains_partial_sources(self):
        class SlowModel(FakeModel):
            async def complete(self, messages, timeout):
                if len(self.inputs) == 2:
                    await asyncio.sleep(10)
                return await super().complete(messages, timeout)
        model = SlowModel([{'queries': ['paper']}, {'extract': [URL]}])
        result = await Agent(model, FakeTavily(), deadline=0.04).run('Question')
        self.assertEqual(result['status'], 'partial')
        self.assertLess(result['elapsed_seconds'], 0.5)
        self.assertEqual(result['stop_reason'], 'deadline')
        self.assertTrue(result['sources'])
        self.assertEqual(result['facts'], [])


    async def test_fabricated_evidence_is_rejected(self):
        for field, value in [('source_id', 'S999'), ('url', 'https://fake.invalid'), ('quote', 'Invented passage')]:
            with self.subTest(field=field):
                draft = report()
                draft['facts'][0]['evidence'][0][field] = value
                model = FakeModel([{'queries': ['paper']}, {'extract': [URL]}, {'done': True, **draft}])
                result = await Agent(model, FakeTavily()).run('Question')
                self.assertNotEqual(result['status'], 'success')
                self.assertEqual(result['facts'], [])
                self.assertTrue(result['unverified'])

    async def test_gap_research_deduplicates_queries_and_urls(self):
        model = FakeModel([{'queries': ['paper cache', 'paper cache']}, {'extract': [URL, URL]},
                           {'done': False, 'gaps': ['followup'], 'queries': ['paper cache', 'followup']},
                           {'extract': [URL]}, {'done': True, **report()}])
        web = FakeTavily()
        result = await Agent(model, web).run('Explain cache and followup')
        self.assertEqual(result['status'], 'success')
        self.assertEqual(web.calls, [('search', 'paper cache'), ('extract', [URL]), ('search', 'followup')])
        self.assertEqual(len(result['sources']), 1)

    async def test_success_search_read_assess_report(self):
        model = FakeModel([{'queries': ['paper cache']}, {'extract': [URL]}, {'done': True, **report()}])
        web = FakeTavily()
        result = await Agent(model, web).run('Explain cache design')
        self.assertEqual(result['status'], 'success')
        self.assertEqual(len(result['facts']), 1)
        self.assertEqual(result['sources'][0]['text'], TEXT)
        self.assertEqual([x[0] for x in web.calls], ['search', 'extract'])
        self.assertIn(TEXT, str(model.inputs[-1]))

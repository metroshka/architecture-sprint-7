"""Адрес LLM: сохранение локального режима и подключение из контейнера."""
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from llm_client import LLMClient, resolve_base_url


class EndpointTests(unittest.TestCase):
    def test_default_remains_loopback(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(resolve_base_url({'host':'127.0.0.1', 'port':8080}), 'http://127.0.0.1:8080')

    def test_override_is_used_by_all_startup_checks(self):
        responses = [{}, {'data':[{'id':'qwen3-8b'}]}, {'default_generation_settings':{'n_ctx':8192}}]
        with patch.dict(os.environ, {'RAG_LLM_BASE_URL':'http://host.docker.internal:8080/'}):
            with patch('llm_client.request_json', side_effect=responses) as request:
                client = LLMClient()
        self.assertEqual(client.base_url, 'http://host.docker.internal:8080')
        self.assertEqual(client.config['host'], '127.0.0.1')
        self.assertEqual([c.args[1] for c in request.call_args_list], ['/health', '/v1/models', '/props'])
        self.assertTrue(all(c.args[0] == client.base_url for c in request.call_args_list))

    def test_invalid_overrides_fail_before_network(self):
        for value in ['', 'localhost:8080', 'file:///tmp/model', 'http://host/v1',
                      'http://user:password@host', 'http://host?token=value', 'http://host#fragment',
                      'http://host:99999']:
            with self.subTest(value=value), patch.dict(os.environ, {'RAG_LLM_BASE_URL':value}):
                with patch('llm_client.request_json') as request:
                    with self.assertRaises(ValueError):
                        LLMClient()
                    request.assert_not_called()


if __name__ == '__main__':
    unittest.main()

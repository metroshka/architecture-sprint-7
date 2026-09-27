"""HTTP-клиент локального llama.cpp с контролем бюджета контекста."""
import json
from time import perf_counter
import urllib.request
from download_llm import CONFIG_PATH


def request_json(base_url, endpoint, payload=None, timeout=120):
    data = None if payload is None else json.dumps(payload, ensure_ascii=False).encode('utf-8')
    request = urllib.request.Request(base_url + endpoint, data=data, headers={'Content-Type':'application/json'})
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(request, timeout=timeout) as response:
        return json.load(response)


class LLMClient:
    def __init__(self):
        self.config = json.loads(CONFIG_PATH.read_text(encoding='utf-8'))
        self.base_url = f"http://{self.config['host']}:{self.config['port']}"
        request_json(self.base_url, '/health', timeout=5)
        models = request_json(self.base_url, '/v1/models', timeout=5)
        if self.config['model_alias'] not in [m.get('id') for m in models.get('data', [])]:
            raise ValueError('Сервер не сообщает ожидаемое имя модели')
        props = request_json(self.base_url, '/props', timeout=5)
        self.server_context = props['default_generation_settings']['n_ctx']
        if self.server_context < self.config['context_tokens']:
            raise ValueError('Контекст сервера меньше настроенного; перезапустите сервер с конфигурацией проекта')

    def complete(self, messages, schema):
        started = perf_counter()
        kwargs = {'enable_thinking':False}
        rendered = request_json(self.base_url, '/apply-template', {'messages':messages, 'chat_template_kwargs':kwargs})
        tokens = request_json(self.base_url, '/tokenize', {'content':rendered['prompt'], 'add_special':True, 'parse_special':True})['tokens']
        prompt_tokens = len(tokens)
        # Резерв для различий форматирования сервером; обрезание контекста не используется.
        if prompt_tokens + self.config['max_answer_tokens'] + 32 > self.config['context_tokens']:
            raise ValueError('Промпт и резерв ответа превышают контекст LLM; сократите вопрос или число фрагментов')
        response = request_json(self.base_url, '/v1/chat/completions', {
            'model':self.config['model_alias'], 'messages':messages,
            'max_tokens':self.config['max_answer_tokens'], 'temperature':0, 'stream':False,
            'chat_template_kwargs':kwargs,
            'response_format':{'type':'json_object', 'schema':schema},
        })
        choice = response['choices'][0]
        if choice.get('finish_reason') != 'stop':
            raise ValueError('Ответ модели не завершён; увеличивать лимит можно только с учётом бюджета контекста')
        return {'data':json.loads(choice['message']['content']),
                'usage':response.get('usage'), 'prompt_tokens_checked':prompt_tokens,
                'llm_seconds':round(perf_counter()-started, 6)}

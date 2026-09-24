"""Проверка готовности сервера и генерации по короткому тестовому контексту."""
import json
from time import perf_counter
import sys
import urllib.error
import urllib.request
from download_llm import CONFIG_PATH, ROOT


def request_json(base_url, endpoint, payload=None, timeout=120):
    data = None if payload is None else json.dumps(payload, ensure_ascii=False).encode('utf-8')
    request = urllib.request.Request(base_url + endpoint, data=data,
                                     headers={'Content-Type':'application/json'})
    # Локальный запрос не должен перенаправляться через прокси окружения.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(request, timeout=timeout) as response:
        return json.load(response)


def main():
    config = json.loads(CONFIG_PATH.read_text(encoding='utf-8'))
    base_url = f"http://{config['host']}:{config['port']}"
    request_json(base_url, '/health', timeout=5)
    models = request_json(base_url, '/v1/models', timeout=5)
    if config['model_alias'] not in [model.get('id') for model in models.get('data', [])]:
        raise ValueError('Сервер не сообщает ожидаемое имя модели qwen3-8b')
    print('Сервер готов. Проверка генерации по тестовому контексту…', flush=True)
    messages = [
        {'role':'system', 'content':'Отвечай по-русски только на основании предоставленного контекста. Дай одно короткое предложение. Если сведений нет, ответь: «Я не знаю». Не добавляй рассуждений.'},
        {'role':'user', 'content':'Тестовый контекст: Столица вымышленной планеты Тевари — город Келар.\nВопрос: Как называется столица планеты Тевари?'}
    ]
    started = perf_counter()
    response = request_json(base_url, '/v1/chat/completions', {
        'model':config['model_alias'], 'messages':messages,
        'max_tokens':config['max_answer_tokens'], 'temperature':0,
        'stream':False, 'chat_template_kwargs':{'enable_thinking':False},
    })
    elapsed = perf_counter() - started
    choice = response['choices'][0]
    answer = choice['message'].get('content') or ''
    passed = 'келар' in answer.casefold() and choice.get('finish_reason') == 'stop'
    report = {'schema_version':1, 'scope':'Проверка генерации LLM по тестовому контексту, без FAISS и базы знаний.',
              'model_config':config, 'messages':messages, 'answer':answer,
              'finish_reason':choice.get('finish_reason'), 'usage':response.get('usage'),
              'request_seconds':round(elapsed, 6), 'passed':passed}
    output = ROOT / 'reports/llm_smoke.json'
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(f'Ответ модели: {answer}')
    print(f'Время запроса: {elapsed:.2f} с. Результат: {output}')
    print('Проверка генерации пройдена.' if passed else 'Проверка не пройдена: требуется разбор ответа и причины завершения.')
    print('Поиск по базе знаний к этому запросу ещё не подключён.')
    return 0 if passed else 1


if __name__ == '__main__':
    try:
        sys.exit(main())
    except urllib.error.HTTPError as error:
        if error.code == 503:
            print('Сервер ещё загружает модель. Дождитесь загрузки и повторите проверку.',file=sys.stderr)
        else:
            print(f'Сервер вернул HTTP {error.code}: {error.read(2000).decode(errors="replace")}',file=sys.stderr)
        sys.exit(1)
    except (OSError, ValueError, KeyError, IndexError, urllib.error.URLError) as error:
        print(f'Проверка не завершена: {error}\nУбедитесь, что сервер запущен в другом терминале.',file=sys.stderr)
        sys.exit(1)

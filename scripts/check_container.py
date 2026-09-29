"""Проверка Linux-контейнера: зависимости, индекс, LLM, ответ и отказ."""
import argparse
from importlib.metadata import distributions
import json
import os
from pathlib import Path
import platform
import sys

from rag_pipeline import RAGBot, display_result
from embedding_settings import ROOT
from vector_store import file_digest
from check_index_environment import main as check_environment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'reports/docker_smoke.json')
    args = parser.parse_args()
    if platform.system() != 'Linux' or os.environ.get('RAG_CONTAINER') != '1':
        raise ValueError('Запустите проверку через docker compose run --rm rag-bot python scripts/check_container.py')
    if check_environment() != 0:
        raise ValueError('Проверка зависимостей контейнера не пройдена')
    bot = RAGBot()
    report = {
        'schema_version':2, 'scope':'Два запроса из Linux-контейнера к внешнему локальному llama.cpp; не нагрузочное испытание.',
        'platform':platform.platform(), 'python':platform.python_version(),
        'machine':platform.machine(), 'llm_endpoint':bot.client.base_url,
        'llm_config':bot.client.config, 'server_context':bot.client.server_context,
        'embedding_config':bot.retriever.config, 'index_vectors':bot.retriever.index.ntotal,
        'packages':{d.metadata['Name']:d.version for d in distributions() if d.metadata['Name']},
        'input_sha256':{name:file_digest(ROOT/name) for name in [
            'Dockerfile', 'docker-compose.yml', 'requirements.txt', 'index/manifest.json',
            'index/faiss.index', 'index/metadata.json', 'config/llm_config.json',
            'config/embedding_config.json', 'prompts/rag_system.txt', 'prompts/few_shot.json',
            'prompts/document_policy.txt', 'scripts/llm_client.py', 'scripts/rag_pipeline.py',
            'scripts/document_security.py', 'scripts/vector_store.py', 'scripts/check_container.py']},
        'results':[], 'passed':False,
    }
    cases = json.loads((ROOT/'data/rag_cases.json').read_text(encoding='utf-8'))
    # Один ответ с источником и один отказ из уже используемого набора.
    for case in (cases[0], cases[-1]):
        print('Вопрос: '+case['question'], flush=True)
        result = bot.ask(case['question'])
        display_result(result)
        passed = result['status'] == case['expected_status']
        if case['expected_status'] == 'answered':
            passed = (passed and bool({s['chunk_id'] for s in result['sources']} & set(case['expected_chunk_ids']))
                      and any(word.casefold() in result['answer'].casefold() for word in case['answer_contains']))
        report['results'].append({'case':case, 'passed':passed, 'result':result})
    report['passed'] = all(row['passed'] for row in report['results'])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(f"Проверка контейнера: {sum(x['passed'] for x in report['results'])}/2. Лог: {args.output}")
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (OSError, ValueError, KeyError, IndexError) as error:
        print(f'Проверка контейнера не завершена: {error}', file=sys.stderr)
        sys.exit(1)

"""Пять вопросов по базе и два вопроса без ответа, с сохранением диалогов."""
import argparse
import json
from pathlib import Path
import sys
from rag_pipeline import RAGBot, display_result
from embedding_settings import ROOT
from vector_store import file_digest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'reports/rag_examples.json')
    args=parser.parse_args()
    cases=json.loads((ROOT/'data/rag_cases.json').read_text(encoding='utf-8'))
    bot=RAGBot()
    results=[]
    for case in cases:
        print('Вопрос: '+case['question'],flush=True)
        try:
            result=bot.ask(case['question'])
            source_ids={c['chunk_id'] for c in result['sources']}
            passed=result['status']==case['expected_status']
            if case['expected_status']=='answered':
                passed=passed and bool(source_ids & set(case['expected_chunk_ids']))
                passed=passed and any(x.casefold() in result['answer'].casefold() for x in case['answer_contains'])
            display_result(result)
            print('ПРОЙДЕНО' if passed else 'ТРЕБУЕТ ПРОВЕРКИ',flush=True)
            results.append({'case':case,'passed':passed,'result':result})
        except (OSError, ValueError, KeyError, IndexError) as error:
            print(f'ОШИБКА: {error}',flush=True)
            results.append({'case':case,'passed':False,'error':str(error)})
    hashes={str(p):file_digest(ROOT/p) for p in [Path('index/manifest.json'),Path('prompts/rag_system.txt'),Path('prompts/few_shot.json'),Path('data/rag_cases.json'),Path('config/llm_config.json'),Path('prompts/document_policy.txt'),Path('scripts/rag_pipeline.py'),Path('scripts/document_security.py'),Path('scripts/llm_client.py')]}
    report={'schema_version':2,'llm_config':bot.client.config,'embedding_config':bot.retriever.config,
            'input_sha256':hashes,'case_count':len(results),'passed_count':sum(r['passed'] for r in results),
            'limitations':'Автоматическая проверка структуры трёх шагов, статуса, ключевых слов и ссылок дополнена дословной проверкой цитат. Смысл ответа и объяснения требует ручной оценки.',
            'results':results}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(f"Итог: {report['passed_count']}/{len(results)}. Лог: {args.output}")
    return 0 if all(r['passed'] for r in results) else 1


if __name__=='__main__':
    try:
        sys.exit(main())
    except (OSError, ValueError, KeyError, IndexError) as error:
        print(f'Проверка RAG не выполнена: {error}',file=sys.stderr)
        sys.exit(1)

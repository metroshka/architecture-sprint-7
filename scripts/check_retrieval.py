"""Три контрольных запроса и сохранение фактических результатов поиска."""
import argparse
import json
from pathlib import Path
import sys
from embedding_settings import ROOT
from vector_store import Retriever, INDEX_PATH, file_digest

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--index-dir', type=Path, default=INDEX_PATH)
    parser.add_argument('--output', type=Path, default=ROOT/'reports/retrieval_examples.json')
    args = parser.parse_args()
    try:
        cases_path = ROOT/'data/retrieval_cases.json'
        cases = json.loads(cases_path.read_text(encoding='utf-8'))
        retriever = Retriever(args.index_dir)
        results = []
        for case in cases:
            result = retriever.search(case['question'], 3)
            hit_ids = [hit['chunk_id'] for hit in result['hits']]
            # Проверка разметки: ожидаемый фрагмент действительно содержит опорную фразу.
            for expected in case['expected_chunk_ids']:
                if not any(c['chunk_id'] == expected and case['evidence'] in c['text'] for c in retriever.chunks):
                    raise ValueError(f'Устарела разметка: {expected}')
            rank = next((i for i, cid in enumerate(hit_ids, 1) if cid in case['expected_chunk_ids']), None)
            result.update(expected_chunk_ids=case['expected_chunk_ids'], evidence=case['evidence'],
                          first_relevant_rank=rank, hit_at_3=rank is not None)
            results.append(result)
            print(f"{'НАЙДЕНО' if rank else 'НЕ НАЙДЕНО'}: {case['question']} | {', '.join(hit_ids)}")
        passed = sum(r['hit_at_3'] for r in results)
        report = {'schema_version': 1, 'index_manifest_sha256': file_digest(args.index_dir/'manifest.json'),
                  'cases_sha256': file_digest(cases_path), 'query_count': len(results),
                  'hit_at_3_count': passed, 'hit_at_3': passed/len(results),
                  'scope': 'Три контрольных запроса; не оценка качества всех вопросов и не проверка LLM.',
                  'results': results}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
        print(f'Найден ожидаемый фрагмент в top-3: {passed}/{len(results)}. Лог: {args.output}')
        sys.exit(0 if passed == len(results) else 1)
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        print(f'Проверка не выполнена: {error}', file=sys.stderr)
        sys.exit(1)

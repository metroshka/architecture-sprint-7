"""Поиск фрагментов без генерации ответа LLM."""
import argparse
import sys
from pathlib import Path
from vector_store import Retriever, INDEX_PATH

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('question')
    parser.add_argument('--top-k', type=int, default=3)
    parser.add_argument('--index-dir', type=Path, default=INDEX_PATH)
    args = parser.parse_args()
    try:
        result = Retriever(args.index_dir).search(args.question, args.top_k)
        for hit in result['hits']:
            print(f"\n{hit['rank']}. {hit['chunk_id']} | {hit['title']} | сходство {hit['score']:.4f}")
            print(f"Источник: {hit['source']} [{hit['start_char']}:{hit['end_char']}]")
            print(hit['text'])
        print(f"\nКодирование вопроса: {result['query_embedding_seconds']:.6f} с; поиск FAISS: {result['faiss_search_seconds']:.6f} с.")
        print('Сходство не является вероятностью правильного ответа. Найденные фрагменты ещё не являются ответом LLM.')
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        print(f'Поиск не выполнен: {error}', file=sys.stderr)
        sys.exit(1)

"""Проверка зависимостей индексации без скачивания модели."""

from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
import platform
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    print(f'Python: {platform.python_version()}; платформа: {platform.system()} {platform.machine()}', flush=True)
    if sys.version_info[:2] != (3, 11):
        print('Для проекта требуется Python 3.11.', file=sys.stderr)
        return 1
    problems = []
    for line in (ROOT / 'requirements.txt').read_text(encoding='utf-8').splitlines():
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        name, expected = line.split('==', 1)
        try:
            actual = version(name)
        except PackageNotFoundError:
            problems.append(f'{name}: не установлен')
            continue
        print(f'{name}: {actual}', flush=True)
        if actual.split('+', 1)[0] != expected:
            problems.append(f'{name}: требуется {expected}, установлена {actual}')
    if problems:
        print('\n'.join(problems), file=sys.stderr)
        print('Выполните: python -m pip install -r requirements.txt', file=sys.stderr)
        return 1
    check = subprocess.run([sys.executable, '-m', 'pip', 'check'], capture_output=True, text=True)
    if check.returncode:
        print(check.stdout + check.stderr, file=sys.stderr)
        return 1
    try:
        import faiss
        import numpy as np
        import torch
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        from sentence_transformers import SentenceTransformer
        from transformers import AutoTokenizer

        # Синтетические векторы проверяют совместимость библиотек, а не качество поиска.
        vectors = np.ascontiguousarray(torch.eye(3, dtype=torch.float32).numpy())
        index = faiss.IndexFlatIP(3)
        index.add(vectors)
        scores, positions = index.search(vectors[1:2], 1)
        if positions.tolist() != [[1]] or not np.isclose(scores[0, 0], 1.0):
            raise RuntimeError('FAISS вернул неожиданный результат на тестовых векторах')
        splitter = RecursiveCharacterTextSplitter(chunk_size=40, chunk_overlap=5)
        fragments = splitter.split_text('First paragraph about a planet.\n\nSecond paragraph about a ship.')
        if len(fragments) < 2 or any(len(fragment) > 40 for fragment in fragments):
            raise RuntimeError('Проверка разбиения текста не пройдена')
        if not callable(SentenceTransformer) or not callable(AutoTokenizer.from_pretrained):
            raise RuntimeError('Не удалось загрузить интерфейсы модели и токенизатора')
    except Exception as error:
        print(f'Проверка библиотек не пройдена: {type(error).__name__}: {error}', file=sys.stderr)
        return 1
    print('Зависимости согласованы; импорты выполнены.')
    print('FAISS: тестовый поиск пройден. LangChain: разбиение текста проверено.')
    print('Окружение для индексации готово. Модель не загружалась, индекс базы знаний не создавался.')
    return 0


if __name__ == '__main__':
    sys.exit(main())

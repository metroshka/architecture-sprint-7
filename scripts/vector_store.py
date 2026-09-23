"""Локальное кодирование и проверенная загрузка FAISS с JSON-метаданными."""
import hashlib
import json
from pathlib import Path
import re
import os
import platform
from time import perf_counter

# Режим текущего процесса: избегаем наблюдаемого сбоя OpenMP на macOS.
if platform.system() == 'Darwin':
    os.environ['OMP_NUM_THREADS'] = '1'

import faiss

import numpy as np
from embedding_settings import ROOT, MODEL_PATH, load_config, digest, passage_input, token_count

INDEX_PATH = ROOT / 'index'


def file_digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_chunks(payload, config, root=ROOT, tokenizer=None):
    if payload.get('schema_version') != 1 or payload.get('embedding_config') != config:
        raise ValueError('Параметры фрагментов не совпадают с конфигурацией; пересоздайте фрагменты и индекс')
    chunks = payload.get('chunks', [])
    if not chunks or payload.get('chunk_count') != len(chunks):
        raise ValueError('Неверное число фрагментов')
    documents, ids, coverage = {}, set(), {}
    for c in chunks:
        source = c['source']
        if not re.fullmatch(r'knowledge_base/[^/\\]+\.md', source):
            raise ValueError('Недопустимый путь документа')
        if source not in documents:
            documents[source] = (root / source).read_text(encoding='utf-8')
            coverage[source] = set()
        text = documents[source]
        if digest(text) != c['document_sha256']:
            raise ValueError(f'Документ изменён: {source}; пересоздайте фрагменты и индекс')
        start, end = c['start_char'], c['end_char']
        if not 0 <= start < end <= len(text) or text[start:end] != c['text']:
            raise ValueError('Текст фрагмента не соответствует координатам')
        if c['title'] != text.split('\n', 1)[0].removeprefix('# ').strip():
            raise ValueError('Заголовок фрагмента не соответствует документу')
        if c['chunk_id'] in ids:
            raise ValueError('Повторяющийся идентификатор фрагмента')
        ids.add(c['chunk_id'])
        coverage[source].update(range(start, end))
        if c['embedding_text'] != passage_input(c['title'], c['text'], config):
            raise ValueError('Изменён вход embedding-модели')
        if not 0 < c['input_tokens'] <= config['input_token_budget']:
            raise ValueError('Превышен лимит токенов')
        if tokenizer is not None and token_count(tokenizer, c['embedding_text']) != c['input_tokens']:
            raise ValueError('Число токенов не соответствует токенизатору')
    actual_sources = {f'knowledge_base/{p.name}' for p in (root / 'knowledge_base').glob('*.md')}
    if actual_sources != set(documents) or len(documents) != payload.get('document_count'):
        raise ValueError('Состав базы знаний изменился; пересоздайте фрагменты и индекс')
    for source, text in documents.items():
        if any(i not in coverage[source] for i in range(text.index('\n') + 1, len(text)) if not text[i].isspace()):
            raise ValueError('Фрагменты не покрывают весь текст документа')
    return chunks


class Encoder:
    def __init__(self, config):
        from sentence_transformers import SentenceTransformer
        self.config = config
        manifest = json.loads((MODEL_PATH / 'download_manifest.json').read_text(encoding='utf-8'))
        if any(manifest.get(k) != config[k] for k in ['model_id', 'revision', 'dimension', 'max_tokens']):
            raise ValueError('Локальная модель не соответствует конфигурации')
        if not config['normalize_embeddings'] or not manifest.get('smoke_test_passed'):
            raise ValueError('Требуются нормализованные векторы и проверенная модель')
        self.model = SentenceTransformer(str(MODEL_PATH), device=config['device'],
                                        local_files_only=True, trust_remote_code=False)
        self.tokenizer = self.model.tokenizer
        if self.model.get_embedding_dimension() != config['dimension'] or self.model.max_seq_length != config['max_tokens']:
            raise ValueError('Размерность или лимит локальной модели изменились')

    def encode(self, texts):
        if not texts or any(not t.strip() or token_count(self.tokenizer, t) > self.config['max_tokens'] for t in texts):
            raise ValueError('Пустой вход или превышение лимита модели; обрезание отключено')
        vectors = self.model.encode(texts, batch_size=8, normalize_embeddings=True,
                                    convert_to_numpy=True, show_progress_bar=False)
        vectors = np.ascontiguousarray(vectors, dtype='float32')
        if vectors.shape != (len(texts), self.config['dimension']) or not np.isfinite(vectors).all():
            raise ValueError('Некорректные векторы модели')
        if not np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-5):
            raise ValueError('Векторы не нормализованы')
        return vectors


def load_index(directory=INDEX_PATH, config=None, root=ROOT):
    config = load_config() if config is None else config
    directory = Path(directory)
    manifest = json.loads((directory / 'manifest.json').read_text(encoding='utf-8'))
    if manifest.get('schema_version') != 1 or manifest.get('embedding_config') != config:
        raise ValueError('Конфигурация отличается от параметров индекса; пересоздайте индекс')
    for name in ['faiss.index', 'metadata.json']:
        if file_digest(directory / name) != manifest['files_sha256'].get(name):
            raise ValueError(f'Нарушена целостность {name}; пересоздайте индекс')
    payload = json.loads((directory / 'metadata.json').read_text(encoding='utf-8'))
    chunks = validate_chunks(payload, config, root)
    index = faiss.read_index(str(directory / 'faiss.index'))
    if (not isinstance(index, faiss.IndexFlatIP) or index.d != config['dimension']
            or index.ntotal != len(chunks) or index.ntotal != manifest['chunk_count']):
        raise ValueError('Структура индекса не соответствует метаданным')
    return index, chunks, manifest


class Retriever:
    def __init__(self, directory=INDEX_PATH):
        self.config = load_config()
        self.index, self.chunks, self.manifest = load_index(directory, self.config)
        self.encoder = Encoder(self.config)

    def search(self, question, top_k=3):
        question = question.strip()
        if not question or not 1 <= top_k <= 20:
            raise ValueError('Нужен непустой вопрос и top_k от 1 до 20')
        started = perf_counter()
        vector = self.encoder.encode([self.config['query_prefix'] + question])
        encoded = perf_counter()
        scores, positions = self.index.search(vector, min(top_k, self.index.ntotal))
        searched = perf_counter()
        hits = [dict(self.chunks[int(pos)], rank=rank, score=float(score))
                for rank, (score, pos) in enumerate(zip(scores[0], positions[0]), 1)]
        return {'question': question, 'top_k': top_k, 'hits': hits,
                'query_embedding_seconds': round(encoded - started, 6),
                'faiss_search_seconds': round(searched - encoded, 6)}

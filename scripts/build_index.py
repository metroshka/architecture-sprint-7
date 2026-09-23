"""Вычисление векторов фрагментов и сохранение IndexFlatIP."""
import argparse
from importlib.metadata import version
import json
from pathlib import Path
import platform
import sys
import tempfile
from time import perf_counter

import os
from embedding_settings import ROOT, load_config
from vector_store import faiss, Encoder, INDEX_PATH, file_digest, validate_chunks, load_index


def build_index(output=INDEX_PATH):
    started = perf_counter()
    config = load_config()
    payload = json.loads((ROOT / 'data/chunks.json').read_text(encoding='utf-8'))
    validate_chunks(payload, config)
    print('Загрузка локальной модели на CPU…', flush=True)
    loading = perf_counter()
    encoder = Encoder(config)
    loaded = perf_counter()
    import torch
    chunks = validate_chunks(payload, config, tokenizer=encoder.tokenizer)
    print(f'Вычисление векторов: {len(chunks)} фрагментов…', flush=True)
    encoding = perf_counter()
    vectors = encoder.encode([c['embedding_text'] for c in chunks])
    encoded = perf_counter()
    index = faiss.IndexFlatIP(config['dimension'])
    index.add(vectors)
    indexed = perf_counter()
    manifest = {
        'schema_version': 1, 'index_type': 'IndexFlatIP', 'metric': 'cosine_on_normalized_vectors',
        'embedding_config': config, 'document_count': payload['document_count'],
        'chunk_count': len(chunks), 'vector_dtype': 'float32', 'batch_size': 8,
        'python': platform.python_version(), 'platform': platform.system(), 'machine': platform.machine(),
        'cpu_threads': torch.get_num_threads(),
        'omp_num_threads': os.environ.get('OMP_NUM_THREADS'),
        'packages': {p: version(p) for p in ['sentence-transformers', 'transformers', 'torch', 'numpy', 'faiss-cpu']},
        'timings_seconds': {'model_load': round(loaded-loading, 6),
                            'embedding_generation': round(encoded-encoding, 6),
                            'faiss_add': round(indexed-encoded, 6)},
    }
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    saving = perf_counter()
    with tempfile.TemporaryDirectory(prefix='.index-build-', dir=output.parent) as temp:
        staging = Path(temp)
        faiss.write_index(index, str(staging / 'faiss.index'))
        metadata = {k: payload[k] for k in ['schema_version', 'embedding_config', 'document_count', 'chunk_count', 'chunks']}
        (staging / 'metadata.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        manifest['files_sha256'] = {name: file_digest(staging / name) for name in ['faiss.index', 'metadata.json']}
        manifest['timings_seconds']['data_save_and_hash'] = round(perf_counter()-saving, 6)
        (staging / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
        restored, _, _ = load_index(staging, config)
        scores, _ = restored.search(vectors, 1)
        import numpy as np
        if not np.allclose(scores[:, 0], 1.0, atol=1e-5):
            raise ValueError('Самопоиск после сохранения не прошёл проверку')
        output.mkdir(exist_ok=True)
        # Manifest переносится последним: прерванная замена обнаруживается по хешам.
        for name in ['faiss.index', 'metadata.json', 'manifest.json']:
            (staging / name).replace(output / name)
    print(f'Готово: {len(chunks)} векторов размерности {config["dimension"]}. Каталог: {output}')
    print(f'Генерация эмбеддингов: {encoded-encoding:.3f} с; добавление в FAISS: {indexed-encoded:.6f} с.')
    print(f'Сохранение, повторная загрузка и самопоиск проверены. Всего в функции построения: {perf_counter()-started:.3f} с.')
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=INDEX_PATH)
    args = parser.parse_args()
    try:
        build_index(args.output_dir)
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        print(f'Индекс не создан: {error}', file=sys.stderr)
        sys.exit(1)

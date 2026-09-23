"""Загрузка закреплённой версии E5 и проверка вычисления эмбеддингов."""
import json
import sys

from embedding_settings import MODEL_PATH, load_config, passage_input, token_count


def main():
    config = load_config()
    print(f"Загрузка: {config['model_id']}\nВерсия: {config['revision']}", flush=True)
    from huggingface_hub import snapshot_download
    # Загружаем один формат весов, без дубликатов ONNX/OpenVINO/PyTorch bin.
    snapshot_download(
        repo_id=config['model_id'], revision=config['revision'],
        local_dir=MODEL_PATH,
        allow_patterns=['config.json', 'model.safetensors', 'modules.json',
                        'sentence_bert_config.json', 'tokenizer.json',
                        'tokenizer_config.json', 'special_tokens_map.json',
                        'sentencepiece.bpe.model', '1_Pooling/config.json', 'README.md'],
    )
    print('Файлы загружены. Проверка модели на CPU…', flush=True)
    import numpy as np
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(str(MODEL_PATH), device=config['device'],
                                local_files_only=True, trust_remote_code=False)
    if model.get_embedding_dimension() != config['dimension'] or model.max_seq_length != config['max_tokens']:
        raise ValueError('Размерность или лимит модели не соответствует конфигурации')
    inputs = [passage_input('Test planet', 'The planet has two moons.', config),
              config['query_prefix'] + 'How many moons does the planet have?']
    if any(token_count(model.tokenizer, text) > config['max_tokens'] for text in inputs):
        raise ValueError('Тестовый вход превышает лимит модели')
    vectors = model.encode(inputs, normalize_embeddings=config['normalize_embeddings'],
                           convert_to_numpy=True, show_progress_bar=False)
    if vectors.shape != (2, config['dimension']) or not np.isfinite(vectors).all():
        raise ValueError('Неверная форма или значения эмбеддингов')
    if not np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-5):
        raise ValueError('Эмбеддинги не нормализованы')
    (MODEL_PATH / 'download_manifest.json').write_text(json.dumps({
        'model_id': config['model_id'], 'revision': config['revision'],
        'dimension': config['dimension'], 'max_tokens': config['max_tokens'],
        'smoke_test_passed': True,
    }, indent=2) + '\n', encoding='utf-8')
    print(f"Модель готова: {config['dimension']} чисел в векторе; лимит {config['max_tokens']} токенов.")
    print('Тест вычисления и нормализации пройден. Индекс базы знаний ещё не создан.')
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (OSError, ValueError, RuntimeError) as error:
        print(f'Подготовка модели не завершена: {error}', file=sys.stderr)
        sys.exit(1)

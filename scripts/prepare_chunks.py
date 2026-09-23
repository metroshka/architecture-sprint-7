"""Разбиение только knowledge_base/*.md с проверкой длины и координат."""
import json
import sys
from time import perf_counter

from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer
from embedding_settings import ROOT, MODEL_PATH, load_config, digest, passage_input, token_count


def split_document(content, source, tokenizer, config):
    heading, separator, body = content.partition('\n')
    if not heading.startswith('# ') or not separator or not body.strip():
        raise ValueError(f'{source}: требуется заголовок и непустой текст')
    title = heading[2:].strip()
    body_offset = len(heading) + 1
    splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        tokenizer, chunk_size=config['chunk_tokens'], chunk_overlap=config['overlap_tokens'],
        separators=['\n\n', '\n', '. ', '; ', ' ', ''], keep_separator='end',
    )
    chunks = []
    previous_start, previous_end = -1, 0
    for number, text in enumerate(splitter.split_text(body), 1):
        # Смещения измеряются символами; token-overlap нельзя вычитать из char-offset.
        start = body.find(text, previous_start + 1)
        while start >= 0:
            overlap = body[start:previous_end] if start < previous_end else ''
            if (start + len(text) > previous_end
                    and len(tokenizer.tokenize(overlap)) <= config['overlap_tokens']):
                break
            start = body.find(text, start + 1)
        if start < 0 or body[previous_end:start].strip():
            raise ValueError(f'{source}: невозможно восстановить координаты без потери текста')
        end = start + len(text)
        embedding_text = passage_input(title, text, config)
        count = token_count(tokenizer, embedding_text)
        if count > config['input_token_budget']:
            raise ValueError(f'{source}: вход содержит {count} токенов; уменьшите chunk_tokens или заголовок')
        chunks.append({
            'chunk_id': f'{source.removesuffix(".md").split("/")[-1]}_{number:03d}',
            'source': source, 'title': title, 'chunk_number': number,
            'start_char': body_offset + start, 'end_char': body_offset + end,
            'document_sha256': digest(content), 'text': text,
            'embedding_text': embedding_text, 'input_tokens': count,
        })
        previous_start, previous_end = start, end
    if not chunks or body[previous_end:].strip():
        raise ValueError(f'{source}: часть текста не попала во фрагменты')
    return chunks


def prepare_corpus(directory, tokenizer, config):
    files = sorted(directory.glob('*.md'))
    if not files:
        raise ValueError('В knowledge_base нет Markdown-документов')
    chunks = []
    for path in files:
        source = f'knowledge_base/{path.name}'
        document_chunks = split_document(path.read_text(encoding='utf-8'), source, tokenizer, config)
        chunks.extend(document_chunks)
        print(f'{path.name}: {len(document_chunks)} фрагм., максимум {max(c["input_tokens"] for c in document_chunks)} токенов')
    return chunks, len(files)


def main():
    config = load_config()
    manifest_path = MODEL_PATH / 'download_manifest.json'
    if not manifest_path.exists():
        raise ValueError('Сначала выполните python scripts/download_embedding_model.py')
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    if any(manifest.get(key) != config[key] for key in ['model_id', 'revision', 'dimension', 'max_tokens']):
        raise ValueError('Локальная модель не соответствует конфигурации; повторите загрузку')
    if not manifest.get('smoke_test_passed'):
        raise ValueError('Модель не прошла проверку вычислений')
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_PATH), local_files_only=True, trust_remote_code=False)
    started = perf_counter()
    chunks, document_count = prepare_corpus(ROOT / 'knowledge_base', tokenizer, config)
    result = {'schema_version': 1, 'embedding_config': config, 'document_count': document_count,
              'chunk_count': len(chunks), 'max_input_tokens': max(c['input_tokens'] for c in chunks),
              'preparation_seconds': round(perf_counter() - started, 4), 'chunks': chunks}
    output = ROOT / 'data' / 'chunks.json'
    # Все проверки завершены до замены результата предыдущего запуска.
    temporary = output.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    temporary.replace(output)
    print(f'Готово: {document_count} документов, {len(chunks)} фрагментов. Сохранено: {output}')
    print('Векторы документов и индекс ещё не созданы.')
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (OSError, ValueError) as error:
        print(f'Подготовка фрагментов не завершена: {error}', file=sys.stderr)
        sys.exit(1)

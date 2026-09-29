"""Разбиение только knowledge_base/*.md с проверкой длины и координат."""
import json
import math
import re
from functools import lru_cache
import sys
from time import perf_counter

from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer
from embedding_settings import ROOT, MODEL_PATH, load_config, digest, passage_input, token_count, is_security_fixture


def split_document(content, source, tokenizer, config, *, allow_security_fixture=False):
    heading, separator, body = content.partition('\n')
    if not heading.startswith('# ') or not separator or not body.strip():
        raise ValueError(f'{source}: требуется заголовок и непустой текст')
    title = heading[2:].strip()
    body_offset = len(heading) + 1
    minimum, maximum = config['min_chunk_words'], config['max_chunk_words']
    if token_count(tokenizer, passage_input(title, '', config)) > config['input_token_budget']:
        raise ValueError(f'{source}: заголовок превышает лимит токенов')
    exception = allow_security_fixture and is_security_fixture(content, source)
    if allow_security_fixture and not exception:
        raise ValueError('Исключение разрешено только для неизменного файла проверки prompt injection')
    words = list(re.finditer(r'\S+', body))
    total = len(words)
    if total < minimum and not exception:
        raise ValueError(f'{source}: {total} слов; минимум {minimum}. Дополните исходный документ содержательным текстом')

    # LangChain предлагает исходные границы; подбор ниже перераспределяет короткие
    # остатки и проверяет полный вход E5, включая заголовок и префикс.
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=maximum, chunk_overlap=0, length_function=lambda text: len(text.split()),
        separators=['\n\n', '\n', '. ', '; ', ' ', ''], keep_separator='end', add_start_index=True,
    )
    proposed = {d.metadata['start_index'] + len(d.page_content)
                for d in splitter.create_documents([body])}
    boundary_priority = {}
    for end in range(1, total + 1):
        gap = body[words[end-1].end():words[end].start()] if end < total else ''
        boundary_priority[end] = (0 if '\n\n' in gap else
                                  1 if re.search(r'[.!?]["\'’”)]*$', words[end-1].group()) else
                                  2 if words[end-1].end() in proposed else 3)

    def text_between(start, end):
        return body[words[start].start():words[end-1].end()]

    @lru_cache(maxsize=None)
    def fits(start, end):
        return token_count(tokenizer, passage_input(title, text_between(start, end), config)) <= config['input_token_budget']

    @lru_cache(maxsize=None)
    def partition(start, parts):
        remaining = total - start
        if not parts * minimum <= remaining <= parts * maximum:
            return None
        if parts == 1:
            return (total,) if fits(start, total) else None
        low = max(start + minimum, total - (parts-1) * maximum)
        high = min(start + maximum, total - (parts-1) * minimum)
        target = start + remaining / parts
        # Предпочитаем абзацы и целые предложения, затем близость к равным долям.
        ends = sorted(range(low, high+1), key=lambda end: (boundary_priority[end], abs(end-target), end))
        for end in ends:
            if fits(start, end):
                tail = partition(end, parts-1)
                if tail is not None:
                    return (end,) + tail
        return None

    boundaries = (total,) if exception and total < minimum and fits(0, total) else None
    if boundaries is None:
        for parts in range(math.ceil(total / maximum), total // minimum + 1):
            boundaries = partition(0, parts)
            if boundaries is not None:
                break
    if boundaries is None:
        raise ValueError(f'{source}: невозможно совместить {minimum}–{maximum} слов и лимит токенов; пересмотрите исходный текст или модель')

    chunks, first_word = [], 0
    for number, last_word in enumerate(boundaries, 1):
        start, end = words[first_word].start(), words[last_word-1].end()
        text = body[start:end]
        embedding_text = passage_input(title, text, config)
        chunk = {
            'chunk_id': f'{source.removesuffix(".md").split("/")[-1]}_{number:03d}',
            'source': source, 'title': title, 'chunk_number': number,
            'start_char': body_offset + start, 'end_char': body_offset + end,
            'document_sha256': digest(content), 'text': text, 'word_count': last_word-first_word,
            'embedding_text': embedding_text, 'input_tokens': token_count(tokenizer, embedding_text),
        }
        if exception and total < minimum:
            chunk['short_document_exception'] = 'prompt_injection_fixture'
        chunks.append(chunk)
        first_word = last_word
    return chunks


def prepare_corpus(directory, tokenizer, config, *, allow_security_fixture=False):
    files = sorted(directory.glob('*.md'))
    if not files:
        raise ValueError('В knowledge_base нет Markdown-документов')
    chunks = []
    for path in files:
        source = f'knowledge_base/{path.name}'
        document_chunks = split_document(
            path.read_text(encoding='utf-8'), source, tokenizer, config,
            allow_security_fixture=allow_security_fixture and source == 'knowledge_base/injection_probe.md')
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
              'min_chunk_words': min(c['word_count'] for c in chunks),
              'max_chunk_words': max(c['word_count'] for c in chunks),
              'preparation_seconds': round(perf_counter() - started, 4), 'chunks': chunks}
    output = ROOT / 'data' / 'chunks.json'
    # Все проверки завершены до замены результата предыдущего запуска.
    temporary = output.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    temporary.replace(output)
    print(f'Готово: {document_count} документов, {len(chunks)} фрагментов. Сохранено: {output}')
    print(f"Размеры фрагментов: {result['min_chunk_words']}–{result['max_chunk_words']} слов; максимум {result['max_input_tokens']} токенов.")
    print('Теперь пересоздайте индекс: python scripts/build_index.py')
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (OSError, ValueError) as error:
        print(f'Подготовка фрагментов не завершена: {error}', file=sys.stderr)
        sys.exit(1)

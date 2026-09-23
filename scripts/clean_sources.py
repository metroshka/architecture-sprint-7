"""Очистка и редакторский отбор текста по data/cleaning_rules.json."""

import argparse
import json
import re
import sys
from pathlib import Path

from download_sources import ROOT, check_cached, digest, load_sources
from source_html import extract_paragraphs

MEDIA_MARKERS = re.compile(
    r'\b(?:films?|franchise|trilog(?:y|ies)|episodes?|actors?|actresses|'
    r'portray(?:ed|al)?|Lucas|Disney|noveli[sz]ation|television|screenplay)\b',
    re.IGNORECASE,
)


def unique_position(text, fragment):
    if not fragment or text.count(fragment) != 1:
        raise ValueError(f'Ожидалось одно вхождение фрагмента: {fragment!r}')
    return text.index(fragment)


def clean_paragraph(paragraph, rule):
    text = paragraph['text']
    if digest(text) != rule['paragraph_sha256'] or paragraph['section'] != rule['section']:
        raise ValueError('Изменился исходный абзац или раздел; требуется проверка правил отбора')
    if 'start_at' in rule:
        text = text[unique_position(text, rule['start_at']):]
    if 'end_before' in rule:
        text = text[:unique_position(text, rule['end_before'])]
    for replacement in rule.get('replacements', []):
        unique_position(text, replacement['old'])
        text = text.replace(replacement['old'], replacement['new'])
    text = text.strip()
    if not text:
        raise ValueError('После очистки получился пустой абзац')
    return text


def prepare_document(source, raw, rules):
    paragraphs = extract_paragraphs(raw['html'])
    selected = []
    origins = []
    indices = [rule['paragraph_index'] for rule in rules]
    if not rules or len(set(indices)) != len(indices):
        raise ValueError('Пустой или дублирующийся отбор абзацев')
    for rule in rules:
        index = rule['paragraph_index']
        if type(index) is not int or not 0 <= index < len(paragraphs):
            raise ValueError('Индекс абзаца вне диапазона')
        paragraph = paragraphs[index]
        text = clean_paragraph(paragraph, rule)
        selected.append(text)
        origins.append({
            'paragraph_index': index, 'section': paragraph['section'],
            'original_sha256': digest(paragraph['text']), 'cleaned_sha256': digest(text),
        })
    content = '\n\n'.join(selected)
    markers = sorted(set(MEDIA_MARKERS.findall(content)))
    if markers:
        raise ValueError(f'Остались маркеры сведений о произведениях: {markers}')
    word_count = len(content.split())
    if word_count < 100:
        raise ValueError(f'Слишком короткий документ: {word_count} слов; нужен пересмотр отбора')
    markdown = f"# {source['title']}\n\n{content}\n"
    return markdown, {
        'id': source['id'], 'source': source,
        'raw_html_sha256': raw['html_sha256'], 'document_sha256': digest(markdown),
        'original_paragraph_count': len(paragraphs),
        'selected_paragraph_count': len(selected), 'word_count': word_count,
        'paragraphs': origins,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw-dir', type=Path, default=ROOT / 'data/raw')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'data/cleaned')
    args = parser.parse_args()
    try:
        if args.raw_dir.resolve() == args.output_dir.resolve():
            raise ValueError('Каталоги исходников и результатов должны различаться')
        sources = load_sources(ROOT / 'data/sources.json')
        rules_path = ROOT / 'data/cleaning_rules.json'
        rules_text = rules_path.read_text(encoding='utf-8')
        specification = json.loads(rules_text)
        rules_by_id = specification['sources']
        if specification['schema_version'] != 1 or set(rules_by_id) != {s['id'] for s in sources}:
            raise ValueError('Правила очистки не соответствуют списку источников')
        documents = []
        for source in sources:
            path = args.raw_dir / (source['id'] + '.json')
            check_cached(path, source)
            raw = json.loads(path.read_text(encoding='utf-8'))
            try:
                markdown, metadata = prepare_document(source, raw, rules_by_id[source['id']])
            except (ValueError, KeyError, TypeError) as error:
                raise ValueError(f"{source['id']}: {error}") from error
            documents.append((source['id'], markdown, metadata))
        # Все входные данные проверяются до записи результатов.
        args.output_dir.mkdir(parents=True, exist_ok=True)
        expected = {identifier + '.md' for identifier, _, _ in documents}
        unexpected = {p.name for p in args.output_dir.glob('*.md')} - expected
        if unexpected:
            raise ValueError(f'В каталоге результатов есть посторонние документы: {sorted(unexpected)}')
        for number, (identifier, markdown, metadata) in enumerate(documents, 1):
            path = args.output_dir / (identifier + '.md')
            temporary = path.with_suffix('.md.tmp')
            temporary.write_text(markdown, encoding='utf-8')
            temporary.replace(path)
            print(f"[{number}/{len(documents)}] {identifier}: {metadata['word_count']} слов")
        report = {
            'schema_version': 1, 'document_count': len(documents),
            'word_count': sum(d[2]['word_count'] for d in documents),
            'cleaning_rules_sha256': digest(rules_text),
            'documents': [d[2] for d in documents],
        }
        report_path = args.output_dir / 'manifest.json'
        temporary = report_path.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        temporary.replace(report_path)
        print(f"Готово: {len(documents)} документов, {report['word_count']} слов. Каталог: {args.output_dir.resolve()}")
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f'Очистка остановлена: {error}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())

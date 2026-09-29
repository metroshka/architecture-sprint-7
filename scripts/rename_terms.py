"""Формирование knowledge_base из очищенных текстов и terms_map.json."""

import argparse
from collections import Counter
import json
from pathlib import Path
import re
import sys

from download_sources import ROOT, digest, load_sources


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f'Повторяющийся ключ JSON: {key}')
        result[key] = value
    return result


class TermReplacer:
    def __init__(self, terms):
        if not isinstance(terms, dict) or not terms:
            raise ValueError('Словарь замен должен быть непустым объектом')
        self.lookup = {}
        for old, new in terms.items():
            if not isinstance(old, str) or not isinstance(new, str) or not old.strip() or not new.strip():
                raise ValueError('Термины и замены должны быть непустыми строками')
            if old != old.strip() or new != new.strip() or '\n' in old + new:
                raise ValueError('Термины должны быть записаны без краевых пробелов и переносов строк')
            if old.casefold() in self.lookup:
                raise ValueError(f'Неоднозначный регистр ключа: {old}')
            self.lookup[old.casefold()] = (old, new)
        alternatives = '|'.join(re.escape(k) for k in sorted(terms, key=lambda k: (-len(k), k)))
        boundary = r'(?<!\w)(?:' + alternatives + r')(?!\w)'
        self.original_pattern = re.compile(boundary, re.IGNORECASE)
        self.replace_pattern = re.compile(
            r'(?P<article>\b(?:a|an)\s+)?(?P<term>' + boundary + ')', re.IGNORECASE
        )

    def replace(self, text):
        counts = Counter()
        article_changes = 0

        def substitute(match):
            nonlocal article_changes
            original = match.group('term')
            key, replacement = self.lookup[original.casefold()]
            counts[key] += 1
            if original[0].isupper():
                replacement = replacement[0].upper() + replacement[1:]
            elif key[0].isupper() and original[0].islower():
                replacement = replacement[0].lower() + replacement[1:]
            article = match.group('article')
            if article:
                # Произношение буквенных обозначений из текущего словаря.
                first = replacement.split()[0]
                vowel = first[0].lower() in 'aeiou' or first in {'M-42C', 'M-48', 'M-61', 'LF-2400', 'SRA-260'}
                corrected = 'an' if vowel else 'a'
                previous = article.strip()
                if previous[0].isupper():
                    corrected = corrected.capitalize()
                article_changes += int(previous != corrected)
                replacement = corrected + article[len(previous):] + replacement
            return replacement

        result = self.replace_pattern.sub(substitute, text)
        return result, counts, article_changes

    def remaining(self, text):
        return sorted({match.group() for match in self.original_pattern.finditer(text)})


def prepare_corpus(input_dir, sources, replacer):
    manifest_path = input_dir / 'manifest.json'
    manifest_text = manifest_path.read_text(encoding='utf-8')
    manifest = json.loads(manifest_text)
    records = manifest['documents']
    by_id = {record['id']: record for record in records}
    expected = {source['id'] for source in sources}
    if (manifest['schema_version'] != 1 or len(by_id) != len(records)
            or set(by_id) != expected or manifest['document_count'] != len(sources)):
        raise ValueError('Метаданные очищенного корпуса не соответствуют списку источников')
    if {path.stem for path in input_dir.glob('*.md')} != expected:
        raise ValueError('Состав очищенных файлов не соответствует списку источников')
    results = []
    total_words = 0
    for number, source in enumerate(sources, 1):
        record = by_id[source['id']]
        text = (input_dir / (source['id'] + '.md')).read_text(encoding='utf-8')
        if digest(text) != record['document_sha256'] or record['source'] != source:
            raise ValueError(f"Изменён очищенный документ или его источник: {source['id']}")
        prefix = f"# {source['title']}\n\n"
        if not text.startswith(prefix) or len(text[len(prefix):].split()) != record['word_count']:
            raise ValueError(f"Нарушен формат или объём документа: {source['id']}")
        total_words += record['word_count']
        renamed, counts, article_changes = replacer.replace(text)
        remaining = replacer.remaining(renamed)
        if remaining or re.search(r'\bStar\s+Wars\b', renamed, re.IGNORECASE):
            raise ValueError(f"Остались исходные термины в {source['id']}: {remaining}")
        if renamed == text:
            raise ValueError(f"В документе нет замен: {source['id']}")
        filename = f'kb_{number:03d}.md'
        title = renamed.splitlines()[0][2:]
        results.append((filename, renamed, {
            'file': filename, 'title': title, 'source': source,
            'cleaned_document_sha256': digest(text), 'document_sha256': digest(renamed),
            'word_count': len(renamed.split('\n\n', 1)[1].split()),
            'replacement_count': sum(counts.values()), 'term_counts': dict(sorted(counts.items())),
            'article_corrections': article_changes,
        }))
    if total_words != manifest['word_count']:
        raise ValueError('Не совпадает общий объём очищенного корпуса')
    if len({digest(text) for _, text, _ in results}) != len(results):
        raise ValueError('После переименования обнаружены одинаковые документы')
    return results, digest(manifest_text), manifest['cleaning_rules_sha256']


def write_text(path, text):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(text, encoding='utf-8')
    temporary.replace(path)


def attribution(documents):
    lines = [
        '# Источники и лицензия базы знаний', '',
        'Тексты документов базы знаний производны от статей английской Wikipedia. '
        'Авторы исходников — участники Wikipedia; история авторов указана для каждой статьи. '
        'Производные тексты распространяются на условиях '
        '[CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/). '
        'Лицензия в этом разделе относится к текстам базы знаний.', '',
        'Изменения: отбор и очистка фрагментов, удаление сведений о произведениях и актёрах, '
        'замена имён и названий, согласование артиклей. Изображения не используются. '
        'Словарь замен и сведения об источниках хранятся отдельно от индексируемых текстов.', '',
        '| Документ | Название | Исходная статья и версия | Авторы |',
        '|---|---|---|---|',
    ]
    for filename, _, record in documents:
        source = record['source']
        lines.append(f"| {filename} | {record['title']} | [{source['title']}]({source['revision_url']}) | [История]({source['history_url']}) |")
    return '\n'.join(lines) + '\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', type=Path, default=ROOT / 'data/cleaned')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'knowledge_base')
    parser.add_argument('--metadata-dir', type=Path, default=ROOT / 'data')
    args = parser.parse_args()
    try:
        directories = [path.resolve() for path in (args.input_dir, args.output_dir, args.metadata_dir)]
        if len(set(directories)) != 3 or any(directories[1] in path.parents for path in (directories[0], directories[2])):
            raise ValueError('Метаданные и исходники должны храниться вне каталога базы знаний')
        sources = load_sources(ROOT / 'data/sources.json')
        terms_text = (ROOT / 'terms_map.json').read_text(encoding='utf-8')
        terms = json.loads(terms_text, object_pairs_hook=unique_object)
        replacer = TermReplacer(terms)
        documents, input_sha256, rules_sha256 = prepare_corpus(args.input_dir, sources, replacer)
        if rules_sha256 != digest((ROOT / 'data/cleaning_rules.json').read_text(encoding='utf-8')):
            raise ValueError('Изменились правила очистки; сначала пересоздайте очищенные документы')
        expected = {filename for filename, _, _ in documents}
        if args.output_dir.exists() and {p.name for p in args.output_dir.iterdir()} - expected:
            raise ValueError('Каталог базы знаний содержит посторонние файлы; проверьте его содержимое')
        counts = Counter()
        for _, _, record in documents:
            counts.update(record['term_counts'])
        report = {
            'schema_version': 1, 'license': 'CC BY-SA 4.0',
            'document_count': len(documents),
            'word_count': sum(record['word_count'] for _, _, record in documents),
            'terms_map_sha256': digest(terms_text), 'cleaned_manifest_sha256': input_sha256,
            'cleaning_rules_sha256': rules_sha256, 'dictionary_size': len(terms),
            'used_term_count': len(counts), 'unused_terms': sorted(set(terms) - set(counts)),
            'replacement_count': sum(counts.values()),
            'article_corrections': sum(d[2]['article_corrections'] for d in documents),
            'remaining_original_terms': [],
            'documents': [record for _, _, record in documents],
        }
        args.output_dir.mkdir(parents=True, exist_ok=True)
        args.metadata_dir.mkdir(parents=True, exist_ok=True)
        for filename, text, record in documents:
            write_text(args.output_dir / filename, text)
            print(f"{filename}: {record['title']} — {record['replacement_count']} замен")
        write_text(args.metadata_dir / 'knowledge_manifest.json', json.dumps(report, ensure_ascii=False, indent=2) + '\n')
        write_text(args.metadata_dir / 'ATTRIBUTION.md', attribution(documents))
        print(f"Готово: {len(documents)} документов, {report['replacement_count']} замен. Остаточных терминов из словаря: 0.")
        print(f'База знаний: {args.output_dir.resolve()}')
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f'Переименование остановлено: {error}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())

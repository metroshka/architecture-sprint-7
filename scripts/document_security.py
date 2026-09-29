"""Фильтрация явных команд и значений учётных данных в найденных фрагментах."""
import re
import unicodedata

MODES = ('off', 'system', 'filtered')
BLOCKED = 'Ответ не выдан: обнаружены потенциально опасные инструкции или чувствительные данные.'
RULES = {
    'override_instructions': re.compile(
        r'\b(?:ignore|disregard|forget)\s+(?:(?:all|the|previous|prior|above|system)\s+)*'
        r'(?:instructions|rules|prompts)\b', re.I),
    'override_instructions_ru': re.compile(
        r'(?:игнорируй|игнорируйте|забудь|отмени)\s+(?:(?:все|предыдущие|системные)\s+)*'
        r'(?:инструкции|правила|указания)', re.I),
    'role_delimiter': re.compile(r'<\|(?:im_start|start_header_id|system)\|>|\[INST\]|<<SYS>>', re.I),
    'credential_value': re.compile(
        r'(?:суперпароль|пароль|password|api[_ -]?key|access[_ -]?token)'
        r'\s*(?:[\w.-]+\s*)?[:=]\s*\S+', re.I),
}


def suspicious_reasons(text):
    normalized = unicodedata.normalize('NFKC', text)
    normalized = ''.join(c for c in normalized if unicodedata.category(c) != 'Cf')
    return [name for name, pattern in RULES.items() if pattern.search(normalized)]


def filter_hits(hits):
    accepted, rejected = [], []
    for hit in hits:
        reasons = suspicious_reasons(hit['title'] + '\n' + hit['text'])
        if reasons:
            rejected.append({'chunk_id': hit['chunk_id'], 'reasons': reasons})
        else:
            accepted.append(hit)
    return accepted, rejected


def visible_text(result):
    return '\n'.join([result['answer'], *result['explanation'].values(), *[
        str(source[key]) for source in result['sources']
        for key in ('chunk_id', 'title', 'source', 'quote')]])

"""Общие параметры подготовки документов и поисковых запросов."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / 'config' / 'embedding_config.json'
MODEL_PATH = ROOT / 'models' / 'multilingual-e5-small'


def load_config():
    config = json.loads(CONFIG_PATH.read_text(encoding='utf-8'))
    if not 0 <= config['overlap_tokens'] < config['chunk_tokens'] < config['input_token_budget'] <= config['max_tokens']:
        raise ValueError('Несогласованные ограничения длины фрагментов')
    return config


def digest(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def passage_input(title, text, config):
    return f"{config['passage_prefix']}{title}\n\n{text}"


def token_count(tokenizer, text):
    return len(tokenizer.encode(text, add_special_tokens=True, truncation=False))

"""Загрузка официальных GGUF-весов Qwen с проверкой размера и SHA-256."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / 'config/llm_config.json'
MODEL_DIR = ROOT / 'models/qwen3-8b'


def verify_model(path, config):
    path = Path(path)
    if path.stat().st_size != config['file_size']:
        raise ValueError('Размер GGUF-файла не соответствует закреплённой версии')
    checksum = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            checksum.update(block)
    if checksum.hexdigest() != config['sha256']:
        raise ValueError('Контрольная сумма GGUF-файла не совпала')
    return checksum.hexdigest()


def download_model(config, directory=MODEL_DIR, force=False, verify_only=False):
    directory = Path(directory)
    path = directory / config['filename']
    if path.exists() and not force:
        print('Проверка локального файла модели…', flush=True)
        checksum = verify_model(path, config)
    else:
        if verify_only:
            raise FileNotFoundError('Локальные веса не найдены. Запустите скрипт без --verify-only.')
        directory.mkdir(parents=True, exist_ok=True)
        # Дополнительный резерв для служебных данных загрузчика.
        required = config['file_size'] + 512 * 1024 * 1024
        if shutil.disk_usage(directory).free < required:
            raise OSError(f'Для загрузки требуется минимум {required / 1e9:.2f} ГБ свободного места')
        from huggingface_hub import hf_hub_download
        print(f"Модель: {config['model_id']}\nВерсия: {config['revision']}\nФайл: {config['filename']} ({config['file_size']/1e9:.2f} ГБ)", flush=True)
        downloaded = hf_hub_download(
            repo_id=config['model_id'], revision=config['revision'],
            filename=config['filename'], local_dir=directory, force_download=force,
        )
        print('Загрузка завершена. Проверка размера и SHA-256…', flush=True)
        checksum = verify_model(downloaded, config)
    manifest = {k: config[k] for k in ['model_id', 'revision', 'filename', 'file_size']}
    manifest.update(sha256=checksum, integrity_verified=True)
    if not verify_only:
        (directory / 'download_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    print('Размер и SHA-256 подтверждены. Веса Qwen3-8B Q4_K_M готовы.')
    print('Генерация ответа не запускалась; работоспособность модели проверяется через llama.cpp отдельно.')
    return path


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--force-download', action='store_true', help='Повторно скачать файл, например после повреждения')
    mode.add_argument('--verify-only', action='store_true', help='Проверить локальный файл без сети и записи')
    args = parser.parse_args()
    try:
        config = json.loads(CONFIG_PATH.read_text(encoding='utf-8'))
        download_model(config, force=args.force_download, verify_only=args.verify_only)
    except (OSError, ValueError) as error:
        print(f'Подготовка весов не завершена: {error}', file=sys.stderr)
        print('Если локальный файл повреждён, выполните: python scripts/download_llm.py --force-download', file=sys.stderr)
        sys.exit(1)

"""Запуск llama-server с параметрами проекта и записью логов в файл."""
import argparse
import json
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
from download_llm import CONFIG_PATH, MODEL_DIR, ROOT, verify_model


def server_command(config, executable, cpu=False):
    return [executable, '--model', str(MODEL_DIR / config['filename']),
            '--host', config['host'], '--port', str(config['port']),
            '--alias', config['model_alias'], '--ctx-size', str(config['context_tokens']),
            '--parallel', str(config['parallel_requests']),
            '--n-predict', str(config['max_answer_tokens']),
            '--gpu-layers', '0' if cpu else str(config['gpu_layers']),
            '--jinja', '--chat-template-kwargs', json.dumps({'enable_thinking':config['enable_thinking']}),
            '--no-context-shift']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cpu', action='store_true', help='Запустить LLM без GPU')
    parser.add_argument('--print-command', action='store_true', help='Показать команду без запуска сервера')
    args = parser.parse_args()
    config = json.loads(CONFIG_PATH.read_text(encoding='utf-8'))
    executable = shutil.which('llama-server')
    if executable is None:
        raise FileNotFoundError('llama-server не найден. Выполните brew install llama.cpp')
    command = server_command(config, executable, args.cpu)
    if args.print_command:
        print(shlex.join(command))
        return 0
    print('Проверка целостности весов перед запуском…', flush=True)
    verify_model(MODEL_DIR / config['filename'], config)
    logs = ROOT / 'logs'
    logs.mkdir(exist_ok=True)
    log_path = logs / 'llama-server.log'
    version = subprocess.run([executable, '--version'], capture_output=True, text=True, check=True)
    print(f"Запуск сервера: http://{config['host']}:{config['port']}", flush=True)
    print(f'Диагностика: {log_path}', flush=True)
    print('Это окно занято сервером. В другом терминале выполните python scripts/check_llm_server.py.', flush=True)
    print('Остановка: Ctrl+C. Готовность модели проверит отдельная команда.', flush=True)
    with log_path.open('w', encoding='utf-8') as log:
        log.write(version.stdout + version.stderr + '\n' + shlex.join(command) + '\n')
        log.flush()
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        try:
            code = process.wait()
        except KeyboardInterrupt:
            # Терминал посылает Ctrl+C также дочернему процессу; ожидаем освобождения памяти.
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.terminate()
                process.wait(timeout=15)
            print('\nСервер остановлен.')
            return 0
    if code:
        print(f'Сервер завершился с кодом {code}. Последние строки лога:', file=sys.stderr)
        print('\n'.join(log_path.read_text(errors='replace').splitlines()[-25:]), file=sys.stderr)
    return 0 if code == 0 else 1


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        print(f'Сервер не запущен: {error}', file=sys.stderr)
        sys.exit(1)

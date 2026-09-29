"""Один вопрос или консольный диалог с RAG-ботом."""
import argparse
import sys
from rag_pipeline import RAGBot, display_result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('question', nargs='?')
    args = parser.parse_args()
    print('Подготовка поиска и подключение к локальной LLM…', flush=True)
    bot = RAGBot()
    if args.question is not None:
        display_result(bot.ask(args.question))
        return 0
    print('Бот готов. Задавайте самостоятельные вопросы. Для выхода: /exit.\n')
    while True:
        try:
            question = input('Вопрос: ').strip()
        except (EOFError, KeyboardInterrupt):
            print('\nДиалог завершён.')
            return 0
        if question == '/exit':
            return 0
        if not question:
            continue
        try:
            display_result(bot.ask(question))
        except (OSError, ValueError, KeyError, IndexError) as error:
            print(f'Не удалось получить проверенный ответ: {error}\n')


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (OSError, ValueError, KeyError, IndexError) as error:
        print(f'Не удалось выполнить запрос: {error}. Проверьте доступность сервера и актуальность индекса.',file=sys.stderr)
        sys.exit(1)

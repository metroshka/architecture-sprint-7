"""Загрузка закреплённых версий статей Wikipedia из data/sources.json."""

import argparse
import hashlib
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
API_URL = "https://en.wikipedia.org/w/api.php"
USER_AGENT = "ArchitectureSprint7/0.1 (https://github.com/metroshka/architecture-sprint-7)"


def digest(html):
    return hashlib.sha256(html.encode("utf-8")).hexdigest()


def load_sources(path):
    manifest = json.loads(path.read_text(encoding="utf-8"))
    sources = manifest["sources"]
    if manifest["schema_version"] != 1 or not sources:
        raise ValueError("Неподдерживаемый или пустой список источников")
    for key in ("id", "page_id", "revision_id"):
        if len({source[key] for source in sources}) != len(sources):
            raise ValueError(f"Повторяющееся значение поля {key}")
    for source in sources:
        if not re.fullmatch(r"[a-z0-9_]+", source["id"]):
            raise ValueError("Некорректный идентификатор источника")
        for key in ("page_id", "revision_id"):
            if type(source[key]) is not int or source[key] <= 0:
                raise ValueError(f"Некорректное поле {key}")
    return sources


def fetch_page(source):
    params = {
        "action": "parse", "format": "json", "formatversion": 2,
        "oldid": source["revision_id"], "prop": "text|sections|revid", "maxlag": 5,
    }
    request = Request(API_URL + "?" + urlencode(params), headers={"User-Agent": USER_AGENT})
    for attempt in range(3):
        try:
            with urlopen(request, timeout=60) as response:
                payload = json.load(response)
        except HTTPError as error:
            if error.code not in (429, 500, 502, 503, 504) or attempt == 2:
                raise
        except (URLError, TimeoutError):
            if attempt == 2:
                raise
        else:
            if "error" not in payload:
                return payload["parse"]
            error = payload["error"]
            if error.get("code") != "maxlag" or attempt == 2:
                raise ValueError(f"Ошибка API: {error}")
        time.sleep(2 ** (attempt + 1))
    raise RuntimeError("Не удалось получить статью")


def make_record(source, page):
    if page["pageid"] != source["page_id"] or page["revid"] != source["revision_id"]:
        raise ValueError("API вернул другую страницу или версию статьи")
    html = page["text"]
    if not isinstance(html, str) or not html.strip():
        raise ValueError("API вернул пустой текст статьи")
    return {
        "source": source,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "format": "html", "html_sha256": digest(html),
        "sections": page["sections"], "html": html,
    }


def check_cached(path, source):
    record = json.loads(path.read_text(encoding="utf-8"))
    if record["source"] != source or digest(record["html"]) != record["html_sha256"]:
        raise ValueError(f"Файл {path.name} отличается от ожидаемого; проверьте его перед повторной загрузкой")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--limit", type=int, help="Количество статей (по умолчанию 1)")
    selection.add_argument("--all", action="store_true", help="Загрузить все статьи из списка")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/raw")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit должен быть положительным числом")
    try:
        sources = load_sources(ROOT / "data/sources.json")
        selected = sources if args.all else sources[:args.limit or 1]
        args.output_dir.mkdir(parents=True, exist_ok=True)
        for number, source in enumerate(selected, 1):
            destination = args.output_dir / (source["id"] + ".json")
            if destination.exists():
                check_cached(destination, source)
                print(f"[{number}/{len(selected)}] Уже сохранено: {source['title']}", flush=True)
                continue
            if number > 1:
                time.sleep(1)
            record = make_record(source, fetch_page(source))
            temporary = destination.with_suffix(".json.tmp")
            temporary.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            temporary.replace(destination)
            print(f"[{number}/{len(selected)}] Сохранено: {source['title']}; версия {source['revision_id']}; {len(record['html'])} символов HTML", flush=True)
        print(f"Готово: {len(selected)} источников. Каталог: {args.output_dir.resolve()}")
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"Загрузка остановлена: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

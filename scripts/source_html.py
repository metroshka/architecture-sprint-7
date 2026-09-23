"""Извлечение абзацев и путей разделов из HTML MediaWiki."""

import re
from html.parser import HTMLParser

VOID_TAGS = {'br', 'hr', 'img', 'meta', 'link', 'input', 'wbr', 'source',
             'area', 'base', 'embed', 'param', 'track', 'col'}
EXCLUDED_TAGS = {'script', 'style', 'table', 'figure', 'aside', 'blockquote'}
EXCLUDED_CLASSES = {'mw-editsection', 'reference', 'mw-ref', 'noprint', 'navbox',
                    'sidebar', 'shortdescription', 'hatnote', 'metadata'}


class ParagraphParser(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.stack = []
        self.skip = 0
        self.buffer = None
        self.kind = None
        self.sections = []
        self.paragraphs = []

    def handle_starttag(self, tag, attrs):
        attributes = dict(attrs)
        blocked = tag in EXCLUDED_TAGS or bool(
            set((attributes.get('class') or '').split()) & EXCLUDED_CLASSES
        )
        if tag in VOID_TAGS:
            if tag == 'br' and self.buffer is not None and not self.skip:
                self.buffer.append(' ')
            return
        self.stack.append((tag, blocked))
        self.skip += int(blocked)
        if not self.skip and (tag == 'p' or re.fullmatch(r'h[1-6]', tag)):
            self.buffer = []
            self.kind = tag

    def handle_endtag(self, tag):
        if not self.skip and self.buffer is not None and tag == self.kind:
            text = re.sub(r'\s+', ' ', ''.join(self.buffer)).strip()
            if tag.startswith('h'):
                level = int(tag[1:])
                self.sections = [s for s in self.sections if s[0] < level]
                self.sections.append((level, text))
            elif text:
                self.paragraphs.append({
                    'index': len(self.paragraphs),
                    'section': ' / '.join(s[1] for s in self.sections),
                    'text': text,
                })
            self.buffer = None
            self.kind = None
        for index in range(len(self.stack) - 1, -1, -1):
            if self.stack[index][0] == tag:
                self.skip -= sum(int(s[1]) for s in self.stack[index:])
                del self.stack[index:]
                break

    def handle_data(self, text):
        if not self.skip and self.buffer is not None:
            self.buffer.append(text)


def extract_paragraphs(html):
    parser = ParagraphParser()
    parser.feed(html)
    parser.close()
    return parser.paragraphs

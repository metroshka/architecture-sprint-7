import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from clean_sources import clean_paragraph, prepare_document
from download_sources import digest
from source_html import extract_paragraphs


class CleaningTests(unittest.TestCase):
    def test_inline_text_and_nested_service_blocks(self):
        html = '''<div><table><tr><td><p>Infobox</p></td></tr></table>
        <h2>History<span class="mw-editsection">edit</span></h2>
        <p>Luke <b>trains</b> with <a href="/wiki/Yoda">Yoda</a>.<sup class="reference">[1]</sup></p>
        <div class="navbox"><div><p>Navigation</p></div></div>
        <p>A &amp; B.<br/>Second line.</p><script>untrusted()</script></div>'''
        rows = extract_paragraphs(html)
        self.assertEqual([r['text'] for r in rows], ['Luke trains with Yoda.', 'A & B. Second line.'])
        self.assertEqual(rows[0]['section'], 'History')

    def test_section_paths(self):
        rows = extract_paragraphs('<h2>A</h2><h3>B</h3><p>One</p><h2>C</h2><p>Two</p>')
        self.assertEqual([r['section'] for r in rows], ['A / B', 'C'])

    def rule(self, text, **extra):
        return {'paragraph_index': 0, 'section': '', 'paragraph_sha256': digest(text), **extra}

    def test_explicit_editorial_changes(self):
        text = 'Film intro. Luke pilots a ship. Film review.'
        rule = self.rule(text, start_at='Luke pilots', end_before='Film review.',
                         replacements=[{'old': 'a ship', 'new': 'an X-wing'}])
        self.assertEqual(clean_paragraph({'text': text, 'section': ''}, rule), 'Luke pilots an X-wing.')

    def test_changed_source_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Изменился'):
            clean_paragraph({'text': 'Different text', 'section': ''}, self.rule('Original text'))

    def test_ambiguous_edit_is_rejected(self):
        text = 'Luke meets Luke.'
        rule = self.rule(text, replacements=[{'old': 'Luke', 'new': 'Name'}])
        with self.assertRaisesRegex(ValueError, 'одно вхождение'):
            clean_paragraph({'text': text, 'section': ''}, rule)

    def test_media_marker_is_rejected(self):
        text = 'The actor appears in a film.'
        raw = {'html': '<p>' + text + '</p>'}
        with self.assertRaisesRegex(ValueError, 'маркеры'):
            prepare_document({'id': 'example'}, raw, [self.rule(text)])

    def test_insufficient_content_is_rejected(self):
        text = 'A brief description.'
        with self.assertRaisesRegex(ValueError, 'короткий'):
            prepare_document({'id': 'example'}, {'html': '<p>' + text + '</p>'}, [self.rule(text)])

    def test_negative_index_is_rejected(self):
        rule = self.rule('Text', paragraph_index=-1)
        with self.assertRaisesRegex(ValueError, 'диапазона'):
            prepare_document({'id': 'example'}, {'html': '<p>Text</p>'}, [rule])


if __name__ == '__main__':
    unittest.main()

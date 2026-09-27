import json
import os
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from embedding_settings import ROOT, MODEL_PATH, load_config, token_count
from prepare_chunks import split_document
from transformers import AutoTokenizer


class ChunkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        location = Path(os.environ.get('E5_TEST_TOKENIZER', str(MODEL_PATH)))
        if not (location / 'tokenizer.json').is_file():
            raise unittest.SkipTest('Сначала загрузите модель или задайте E5_TEST_TOKENIZER')
        cls.tokenizer = AutoTokenizer.from_pretrained(str(location), local_files_only=True)
        cls.config = load_config()

    def assert_coverage(self, content, chunks):
        covered = set()
        for chunk in chunks:
            start, end = chunk['start_char'], chunk['end_char']
            self.assertEqual(content[start:end], chunk['text'])
            covered.update(range(start, end))
            self.assertEqual(chunk['input_tokens'], token_count(self.tokenizer, chunk['embedding_text']))
            self.assertLessEqual(chunk['input_tokens'], self.config['input_token_budget'])
            self.assertEqual(chunk['word_count'], len(chunk['text'].split()))
            self.assertGreaterEqual(chunk['word_count'], 100)
            self.assertLessEqual(chunk['word_count'], 300)
        body_start = content.index('\n') + 1
        self.assertTrue(all(i in covered for i in range(body_start, len(content)) if not content[i].isspace()))

    def test_repeated_sentences_keep_offsets(self):
        content = '# Sovar\n\n' + 'The planet has two moons. ' * 250
        chunks = split_document(content, 'knowledge_base/kb_001.md', self.tokenizer, self.config)
        self.assertGreater(len(chunks), 1)
        self.assert_coverage(content, chunks)
        self.assertEqual(len({c['chunk_id'] for c in chunks}), len(chunks))
        self.assertTrue(all(b['start_char'] >= a['end_char'] for a, b in zip(chunks, chunks[1:])))

    def test_unicode_and_paragraphs_preserved(self):
        content = '# Совар\n\n' + ('Планета Совар — место жизни исследователей.\n\nЛиор изучает спутники.\n' * 70)
        chunks = split_document(content, 'knowledge_base/kb_002.md', self.tokenizer, self.config)
        self.assert_coverage(content, chunks)

    def test_long_title_is_rejected_instead_of_truncated(self):
        content = '# ' + 'Planet ' * 600 + '\n\nShort body.'
        with self.assertRaisesRegex(ValueError, 'токенов'):
            split_document(content, 'knowledge_base/kb_003.md', self.tokenizer, self.config)

    def test_empty_body_rejected(self):
        with self.assertRaises(ValueError):
            split_document('# Sovar\n\n', 'knowledge_base/kb_004.md', self.tokenizer, self.config)

    def test_short_document_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'минимум 100'):
            split_document('# Sovar\n\nTwo moons.\n', 'knowledge_base/kb_005.md', self.tokenizer, self.config)

    def test_short_tail_is_redistributed_without_padding(self):
        content = '# Sovar\n\n' + 'moon ' * 301
        chunks = split_document(content, 'knowledge_base/kb_005.md', self.tokenizer, self.config)
        self.assertEqual(len(chunks), 2)
        self.assertEqual(sum(c['word_count'] for c in chunks), 301)
        self.assert_coverage(content, chunks)

    def test_token_limit_can_require_split_below_maximum_words(self):
        content = '# Sovar\n\n' + 'intercontinental ' * 200
        chunks = split_document(content, 'knowledge_base/kb_005.md', self.tokenizer, self.config)
        self.assertGreater(len(chunks), 1)
        self.assert_coverage(content, chunks)

    def test_impossible_token_and_word_limits_fail(self):
        content = '# Sovar\n\n' + ('абвгдежзийклмнопрстуфхцчшщъыьэюя ' * 100)
        with self.assertRaisesRegex(ValueError, 'невозможно совместить'):
            split_document(content, 'knowledge_base/kb_005.md', self.tokenizer, self.config)

    def test_all_thirty_documents_meet_limits_and_preserve_text(self):
        documents = sorted((ROOT / 'knowledge_base').glob('*.md'))
        self.assertEqual(len(documents), 30)
        for path in documents:
            with self.subTest(document=path.name):
                content = path.read_text(encoding='utf-8')
                chunks = split_document(content, f'knowledge_base/{path.name}', self.tokenizer, self.config)
                self.assert_coverage(content, chunks)
                self.assertEqual(sum(c['word_count'] for c in chunks), len(content.partition('\n')[2].split()))

    def test_only_exact_security_fixture_can_bypass_minimum(self):
        content = '# Security probe\n\n' + (ROOT / 'data/security/malicious.txt').read_text(encoding='utf-8')
        source = 'knowledge_base/injection_probe.md'
        with self.assertRaisesRegex(ValueError, 'минимум 100'):
            split_document(content, source, self.tokenizer, self.config)
        chunks = split_document(content, source, self.tokenizer, self.config, allow_security_fixture=True)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0]['text'], content.partition('\n')[2].strip())
        self.assertEqual(chunks[0]['short_document_exception'], 'prompt_injection_fixture')
        for bad_content, bad_source in [(content+'Changed.', source), (content, 'knowledge_base/kb_005.md')]:
            with self.assertRaisesRegex(ValueError, 'Исключение разрешено'):
                split_document(bad_content, bad_source, self.tokenizer, self.config, allow_security_fixture=True)


if __name__ == '__main__':
    unittest.main()

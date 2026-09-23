import json
import os
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from embedding_settings import MODEL_PATH, load_config, token_count
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
        body_start = content.index('\n') + 1
        self.assertTrue(all(i in covered for i in range(body_start, len(content)) if not content[i].isspace()))

    def test_repeated_sentences_and_overlap_keep_offsets(self):
        content = '# Sovar\n\n' + 'The planet has two moons. ' * 250
        chunks = split_document(content, 'knowledge_base/kb_001.md', self.tokenizer, self.config)
        self.assertGreater(len(chunks), 1)
        self.assert_coverage(content, chunks)
        self.assertEqual(len({c['chunk_id'] for c in chunks}), len(chunks))
        self.assertTrue(any(b['start_char'] < a['end_char'] for a, b in zip(chunks, chunks[1:])))

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

    def test_short_document_has_title_and_prefix(self):
        chunks = split_document('# Sovar\n\nTwo moons.\n', 'knowledge_base/kb_005.md', self.tokenizer, self.config)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0]['embedding_text'], 'passage: Sovar\n\nTwo moons.')


if __name__ == '__main__':
    unittest.main()

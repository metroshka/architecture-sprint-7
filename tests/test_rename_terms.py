import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from download_sources import digest
from rename_terms import TermReplacer, prepare_corpus, unique_object


class RenamingTests(unittest.TestCase):
    def test_longest_name_wins(self):
        result, counts, _ = TermReplacer({'R2': 'K8', 'R2-D2': 'K8-M4'}).replace('R2-D2 follows R2.')
        self.assertEqual(result, 'K8-M4 follows K8.')
        self.assertEqual(dict(counts), {'R2-D2': 1, 'R2': 1})

    def test_word_boundaries_and_possessives(self):
        result, _, _ = TermReplacer({'Luke': 'Lior', 'Han': 'Taren'}).replace("Luke's Luke’s Lukewarm chandelier")
        self.assertEqual(result, "Lior's Lior’s Lukewarm chandelier")

    def test_replacement_does_not_cascade(self):
        replacer = TermReplacer({'Luke': 'Anakin', 'Anakin': 'Aren'})
        result, _, _ = replacer.replace('Luke meets Anakin.')
        self.assertEqual(result, 'Anakin meets Aren.')
        self.assertEqual(replacer.remaining(result), ['Anakin'])

    def test_plural_and_case(self):
        result, _, _ = TermReplacer({'Rebel': 'Concord', 'Rebels': 'Concord members'}).replace('Rebels and a rebel.')
        self.assertEqual(result, 'Concord members and a concord.')

    def test_articles(self):
        result, _, changes = TermReplacer({'X-wing': 'Spearfin', 'kyber crystal': 'oriven crystal'}).replace('An X-wing carries a kyber crystal.')
        self.assertEqual(result, 'A Spearfin carries an oriven crystal.')
        self.assertEqual(changes, 2)

    def test_case_collision_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'регистр'):
            TermReplacer({'Luke': 'Lior', 'luke': 'Other'})

    def test_duplicate_json_key_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Повторяющийся'):
            json.loads('{"Luke":"Lior","Luke":"Other"}', object_pairs_hook=unique_object)

    def test_real_aliases_remain_consistent(self):
        terms = json.loads((Path(__file__).resolve().parents[1] / 'terms_map.json').read_text())
        text, _, _ = TermReplacer(terms).replace('Anakin Skywalker becomes Darth Vader. Vader is the father of Luke Skywalker. Luke meets Anakin.')
        self.assertEqual(text, 'Aren Veylan becomes Vhar Kordane. Kordane is the father of Lior Veylan. Lior meets Aren.')

    def make_corpus(self, directory):
        source = {'id': 'luke', 'title': 'Luke'}
        text = '# Luke\n\nLuke flies.\n'
        (directory / 'luke.md').write_text(text)
        record = {'id': 'luke', 'source': source, 'document_sha256': digest(text), 'word_count': 2}
        manifest = {'schema_version': 1, 'document_count': 1, 'word_count': 2,
                    'cleaning_rules_sha256': 'fixture', 'documents': [record]}
        (directory / 'manifest.json').write_text(json.dumps(manifest))
        return [source]

    def test_neutral_filename_and_no_original_title(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            sources = self.make_corpus(directory)
            docs, _, _ = prepare_corpus(directory, sources, TermReplacer({'Luke': 'Lior'}))
            self.assertEqual(docs[0][0], 'kb_001.md')
            self.assertEqual(docs[0][1], '# Lior\n\nLior flies.\n')
            self.assertEqual(docs[0][2]['source'], sources[0])

    def test_modified_document_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            sources = self.make_corpus(directory)
            (directory / 'luke.md').write_text('# Luke\n\nChanged.\n')
            with self.assertRaisesRegex(ValueError, 'Изменён'):
                prepare_corpus(directory, sources, TermReplacer({'Luke': 'Lior'}))

    def test_residual_source_term_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            sources = self.make_corpus(directory)
            with self.assertRaisesRegex(ValueError, 'Остались'):
                prepare_corpus(directory, sources, TermReplacer({'Luke': 'Anakin', 'Anakin': 'Aren'}))


if __name__ == '__main__':
    unittest.main()

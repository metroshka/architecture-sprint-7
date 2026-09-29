import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from download_llm import verify_model, download_model


class ModelIntegrityTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.path = self.directory / 'fixture.gguf'
        self.data = b'GGUF test fixture'
        self.path.write_bytes(self.data)
        self.config = {'model_id':'test/model', 'revision':'test', 'filename':self.path.name,
                       'file_size':len(self.data), 'sha256':hashlib.sha256(self.data).hexdigest()}

    def test_valid_file(self):
        self.assertEqual(verify_model(self.path, self.config), self.config['sha256'])

    def test_truncated_file_rejected(self):
        self.path.write_bytes(self.data[:-1])
        with self.assertRaisesRegex(ValueError, 'Размер'):
            verify_model(self.path, self.config)

    def test_modified_file_of_same_size_rejected(self):
        self.path.write_bytes(b'X' + self.data[1:])
        with self.assertRaisesRegex(ValueError, 'Контрольная сумма'):
            verify_model(self.path, self.config)

    def test_cached_file_does_not_need_network(self):
        with patch('huggingface_hub.hf_hub_download', side_effect=AssertionError('Network call')):
            download_model(self.config, self.directory)
        manifest = json.loads((self.directory/'download_manifest.json').read_text())
        self.assertTrue(manifest['integrity_verified'])
        self.assertEqual(manifest['revision'], self.config['revision'])

    def test_verify_only_does_not_write(self):
        download_model(self.config, self.directory, verify_only=True)
        self.assertFalse((self.directory/'download_manifest.json').exists())

    def test_missing_file_verify_only_rejected(self):
        self.path.unlink()
        with self.assertRaises(FileNotFoundError):
            download_model(self.config, self.directory, verify_only=True)


if __name__ == '__main__':
    unittest.main()

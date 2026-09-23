import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

import faiss
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from embedding_settings import load_config, digest, passage_input
from vector_store import validate_chunks, file_digest, load_index


class VectorStoreTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root/'knowledge_base').mkdir()
        self.content = '# Sovar\n\nThe planet has two moons.\n'
        (self.root/'knowledge_base/kb_001.md').write_text(self.content)
        self.config = dict(load_config(), dimension=3)
        body = 'The planet has two moons.'
        chunk = {'chunk_id':'kb_001_001', 'source':'knowledge_base/kb_001.md', 'title':'Sovar',
                 'start_char':9, 'end_char':9+len(body), 'text':body, 'chunk_number':1,
                 'document_sha256':digest(self.content), 'input_tokens':12,
                 'embedding_text':passage_input('Sovar',body,self.config)}
        self.payload = {'schema_version':1, 'embedding_config':self.config,
                        'chunk_count':1, 'document_count':1, 'chunks':[chunk]}

    def save(self, dimension=3):
        directory = self.root/'index'
        directory.mkdir()
        index = faiss.IndexFlatIP(dimension)
        vector = np.zeros((1,dimension),dtype='float32');vector[0,0]=1
        index.add(vector)
        faiss.write_index(index,str(directory/'faiss.index'))
        (directory/'metadata.json').write_text(json.dumps(self.payload))
        manifest = {'schema_version':1,'embedding_config':self.config,'chunk_count':1,
                    'files_sha256':{name:file_digest(directory/name) for name in ['faiss.index','metadata.json']}}
        (directory/'manifest.json').write_text(json.dumps(manifest))
        return directory

    def test_roundtrip_preserves_vector_to_text_mapping(self):
        directory = self.save()
        index,chunks,_ = load_index(directory,self.config,self.root)
        scores,positions=index.search(np.array([[1,0,0]],dtype='float32'),1)
        self.assertAlmostEqual(float(scores[0,0]),1)
        self.assertEqual(chunks[int(positions[0,0])]['text'],'The planet has two moons.')

    def test_changed_source_rejected(self):
        directory=self.save()
        (self.root/'knowledge_base/kb_001.md').write_text(self.content+'Changed.')
        with self.assertRaisesRegex(ValueError,'Документ изменён'):
            load_index(directory,self.config,self.root)

    def test_added_document_rejected(self):
        (self.root/'knowledge_base/kb_002.md').write_text('# New\n\nNew text.')
        with self.assertRaisesRegex(ValueError,'Состав базы'):
            validate_chunks(self.payload,self.config,self.root)

    def test_modified_metadata_rejected_before_reading_index(self):
        directory=self.save()
        with (directory/'metadata.json').open('a') as f:f.write(' ')
        with self.assertRaisesRegex(ValueError,'целостность'):
            load_index(directory,self.config,self.root)

    def test_different_revision_rejected(self):
        directory=self.save()
        config=dict(self.config,revision='different')
        with self.assertRaisesRegex(ValueError,'Конфигурация'):
            load_index(directory,config,self.root)

    def test_wrong_dimension_rejected(self):
        directory=self.save(dimension=4)
        with self.assertRaisesRegex(ValueError,'Структура индекса'):
            load_index(directory,self.config,self.root)

    def test_omitted_text_rejected(self):
        payload=copy.deepcopy(self.payload)
        c=payload['chunks'][0]
        c['text']='The planet';c['end_char']=c['start_char']+len(c['text'])
        c['embedding_text']=passage_input(c['title'],c['text'],self.config)
        with self.assertRaisesRegex(ValueError,'не покрывают'):
            validate_chunks(payload,self.config,self.root)

    def test_wrong_prefix_rejected(self):
        self.payload['chunks'][0]['embedding_text']='query: The planet has two moons.'
        with self.assertRaisesRegex(ValueError,'вход embedding'):
            validate_chunks(self.payload,self.config,self.root)


if __name__=='__main__':
    unittest.main()

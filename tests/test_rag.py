import unittest
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.vectorstores import InMemoryVectorStore
from rag import read_document, split_documents


class IngestionTests(unittest.TestCase):
    def test_source_and_retrieval(self):
        chunks = split_documents(read_document('handbook.md', b'Annual leave is 20 days.'))
        store = InMemoryVectorStore.from_documents(chunks, DeterministicFakeEmbedding(size=32))
        result = store.similarity_search('leave', k=1)
        self.assertEqual(result[0].metadata['source'], 'handbook.md')
        self.assertIn('20 days', result[0].page_content)

    def test_empty_document(self):
        with self.assertRaises(ValueError):
            read_document('empty.txt', b'  ')

    def test_unsupported_file(self):
        with self.assertRaises(ValueError):
            read_document('program.exe', b'hello')

    def test_chunk_limit(self):
        with self.assertRaises(ValueError):
            split_documents(read_document('big.txt', b'word ' * 60000))


if __name__ == '__main__':
    unittest.main()

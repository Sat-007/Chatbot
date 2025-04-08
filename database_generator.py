from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.embeddings import Embeddings
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_chroma import Chroma
from uuid import uuid4
import requests


from dotenv import load_dotenv
load_dotenv()


DATA_PATH = r'data'
CHROMA_DB_PATH = r'chroma_db'

# ollama pull nomic-embed-text
# ollama run nomic-embed-text

class OllamaEmbeddings(Embeddings):
    def __init__(self, model: str = "nomic-embed-text"):
        self.model = model

    def embed_documents(self, texts):
        return [self._embed(t.page_content if hasattr(t, "page_content") else t) for t in texts]

    def embed_query(self, text):
        return self._embed(text)

    def _embed(self, text):
        response = requests.post(
            "http://localhost:11434/api/embeddings",
            json={"model": self.model, "prompt": text}
        )
        response.raise_for_status()
        return response.json()["embedding"]

embeddings_model = OllamaEmbeddings(model="nomic-embed-text")

vector_store = Chroma(
    collection_name = 'pdf_collections',
    embedding_function=embeddings_model,
    persist_directory=CHROMA_DB_PATH
)

loader = PyPDFDirectoryLoader(DATA_PATH)

raw_documents = loader.load()
#print(raw_documents)

text_split = RecursiveCharacterTextSplitter(
    chunk_size = 300,
    chunk_overlap = 100,
    length_function = len,
    is_separator_regex=False    
)


chunks = text_split.split_documents(raw_documents)

uuids = [str(uuid4()) for _ in range(len(chunks))]

vector_store.add_documents(documents=chunks, ids=uuids)
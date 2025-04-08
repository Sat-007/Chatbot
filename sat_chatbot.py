from langchain_chroma import Chroma
from langchain_community.llms import Ollama
from langchain_core.embeddings import Embeddings
from langchain.prompts import PromptTemplate
import gradio as gr
import subprocess
import requests
from dotenv import load_dotenv

load_dotenv()

CHROMA_PATH = r"chroma_db"
COLLECTION_NAME = "pdf_collections"

def ensure_ollama_model(model_name="llama3"):
    try:
        subprocess.run(["ollama", "show", model_name], check=True, capture_output=True)
    except subprocess.CalledProcessError:
        subprocess.run(["ollama", "pull", model_name], check=True)
       

ensure_ollama_model("llama3")




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
    collection_name=COLLECTION_NAME,
    embedding_function=embeddings_model,
    persist_directory=CHROMA_PATH
)

retriever = vector_store.as_retriever(search_kwargs={'k': 5})
llm = Ollama(model="llama3")



prompt_template = """
You pocket AI ready for duty!! Will answer any questions related to my domain 
If the context does not contain the answer, respond with "I dont know, please ask me questions about the docuements!!!!!!!!!!!!"

Context:
{context}

Question: {question}

Answer:
"""
prompt = PromptTemplate(input_variables=["context", "question"], template=prompt_template)


def stream_response(message, history):
    docs = retriever.get_relevant_documents(message)
    context = "\n\n".join([doc.page_content for doc in docs])
    final_prompt = prompt.format(context=context, question=message)

    partial_message = ""
    for response in llm.stream(final_prompt):
        partial_message += response
        yield partial_message

chatbot = gr.ChatInterface(
    fn=stream_response,
    textbox=gr.Textbox(
        placeholder="Ask your question about the documents...",
        container=False,
        autoscroll=True,
        scale=7
    ),
)


chatbot.launch()

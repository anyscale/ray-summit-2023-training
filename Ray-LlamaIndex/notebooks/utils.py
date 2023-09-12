import numpy as np
from langchain.embeddings import OpenAIEmbeddings
from langchain.embeddings.huggingface import HuggingFaceEmbeddings

from llama_index.vector_stores import PGVectorStore
from llama_index import VectorStoreIndex, ServiceContext
from llama_index.llms import Anyscale, OpenAI

def get_embedding_model(model_name):
    if model_name == "text-embedding-ada-002":
            return OpenAIEmbeddings(
                model=model_name,
                openai_api_base=os.environ["OPENAI_API_BASE"],
                openai_api_key=os.environ["OPENAI_API_KEY"])
    else:
        model_kwargs = {"device": "cuda"}
        encode_kwargs = {"device": "cuda", "batch_size": 100}

        return HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs)
    


def get_postgres_store():
    return PGVectorStore.from_params(
            database="postgres", 
            user="postgres", 
            password="postgres", 
            host="localhost", 
            table_name="document",
            port="5432",
            embed_dim=768,
        )
    
def _get_vector_store_index(
    service_context,
    ):

    vector_store = get_postgres_store()
    index = VectorStoreIndex.from_vector_store(vector_store, service_context=service_context)
    return index


def get_query_engine(
    llm_model_name: str = "meta-llama/Llama-2-70b-chat-hf",
    temperature: float = 0.1,
    embedding_model_name = "thenlper/gte-base",
    similarity_top_k=2
):
    embed_model = get_embedding_model(embedding_model_name)

    if "llama" in llm_model_name:
        llm = Anyscale(model=llm_model_name, temperature=temperature)
    else:
        llm = OpenAI(model=llm_model_name, temperature=temperature)
    
    service_context = ServiceContext.from_defaults(embed_model=embed_model, llm=llm)

    index = _get_vector_store_index(service_context)
    return index.as_query_engine(similarity_top_k=similarity_top_k)

def get_retriever(    
    embedding_model_name = "thenlper/gte-base",
    similarity_top_k=2
):

    embed_model = get_embedding_model(embedding_model_name)
    service_context = ServiceContext.from_defaults(embed_model=embed_model)

    index = _get_vector_store_index(service_context)
    return index.as_query_engine(similarity_top_k=similarity_top_k)

    

    
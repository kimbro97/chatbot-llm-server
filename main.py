from fastapi import FastAPI
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from pydantic import BaseModel

from langchain_community.document_loaders import Docx2txtLoader

from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain import hub

import os

from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore

class QuestionRequest(BaseModel):
    question: str

llm = ChatOpenAI(model="gpt-4o")

load_dotenv()

app = FastAPI()

@app.post("/v1/question") 
async def llm_search(request: QuestionRequest):
    loader = Docx2txtLoader("./tax.docx")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 1500,
        chunk_overlap = 200,
    )

    document_list = loader.load_and_split(text_splitter=text_splitter)

    embedding = OpenAIEmbeddings(model="text-embedding-3-large")

    index_name = 'tax-index'
    pinecone_api_key = os.environ.get("PINECONE_API_KEY")
    pc = Pinecone(api_key=pinecone_api_key)

    database = PineconeVectorStore.from_documents(document_list, embedding, index_name=index_name)
    
    retriever = database.as_retriever(search_kwargs={'k': 1})
    retriever.invoke(request.question)
    retrieved_docs = retriever.invoke(request.question)

    print(retrieved_docs)
    print(len(retrieved_docs))

    rlm_rag_prompt = hub.pull("rlm/rag-prompt")

    rlm_rag_chain = rlm_rag_prompt | llm
    ai_message = rlm_rag_chain.invoke({"context": retrieved_docs, "question": request.question})
    
    return ai_message.content
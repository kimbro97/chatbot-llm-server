from fastapi import FastAPI
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from pydantic import BaseModel

from langchain_community.document_loaders import Docx2txtLoader

from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_chroma import Chroma

from langchain import hub

from langchain_core.prompts import PromptTemplate

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

    # database = Chroma.from_documents(documents=document_list, embedding=embedding, collection_name="chroma-tax", persist_directory="./chroma")
    database = Chroma.from_documents(collection_name="chroma-tax", persist_directory="./chroma")

    query = "연봉 5천만원인 직장인의 소득세는 얼마인가요?"

    retrieved_docs = database.similarity_search(query, k=3)

    # prompt = f"""[Identity]
    # - 당신은 최고의 한국 소득세 전문가 입니다
    # - [Context]를 참고해서 사용자의 질문에 답변해주세요

    # [Context]
    # {retrieved_docs}

    # Question: {query}
    # """

    prompt_with_template = '아래 질문에 답변해주세요:\n\n {question}'

    prompt_template = PromptTemplate(template=prompt_with_template, input_variables=["question"])
    prompt_chain = prompt_template | llm

    ai_message = prompt_chain.invoke({"question": request.question})

    print(ai_message.content)
    # answer = llm.invoke(request.question)
    return request
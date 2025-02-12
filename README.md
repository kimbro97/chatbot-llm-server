## 설치한 패키지 
``` sh
$ pip install FastAPI
$ pip install langchain-openai
$ pip install python-dotenv
$ pip install uvicorn
$ pip install langchain langchain-community langchain-text-splitters docx2txt langchain-chroma
$ pip install langchainhub
$ pip install pinecone-notebooks
```

## 실행방법
``` sh
$ uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```
from fastapi import FastAPI, UploadFile, File, Query
from fastapi.responses import JSONResponse
from langchain_community.llms import LlamaCpp
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
import os
import shutil

# Khởi tạo FastAPI
app = FastAPI()

# Khai báo model và embedding
MODEL_PATH = "vinallama-7b-chat_q5_0.gguf"
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

# Khởi tạo LLM
llm = LlamaCpp(
    model_path=MODEL_PATH,
    temperature=0.5,
    max_tokens=256,
    n_ctx=2048,
    n_threads=4,
    verbose=False
)

# FAISS database toàn cục
db = None

# Hàm tải và xử lý file, cập nhật FAISS DB
def update_vectorstore(file_path: str):
    global db
    loader = TextLoader(file_path, encoding='utf-8')
    documents = loader.load()

    text_splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    docs = text_splitter.split_documents(documents)

    if db is None:
        db = FAISS.from_documents(docs, embedding_model)
    else:
        db.add_documents(docs)

# Endpoint để upload file .txt và cập nhật vectorstore
@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    if not file.filename.endswith(".txt"):
        return JSONResponse(content={"error": "Chỉ hỗ trợ file .txt"}, status_code=400)

    file_location = f"uploaded_{file.filename}"
    with open(file_location, "wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        update_vectorstore(file_location)
        return {"message": f"Đã cập nhật dữ liệu từ {file.filename}"}
    finally:
        os.remove(file_location)  # Xóa file sau khi xử lý

# Endpoint để hỏi
@app.get("/ask")
async def ask_question(question: str = Query(..., description="Câu hỏi cần trả lời")):
    if db is None:
        return JSONResponse(content={"error": "Chưa có dữ liệu để truy vấn. Hãy upload file trước."}, status_code=400)

    retriever = db.as_retriever()
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    answer = qa_chain.run(question)
    return {"question": question, "answer": answer}

@app.get("/ask_2")
async def ask_question(question: str = Query(..., description="Câu hỏi cần trả lời")):
    if db is None or db.index.ntotal == 0:
        # Nếu không có dữ liệu, dùng mô hình LLM để trả lời trực tiếp
        direct_answer = llm.invoke(question)
        return {"question": question, "answer": direct_answer}

    retriever = db.as_retriever()
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    answer = qa_chain.run(question)
    return {"question": question, "answer": answer}

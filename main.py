from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.responses import JSONResponse
import tempfile
from langchain_community.llms import LlamaCpp
from langchain_community.document_loaders import PyPDFLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain.chains import RetrievalQA
from langchain_chroma import Chroma
from langchain_huggingface.embeddings import HuggingFaceEmbeddings # hoặc embeddings bạn đang dùng

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
# Khởi tạo embeddings (hoặc thay bằng bất kỳ embedding model nào bạn dùng)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vector_db = Chroma(
            persist_directory="./chroma_langchain_db",
            embedding_function =embeddings
        )
@app.post("/process-pdf/")
async def process_pdf(file: UploadFile = File(...)):
    try:
        # Step 1: Save PDF to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_path = tmp_file.name

        # Step 2: Load the PDF
        loader = PyPDFLoader(tmp_path)
        documents = loader.load()

        # Step 3: Semantic Split
        semantic_splitter = SemanticChunker(
            embeddings=embeddings,
            buffer_size=1,
            breakpoint_threshold_type="percentile",
            breakpoint_threshold_amount=95,
            min_chunk_size=500,
            add_start_index=True
        )
        docs = semantic_splitter.split_documents(documents)

        # Step 4: Store into Chroma vector store
        vector_db = Chroma.from_documents(
            documents=docs,
            embedding=embeddings,
            persist_directory="./chroma_langchain_db"    # để lưu trữ vĩnh viễn
        )

        return JSONResponse(content={"message": f"Processed and stored {len(docs)} chunks."})

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/chroma-doc-count/")
def get_chroma_doc_count():
    try:
        vector_db = Chroma(
            persist_directory="./chroma_langchain_db",
            embedding_function =embeddings
        )
        # Get number of stored documents
        doc_count = len(vector_db.get()['documents'])

        return {"document_count": doc_count}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Endpoint để hỏi
@app.get("/ask")
async def ask_question(question: str = Query(..., description="Câu hỏi cần trả lời")):
    vector_db = Chroma(
            persist_directory="./chroma_langchain_db",
            embedding_function =embeddings
        )
    if vector_db is None:
        return JSONResponse(content={"error": "Chưa có dữ liệu để truy vấn. Hãy upload file trước."}, status_code=400)

    retriever = vector_db.as_retriever()
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    answer = qa_chain.run(question)
    return {"question": question, "answer": answer}

@app.get("/ask_2")
async def ask_question(question: str = Query(..., description="Câu hỏi cần trả lời")):
    vector_db = Chroma(
            persist_directory="./chroma_langchain_db",
            embedding_function =embeddings
        )
    if vector_db is None or vector_db.index.ntotal == 0:
        # Nếu không có dữ liệu, dùng mô hình LLM để trả lời trực tiếp
        direct_answer = llm.invoke(question)

        return {"question": question, "answer": direct_answer}

    retriever = vector_db.as_retriever()
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    answer = qa_chain.run(question)
    return {"question": question, "answer": answer}
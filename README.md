# RAG-system-vinallama
# 🤖 Trợ Lý AI Đọc PDF – FastAPI + GGUF + LlamaCpp + Chroma 

Ứng dụng AI đơn giản, gọn nhẹ cho phép bạn **tải file PDF**, trích xuất nội dung vào **ChromaDB**, và **đặt câu hỏi bằng tiếng Việt** về nội dung đã nạp.  
⚡ Tất cả chỉ trong **một file duy nhất: `main.py`** – không cần backend phức tạp, không cần OpenAI API, không tốn phí.

---

## 📂 Cấu trúc tối giản

| Tên file     | Mô tả                                                                 |
|--------------|----------------------------------------------------------------------|
| `main.py`    | File duy nhất – tích hợp FastAPI, ChromaDB, LlamaCpp và xử lý PDF   |

---

## 🧠 Công nghệ sử dụng

- **FastAPI** – API backend hiện đại, dễ dùng
- **LlamaCpp + GGUF** – Mô hình ngôn ngữ chạy offline, không cần OpenAI API
- **ChromaDB** – Vector database để lưu trữ văn bản và tìm kiếm ngữ nghĩa
- **PyPDFLoader** – Trích xuất nội dung từ PDF
- **LangChain** – Kết nối các thành phần (LLM + vector + logic hỏi đáp)

---

## ✅ Tính năng nổi bật

✔️ **Tải lên file PDF trực tiếp qua API**  
✔️ Tự động trích xuất nội dung bằng SemanticChunker khi các đoạn không còn giống nhau tới 95% và lưu vào ChromaDB  
✔️ **Đặt câu hỏi bằng tiếng Việt** về nội dung PDF đã upload  
✔️ Dùng mô hình **GGUF offline**, không cần internet, không tốn chi phí  
✔️ **Gọn – nhẹ – chạy được cả trên máy cấu hình thấp**



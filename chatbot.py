import os
import streamlit as st
import shutil
import re
import uuid
import unicodedata
import pickle # Thêm pickle để lưu/đọc cache BM25
from pathlib import Path
from typing import List

# --- CẤU HÌNH TRANG ---
st.set_page_config(
    page_title="KTC Chatbot - THCS & THPT Phạm Kiệt",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- KHỐI IMPORT AN TOÀN (Tránh crash nếu thiếu thư viện phụ) ---
try:
    import nest_asyncio
    nest_asyncio.apply()
    from llama_parse import LlamaParse
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_community.vectorstores import FAISS
    from langchain_community.retrievers import BM25Retriever
    from langchain.retrievers import EnsembleRetriever
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_core.documents import Document
    from groq import Groq
    
    # Flashrank (Tùy chọn)
    try:
        from flashrank import Ranker, RerankRequest
        HAS_FLASHRANK = True
    except ImportError:
        HAS_FLASHRANK = False
        
    DEPENDENCIES_OK = True
except ImportError as e:
    DEPENDENCIES_OK = False
    IMPORT_ERROR = str(e)

# =============================
# 1. CẤU HÌNH HỆ THỐNG
# =============================

class AppConfig:
    # --- ĐIỀN API KEY CỦA THẦY VÀO ĐÂY ---
    GROQ_API_KEY = "gsk_..."  # Thay bằng key thật của thầy
    LLAMA_CLOUD_API_KEY = "llx-..." # Thay bằng key thật (nếu dùng LlamaParse)
    
    LLM_MODEL = "llama-3.3-70b-versatile"
    EMBEDDING_MODEL = "dangvantuan/vietnamese-embedding"
    
    UPLOAD_DIR = "PDF_KNOWLEDGE"
    VECTOR_DB_DIR = "FAISS_DB"
    BM25_PATH = os.path.join(VECTOR_DB_DIR, "bm25_docs.pkl")
    LOGO_PROJECT = "LOGO.jpg"
    
    SYSTEM_PROMPT = """Bạn là Trợ lý học tập môn Tin học, hỗ trợ giáo viên và học sinh trường Phạm Kiệt theo SGK Kết nối tri thức.
    
    QUY TẮC TRẢ LỜI:
    1. Căn cứ CHÍNH XÁC vào ngữ cảnh (Context) được cung cấp.
    2. Nếu ngữ cảnh có thông tin: Trả lời chi tiết, sư phạm, dễ hiểu.
    3. TRÍCH DẪN NGUỒN: Bắt buộc ghi rõ (Sách nào -> Chủ đề nào -> Bài nào).
    4. Nếu không có thông tin: Trả lời "Dựa trên tài liệu SGK hiện có, tôi chưa tìm thấy thông tin này."
    """

# =============================
# 2. XỬ LÝ DỮ LIỆU (FIX LOGIC KNTT)
# =============================

class KNTT_Processor:
    @staticmethod
    def normalize_text(text: str) -> str:
        if not text: return ""
        text = unicodedata.normalize("NFC", text)
        return re.sub(r'\s+', ' ', text).strip()

    @staticmethod
    def parse_kntt_structure(raw_text: str, filename: str) -> List[Document]:
        """
        Phân tích cấu trúc KNTT: Tên sách -> Chủ đề -> Bài.
        Regex được cải tiến để bắt cả Markdown (## Chủ đề...)
        """
        lines = raw_text.split('\n')
        docs = []
        
        # Regex cải tiến: Bắt chấp nhận dấu #, *, khoảng trắng ở đầu dòng
        # Bắt: "## Chủ đề 1:", "**Chủ đề A**", "Chủ đề 1."
        topic_pattern = re.compile(r'^[\#\*\s]*(?:Chủ đề|CHỦ ĐỀ)\s+([0-9A-Za-z]+)(?:[:\.]|\s+)(.+?)(?:[\#\*]*)$', re.IGNORECASE)
        
        # Bắt: "### Bài 1:", "Bài 5.", "**Bài 17**"
        lesson_pattern = re.compile(r'^[\#\*\s]*(?:Bài|BÀI)\s+([0-9]+)(?:[:\.]|\s+)(.+?)(?:[\#\*]*)$', re.IGNORECASE)

        current_topic = "Chưa xác định"
        current_lesson = "Chưa xác định"
        
        # Biến cờ (Flag) để biết đã vào vùng nội dung hợp lệ chưa
        in_valid_section = False 
        
        buffer = []
        source_name = os.path.splitext(filename)[0]

        def flush_buffer():
            if buffer and in_valid_section:
                content = "\n".join(buffer).strip()
                if len(content) > 50: # Chỉ lưu nếu nội dung đủ dài > 50 ký tự
                    doc = Document(
                        page_content=content,
                        metadata={
                            "source": source_name,
                            "topic": current_topic,
                            "lesson": current_lesson,
                            "chunk_uid": str(uuid.uuid4())
                        }
                    )
                    docs.append(doc)

        for line in lines:
            clean_line = KNTT_Processor.normalize_text(line)
            if not clean_line: continue

            # Kiểm tra Chủ đề
            topic_match = topic_pattern.match(clean_line)
            if topic_match:
                flush_buffer() # Lưu nội dung bài cũ
                t_id = topic_match.group(1).strip()
                t_name = topic_match.group(2).strip()
                current_topic = f"Chủ đề {t_id}: {t_name}"
                current_lesson = "Đang chờ bài..."
                in_valid_section = False # Reset flag, chờ gặp Bài mới bật lên
                buffer = []
                continue

            # Kiểm tra Bài
            lesson_match = lesson_pattern.match(clean_line)
            if lesson_match:
                flush_buffer()
                l_id = lesson_match.group(1).strip()
                l_name = lesson_match.group(2).strip()
                current_lesson = f"Bài {l_id}: {l_name}"
                
                # QUAN TRỌNG: Chỉ khi có Chủ đề VÀ Bài thì mới bật cờ ghi dữ liệu
                if "Chưa xác định" not in current_topic:
                    in_valid_section = True
                buffer = []
                continue

            # Chỉ lưu nội dung nếu đang ở trong vùng hợp lệ (Topic + Lesson)
            if in_valid_section:
                buffer.append(clean_line)
        
        flush_buffer() # Lưu đoạn cuối
        return docs

class VectorStoreManager:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(model_name=AppConfig.EMBEDDING_MODEL)
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)

    def build_db(self, uploaded_files):
        if not os.path.exists(AppConfig.UPLOAD_DIR): os.makedirs(AppConfig.UPLOAD_DIR)
        
        all_docs = []
        status = st.empty()
        progress_bar = st.progress(0)

        for i, file in enumerate(uploaded_files):
            status.text(f"⏳ Đang đọc file: {file.name}...")
            file_path = os.path.join(AppConfig.UPLOAD_DIR, file.name)
            with open(file_path, "wb") as f: f.write(file.getbuffer())

            # 1. Parse PDF
            # LƯU Ý: Thầy cần cài đặt biến môi trường LLAMA_CLOUD_API_KEY hoặc set trực tiếp
            if AppConfig.LLAMA_CLOUD_API_KEY.startswith("llx-"):
                os.environ["LLAMA_CLOUD_API_KEY"] = AppConfig.LLAMA_CLOUD_API_KEY
                
            try:
                parser = LlamaParse(result_type="markdown", language="vi") # Dùng markdown để giữ cấu trúc tốt hơn
                documents = parser.load_data(file_path)
                raw_text = documents[0].text
            except Exception as e:
                st.error(f"Lỗi LlamaParse file {file.name}: {e}")
                continue

            # 2. Xử lý Logic KNTT
            status.text(f"⚙️ Đang cấu trúc hóa: {file.name}...")
            kntt_docs = KNTT_Processor.parse_kntt_structure(raw_text, file.name)
            
            if not kntt_docs:
                st.warning(f"⚠️ File {file.name}: Không tìm thấy cấu trúc 'Chủ đề -> Bài'. Kiểm tra lại file PDF.")
                continue

            # 3. Chia nhỏ chunk
            chunks = self.text_splitter.split_documents(kntt_docs)
            all_docs.extend(chunks)
            progress_bar.progress((i + 1) / len(uploaded_files))

        if not all_docs:
            st.error("❌ Không tạo được dữ liệu nào hợp lệ! Vui lòng kiểm tra file PDF đầu vào.")
            return None

        # 4. Lưu Vector DB & BM25
        status.text("💾 Đang lưu vào bộ nhớ...")
        if not os.path.exists(AppConfig.VECTOR_DB_DIR): os.makedirs(AppConfig.VECTOR_DB_DIR)
        
        # Save FAISS
        db = FAISS.from_documents(all_docs, self.embeddings)
        db.save_local(AppConfig.VECTOR_DB_DIR)
        
        # Save BM25 Docs (Pickle)
        with open(AppConfig.BM25_PATH, "wb") as f:
            pickle.dump(all_docs, f)

        status.empty()
        progress_bar.empty()
        return db

    def load_db(self):
        if os.path.exists(AppConfig.VECTOR_DB_DIR) and os.path.exists(os.path.join(AppConfig.VECTOR_DB_DIR, "index.faiss")):
            return FAISS.load_local(AppConfig.VECTOR_DB_DIR, self.embeddings, allow_dangerous_deserialization=True)
        return None

# =============================
# 3. RAG ENGINE (HYBRID SEARCH)
# =============================

class RAGEngine:
    @staticmethod
    def get_retriever(vector_db):
        # 1. FAISS Retriever
        faiss_retriever = vector_db.as_retriever(search_kwargs={"k": 4})
        
        # 2. BM25 Retriever
        bm25_retriever = None
        if os.path.exists(AppConfig.BM25_PATH):
            try:
                with open(AppConfig.BM25_PATH, "rb") as f:
                    docs = pickle.load(f)
                bm25_retriever = BM25Retriever.from_documents(docs)
                bm25_retriever.k = 4
            except Exception:
                pass
        
        # 3. Ensemble
        if bm25_retriever:
            return EnsembleRetriever(
                retrievers=[bm25_retriever, faiss_retriever],
                weights=[0.4, 0.6]
            )
        return faiss_retriever

    @staticmethod
    def generate_response(client, retriever, query):
        # A. Truy xuất
        docs = retriever.invoke(query)
        
        # B. Rerank (Nếu có thư viện Flashrank)
        if HAS_FLASHRANK and docs:
            try:
                ranker = Ranker(model_name="ms-marco-MiniLM-L-12-v2", cache_dir="./opt")
                rerank_req = RerankRequest(query=query, passages=[
                    {"id": d.metadata.get("chunk_uid", "0"), "text": d.page_content, "meta": d.metadata} 
                    for d in docs
                ])
                results = ranker.rank(rerank_req)
                # Lấy top 3 và map lại format document
                final_results = results[:3]
                context_str = ""
                for r in final_results:
                    meta = r['meta']
                    src = f"{meta.get('source')} → {meta.get('topic')} → {meta.get('lesson')}"
                    context_str += f"\n[Nguồn: {src}]\nNội dung: {r['text']}\n---\n"
            except Exception as e:
                # Fallback nếu lỗi rerank
                context_str = "\n---\n".join([f"[Nguồn: {d.metadata.get('source')} -> {d.metadata.get('topic')} -> {d.metadata.get('lesson')}]\n{d.page_content}" for d in docs[:3]])
        else:
            context_str = "\n---\n".join([f"[Nguồn: {d.metadata.get('source')} -> {d.metadata.get('topic')} -> {d.metadata.get('lesson')}]\n{d.page_content}" for d in docs[:3]])

        # C. Tạo Prompt
        full_prompt = f"""{AppConfig.SYSTEM_PROMPT}
        
        DỮ LIỆU THAM KHẢO:
        {context_str}
        
        CÂU HỎI: {query}
        """

        # D. Gọi LLM Stream
        try:
            chat_completion = client.chat.completions.create(
                messages=[{"role": "user", "content": full_prompt}],
                model=AppConfig.LLM_MODEL,
                stream=True,
            )
            for chunk in chat_completion:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        except Exception as e:
            yield f"Lỗi kết nối LLM: {str(e)}"

# =============================
# 4. GIAO DIỆN CHÍNH (UI)
# =============================

def main():
    if not DEPENDENCIES_OK:
        st.error(f"❌ Lỗi thư viện: {IMPORT_ERROR}. Vui lòng kiểm tra requirements.txt")
        st.stop()

    # --- Sidebar ---
    with st.sidebar:
        st.image(AppConfig.LOGO_PROJECT if os.path.exists(AppConfig.LOGO_PROJECT) else "https://via.placeholder.com/150", width=100)
        st.title("🗂️ QUẢN LÝ DỮ LIỆU")
        
        uploaded_files = st.file_uploader("Nạp SGK (PDF)", type=["pdf"], accept_multiple_files=True)
        
        if st.button("🚀 Xây dựng Tri thức (Build RAG)"):
            if not uploaded_files:
                st.warning("Vui lòng chọn file PDF!")
            elif not AppConfig.GROQ_API_KEY.startswith("gsk_"):
                st.error("Chưa cấu hình API Key Groq trong code!")
            else:
                manager = VectorStoreManager()
                with st.spinner("Đang phân tích cấu trúc SGK..."):
                    db = manager.build_db(uploaded_files)
                    if db:
                        st.success("✅ Đã học xong! Sẵn sàng trả lời.")
                        st.session_state.retriever_engine = RAGEngine.get_retriever(db)
                        st.rerun()

    # --- Main Chat ---
    st.title("🤖 TRỢ LÝ HỌC TẬP (CHUẨN KNTT)")
    
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "Chào bạn! Tôi là trợ lý AI chuyên về SGK Tin học. Bạn cần tìm hiểu Chủ đề hay Bài nào?"}]

    # Load DB khi khởi động lại trang
    if "retriever_engine" not in st.session_state:
        manager = VectorStoreManager()
        db = manager.load_db()
        if db:
            st.session_state.retriever_engine = RAGEngine.get_retriever(db)

    # Hiển thị lịch sử chat
    for msg in st.session_state.messages:
        avatar = "🧑‍🎓" if msg["role"] == "user" else "🤖"
        st.chat_message(msg["role"], avatar=avatar).markdown(msg["content"])

    # Xử lý input
    if prompt := st.chat_input("Hỏi về bài học..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user", avatar="🧑‍🎓").markdown(prompt)

        with st.chat_message("assistant", avatar="🤖"):
            if "retriever_engine" not in st.session_state:
                st.error("⚠️ Chưa có dữ liệu! Vui lòng nạp SGK ở menu bên trái.")
            else:
                try:
                    client = Groq(api_key=AppConfig.GROQ_API_KEY)
                    response_gen = RAGEngine.generate_response(client, st.session_state.retriever_engine, prompt)
                    st.write_stream(response_gen)
                    # Lưu lại response vào history (cần ghép chuỗi stream nếu muốn lưu - ở đây demo hiển thị trực tiếp)
                except Exception as e:
                    st.error(f"Lỗi hệ thống: {e}")

if __name__ == "__main__":
    main()
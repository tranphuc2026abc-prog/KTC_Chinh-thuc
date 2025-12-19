import os
import glob
import base64
import streamlit as st
import shutil
import pickle
import re
import uuid
import unicodedata 
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Generator

# =========================================================
# 1. IMPORT & KIỂM TRA THƯ VIỆN (GIỮ NGUYÊN)
# =========================================================
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
    
    # Rerank optimization (Có kiểm tra lỗi nếu chưa cài)
    try:
        from flashrank import Ranker, RerankRequest
        HAS_FLASHRANK = True
    except ImportError:
        HAS_FLASHRANK = False
        
    DEPENDENCIES_OK = True
except ImportError as e:
    DEPENDENCIES_OK = False
    IMPORT_ERROR = str(e)

# =========================================================
# 2. CẤU HÌNH HỆ THỐNG (APP CONFIG)
# =========================================================

st.set_page_config(
    page_title="KTC Chatbot - THCS & THPT Phạm Kiệt",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

class AppConfig:
    # --- API KEY (Thầy điền vào đây) ---
    GROQ_API_KEY = "gsk_..." 
    LLAMA_CLOUD_API_KEY = "llx-..." # Key LlamaParse nếu dùng
    
    # Model Config
    LLM_MODEL = "llama-3.3-70b-versatile" 
    EMBEDDING_MODEL = "dangvantuan/vietnamese-embedding"
    
    # Paths
    UPLOAD_DIR = "PDF_KNOWLEDGE"
    VECTOR_DB_DIR = "FAISS_DB"
    BM25_PATH = os.path.join(VECTOR_DB_DIR, "bm25_docs.pkl")
    LOGO_PROJECT = "LOGO.jpg" 
    
    # Prompt
    SYSTEM_PROMPT = """Bạn là Trợ lý học tập môn Tin học, hỗ trợ giáo viên và học sinh trường Phạm Kiệt theo SGK Kết nối tri thức (KNTT).
    
    NHIỆM VỤ:
    - Trả lời câu hỏi dựa trên ngữ cảnh (Context) được cung cấp.
    - TUYỆT ĐỐI KHÔNG bịa đặt thông tin.
    
    YÊU CẦU ĐẦU RA:
    1. Nội dung: Giải thích rõ ràng, sư phạm, phù hợp lứa tuổi học sinh.
    2. Trích dẫn nguồn BẮT BUỘC: Cuối câu trả lời phải ghi rõ nguồn theo định dạng: 
       (Nguồn: Tên sách > Chủ đề... > Bài...)
    3. Nếu không tìm thấy thông tin trong Context: Trả lời "Dựa trên tài liệu SGK hiện có, tôi chưa tìm thấy thông tin này."
    """

# =========================================================
# 3. XỬ LÝ DỮ LIỆU & RAG (PHẦN ĐIỀU CHỈNH KỸ THUẬT)
# =========================================================

class VectorStoreManager:
    def __init__(self):
        # Embeddings cho tiếng Việt
        self.embeddings = HuggingFaceEmbeddings(model_name=AppConfig.EMBEDDING_MODEL)
        # Splitter cắt nhỏ chunk (dùng sau khi đã parse cấu trúc)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=150,
            separators=["\n\n", "\n", ".", " ", ""]
        )

    def _normalize_text(self, text: str) -> str:
        """Làm sạch văn bản cơ bản"""
        if not text: return ""
        text = unicodedata.normalize("NFC", text)
        return re.sub(r'\s+', ' ', text).strip()

    def _parse_kntt_logic(self, raw_text: str, filename: str) -> List[Document]:
        """
        LOGIC MỚI: Tách Chủ đề -> Bài.
        Sử dụng Regex linh hoạt để bắt tiêu đề trong PDF/Markdown.
        """
        lines = raw_text.split('\n')
        structured_docs = []
        
        # Regex bắt tiêu đề (Chấp nhận cả Markdown ##, **, và chữ thường/hoa)
        # Bắt: "Chủ đề 1:", "## CHỦ ĐỀ A", "Chủ đề 3. Máy tính"
        topic_pattern = re.compile(r'^[\#\*\s]*(?:Chủ đề|CHỦ ĐỀ)\s+([0-9A-Za-z]+)(?:[:\.]|\s+)(.+?)(?:[\#\*]*)$', re.IGNORECASE)
        
        # Bắt: "Bài 1:", "### BÀI 5.", "Bài 17:"
        lesson_pattern = re.compile(r'^[\#\*\s]*(?:Bài|BÀI)\s+([0-9]+)(?:[:\.]|\s+)(.+?)(?:[\#\*]*)$', re.IGNORECASE)

        # Trạng thái
        current_topic = None
        current_lesson = None
        buffer = []
        
        # Tên nguồn (Bỏ đuôi .pdf)
        source_name = os.path.splitext(filename)[0]

        def commit_buffer():
            """Lưu đoạn văn hiện tại nếu đủ thông tin nguồn"""
            if buffer and current_topic and current_lesson:
                content = "\n".join(buffer).strip()
                if len(content) > 20: # Bỏ qua đoạn quá ngắn
                    doc = Document(
                        page_content=content,
                        metadata={
                            "source": source_name,
                            "topic": current_topic,
                            "lesson": current_lesson,
                            "chunk_uid": str(uuid.uuid4())
                        }
                    )
                    structured_docs.append(doc)

        for line in lines:
            line_clean = self._normalize_text(line)
            if not line_clean: continue

            # 1. Phát hiện Chủ đề
            topic_match = topic_pattern.match(line_clean)
            if topic_match:
                commit_buffer() # Lưu nội dung cũ
                t_id = topic_match.group(1).strip()
                t_name = topic_match.group(2).strip()
                current_topic = f"Chủ đề {t_id}: {t_name}"
                current_lesson = None # Reset bài khi sang chủ đề mới
                buffer = []
                continue

            # 2. Phát hiện Bài
            lesson_match = lesson_pattern.match(line_clean)
            if lesson_match:
                commit_buffer()
                l_id = lesson_match.group(1).strip()
                l_name = lesson_match.group(2).strip()
                current_lesson = f"Bài {l_id}: {l_name}"
                buffer = []
                continue

            # 3. Thu thập nội dung (CHỈ KHI ĐÃ CÓ CHỦ ĐỀ VÀ BÀI)
            if current_topic and current_lesson:
                buffer.append(line_clean)
        
        # Commit đoạn cuối cùng
        commit_buffer()
        return structured_docs

    def build_db(self, uploaded_files):
        """Xây dựng lại Vector DB từ file PDF"""
        if not os.path.exists(AppConfig.UPLOAD_DIR):
            os.makedirs(AppConfig.UPLOAD_DIR)

        all_docs = []
        # Thanh tiến trình UI
        progress_text = "Đang khởi động tiến trình học..."
        my_bar = st.progress(0, text=progress_text)

        for i, uploaded_file in enumerate(uploaded_files):
            # Lưu file
            file_path = os.path.join(AppConfig.UPLOAD_DIR, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # Cập nhật UI
            my_bar.progress((i / len(uploaded_files)), text=f"Đang đọc tài liệu: {uploaded_file.name}")
            
            # 1. Parse PDF (LlamaParse -> Markdown)
            try:
                # Set API Key môi trường nếu cần
                if AppConfig.LLAMA_CLOUD_API_KEY.startswith("llx-"):
                    os.environ["LLAMA_CLOUD_API_KEY"] = AppConfig.LLAMA_CLOUD_API_KEY
                
                parser = LlamaParse(result_type="markdown", language="vi")
                parsed_docs = parser.load_data(file_path)
                
                if parsed_docs:
                    raw_text = parsed_docs[0].text
                    # 2. ÁP DỤNG LOGIC KNTT (FIX MỚI)
                    kntt_docs = self._parse_kntt_logic(raw_text, uploaded_file.name)
                    
                    if kntt_docs:
                        # 3. Split chunk (Giữ metadata)
                        chunks = self.text_splitter.split_documents(kntt_docs)
                        all_docs.extend(chunks)
                    else:
                        st.warning(f"⚠️ File {uploaded_file.name}: Không tìm thấy cấu trúc 'Chủ đề -> Bài'.")
            except Exception as e:
                st.error(f"Lỗi khi đọc file {uploaded_file.name}: {e}")

        my_bar.progress(100, text="Đang mã hóa dữ liệu vào bộ nhớ AI...")
        
        if not all_docs:
            st.error("❌ Không có dữ liệu hợp lệ để tạo Database.")
            return None

        # 4. Lưu FAISS DB
        vector_db = FAISS.from_documents(all_docs, self.embeddings)
        vector_db.save_local(AppConfig.VECTOR_DB_DIR)
        
        # 5. Lưu BM25 Cache (Cho Hybrid Search)
        with open(AppConfig.BM25_PATH, "wb") as f:
            pickle.dump(all_docs, f)
            
        my_bar.empty()
        return vector_db

    def load_db(self):
        """Load DB từ ổ cứng"""
        if os.path.exists(AppConfig.VECTOR_DB_DIR) and os.path.exists(os.path.join(AppConfig.VECTOR_DB_DIR, "index.faiss")):
            return FAISS.load_local(
                AppConfig.VECTOR_DB_DIR, 
                self.embeddings, 
                allow_dangerous_deserialization=True
            )
        return None

# =========================================================
# 4. ENGINE TÌM KIẾM VÀ TRẢ LỜI (RAG ENGINE)
# =========================================================

class RAGEngine:
    @staticmethod
    def get_retriever(vector_db):
        # 1. FAISS Retriever (Semantic)
        faiss_retriever = vector_db.as_retriever(search_kwargs={"k": 5})
        
        # 2. BM25 Retriever (Keyword) - Load từ cache pickle
        bm25_retriever = None
        if os.path.exists(AppConfig.BM25_PATH):
            try:
                with open(AppConfig.BM25_PATH, "rb") as f:
                    docs = pickle.load(f)
                bm25_retriever = BM25Retriever.from_documents(docs)
                bm25_retriever.k = 5
            except:
                pass
        
        # 3. Hybrid (Ensemble)
        if bm25_retriever:
            return EnsembleRetriever(
                retrievers=[bm25_retriever, faiss_retriever],
                weights=[0.4, 0.6]
            )
        return faiss_retriever

    @staticmethod
    def generate_response(client, retriever, query):
        # Bước 1: Retrieve
        docs = retriever.invoke(query)
        
        # Bước 2: Rerank (Nếu có thư viện Flashrank)
        final_docs = docs
        if HAS_FLASHRANK and docs:
            try:
                ranker = Ranker(model_name="ms-marco-MiniLM-L-12-v2", cache_dir="./opt")
                rerank_request = RerankRequest(query=query, passages=[
                    {"id": d.metadata.get("chunk_uid", "0"), "text": d.page_content, "meta": d.metadata} 
                    for d in docs
                ])
                results = ranker.rank(rerank_request)
                # Chuyển đổi lại format
                final_docs = []
                for r in results[:3]: # Lấy top 3 tốt nhất
                    final_docs.append(Document(page_content=r['text'], metadata=r['meta']))
            except Exception:
                final_docs = docs[:3] # Fallback
        else:
            final_docs = docs[:3]

        # Bước 3: Tạo Context string với Metadata chuẩn
        context_text = ""
        for d in final_docs:
            source = d.metadata.get('source', 'N/A')
            topic = d.metadata.get('topic', 'N/A')
            lesson = d.metadata.get('lesson', 'N/A')
            
            context_text += f"\n---\n[NGUỒN: {source} > {topic} > {lesson}]\nNội dung: {d.page_content}\n"

        # Bước 4: Tạo Prompt
        full_prompt = f"""{AppConfig.SYSTEM_PROMPT}
        
        THÔNG TIN NGỮ CẢNH (CONTEXT):
        {context_text}
        
        CÂU HỎI CỦA HỌC SINH: {query}
        
        TRẢ LỜI:"""

        # Bước 5: Gọi LLM (Stream)
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
            yield f"Lỗi kết nối AI: {str(e)}"

# =========================================================
# 5. GIAO DIỆN CHÍNH (MAIN UI) - GIỮ NGUYÊN
# =========================================================

def main():
    if not DEPENDENCIES_OK:
        st.error(f"❌ Thiếu thư viện: {IMPORT_ERROR}")
        return

    # --- Sidebar ---
    with st.sidebar:
        # Logo project
        if os.path.exists(AppConfig.LOGO_PROJECT):
            st.image(AppConfig.LOGO_PROJECT, width=120)
        else:
            st.image("https://via.placeholder.com/150", width=100)
            
        st.title("🗂️ KHO TRI THỨC")
        st.markdown("---")
        
        uploaded_files = st.file_uploader(
            "Nạp SGK (PDF)", 
            type=["pdf"], 
            accept_multiple_files=True
        )
        
        if st.button("🔄 Cập nhật Tri thức", use_container_width=True):
            if uploaded_files:
                if not AppConfig.GROQ_API_KEY.startswith("gsk_"):
                     st.error("⚠️ Vui lòng điền API KEY vào code!")
                else:
                    manager = VectorStoreManager()
                    with st.spinner("Đang cấu trúc hóa dữ liệu..."):
                        db = manager.build_db(uploaded_files)
                        if db:
                            st.success("✅ Đã học xong!")
                            st.session_state.vector_db = db
                            # Xóa cache retriever cũ
                            if "retriever_engine" in st.session_state:
                                del st.session_state.retriever_engine
                            st.rerun()
            else:
                st.warning("⚠️ Vui lòng chọn file PDF!")

        st.markdown("---")
        st.info("Hệ thống RAG hỗ trợ tra cứu SGK KNTT theo chuẩn: \nChủ đề -> Bài.")

    # --- Main Chat ---
    st.title("🤖 TRỢ LÝ HỌC TẬP TIN HỌC")
    
    # Init Chat History
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Init Retriever Engine (Load DB)
    if "retriever_engine" not in st.session_state:
        manager = VectorStoreManager()
        db = manager.load_db()
        if db:
            st.session_state.retriever_engine = RAGEngine.get_retriever(db)
            st.toast("✅ Dữ liệu SGK đã sẵn sàng!", icon="📚")

    # Display Chat
    for msg in st.session_state.messages:
        bot_avatar = AppConfig.LOGO_PROJECT if os.path.exists(AppConfig.LOGO_PROJECT) else "🤖"
        avatar = "🧑‍🎓" if msg["role"] == "user" else bot_avatar
        with st.chat_message(msg["role"], avatar=avatar):
            st.markdown(msg["content"], unsafe_allow_html=True) 

    # Input Area
    user_input = st.chat_input("Nhập câu hỏi học tập...")
    
    if user_input:
        # Hiển thị câu hỏi User
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user", avatar="🧑‍🎓"):
            st.markdown(user_input)

        # Hiển thị câu trả lời AI
        with st.chat_message("assistant", avatar=AppConfig.LOGO_PROJECT if os.path.exists(AppConfig.LOGO_PROJECT) else "🤖"):
            if "retriever_engine" not in st.session_state:
                 st.warning("⚠️ Chưa có dữ liệu! Vui lòng nạp SGK ở cột trái.")
            else:
                response_placeholder = st.empty()
                full_response = ""
                
                # Gọi Engine
                try:
                    groq_client = Groq(api_key=AppConfig.GROQ_API_KEY)
                    response_gen = RAGEngine.generate_response(
                        groq_client,
                        st.session_state.retriever_engine,
                        user_input
                    )

                    for chunk in response_gen:
                        full_response += chunk
                        response_placeholder.markdown(full_response + "▌", unsafe_allow_html=True)
                    
                    response_placeholder.markdown(full_response, unsafe_allow_html=True)
                    st.session_state.messages.append({"role": "assistant", "content": full_response})
                    
                except Exception as e:
                    st.error(f"Lỗi hệ thống: {e}")

if __name__ == "__main__":
    main()
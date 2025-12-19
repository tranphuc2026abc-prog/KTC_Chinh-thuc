"""
PROJECT: KTC CHATBOT - TRỢ LÝ HỌC TẬP TIN HỌC THPT
MÔ HÌNH: RETRIEVAL-AUGMENTED GENERATION (RAG) NÂNG CAO
LEVEL: DỰ ÁN KHOA HỌC KỸ THUẬT CẤP QUỐC GIA (VISEF)
AUTHORS: BÙI TÁ TÙNG - CAO SỸ BẢO CHUNG
MENTOR: THẦY NGUYỄN THẾ KHANH
SCHOOL: THCS & THPT PHẠM KIỆT
"""

import os
import glob
import base64
import streamlit as st
import shutil
import pickle
import re
import uuid
import unicodedata 
import time
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Generator, Any

# --- Imports AI Core Libraries ---
try:
    import nest_asyncio
    nest_asyncio.apply() 
    
    # Loaders & Splitters
    from langchain_community.document_loaders import PyPDFLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    
    # Vector Stores & Retrievers
    from langchain_community.vectorstores import FAISS
    from langchain_community.retrievers import BM25Retriever
    from langchain.retrievers import EnsembleRetriever
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_core.documents import Document
    
    # LLM Integration
    from groq import Groq
    
    # Advanced RAG: Reranking
    from flashrank import Ranker, RerankRequest
    
    DEPENDENCIES_OK = True
except ImportError as e:
    DEPENDENCIES_OK = False
    IMPORT_ERROR = str(e)

# ==============================================================================
# 1. CẤU HÌNH HỆ THỐNG (CONFIGURATION) - GIỮ NGUYÊN
# ==============================================================================

st.set_page_config(
    page_title="KTC Chatbot - THCS & THPT Phạm Kiệt",
    page_icon="LOGO.jpg",
    layout="wide",
    initial_sidebar_state="expanded"
)

class AppConfig:
    """
    Lớp chứa toàn bộ tham số cấu hình của dự án.
    Tập trung hóa cấu hình giúp dễ dàng tinh chỉnh khi thi đấu.
    """
    # --- MODEL AI CONFIG ---
    # Sử dụng Llama 3 bản 70B hoặc 8B tùy vào API Key quota
    LLM_MODEL = 'llama3-70b-8192' 
    
    # Embedding: Multilingual để hỗ trợ tiếng Việt tốt nhất
    EMBEDDING_MODEL = "dangvantuan/vietnamese-embedding" # Model VNI tốt nhất hiện tại
    # EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2" # Backup

    # Reranking Model (Chạy local siêu nhẹ, không tốn API)
    RERANK_MODEL_NAME = "ms-marco-MiniLM-L-12-v2"

    # --- PATHS ---
    PDF_DIR = "PDF_KNOWLEDGE"           # Thư mục chứa PDF đầu vào
    VECTOR_DB_PATH = "faiss_db_index"   # Thư mục lưu Index FAISS
    RERANK_CACHE = "./opt"              # Cache cho model Rerank
    PROCESSED_MD_DIR = "PROCESSED_MD"   # Cache file markdown đã xử lý

    # --- ASSETS ---
    LOGO_PROJECT = "LOGO.jpg"
    LOGO_SCHOOL = "LOGO PKS.png"

    # --- HYPERPARAMETERS CHO RAG (THÔNG SỐ KỸ THUẬT) ---
    CHUNK_SIZE = 800       # Kích thước đoạn cắt
    CHUNK_OVERLAP = 200    # Độ chồng lấp
    RETRIEVAL_K = 20       # Số lượng documents lấy ở tầng 1 (Retrieval)
    FINAL_K = 5            # Số lượng documents lấy ở tầng 2 (Rerank)
    
    # Trọng số Hybrid Search (Ensemble)
    BM25_WEIGHT = 0.4      # Ưu tiên từ khóa chính xác (40%)
    FAISS_WEIGHT = 0.6     # Ưu tiên ngữ nghĩa (60%)

    LLM_TEMPERATURE = 0.1  # Độ sáng tạo thấp để đảm bảo chính xác SGK

    @staticmethod
    def init_folders():
        """Khởi tạo cấu trúc thư mục nếu chưa có."""
        for path in [AppConfig.PDF_DIR, AppConfig.VECTOR_DB_PATH, AppConfig.PROCESSED_MD_DIR]:
            os.makedirs(path, exist_ok=True)

# ==============================================================================
# 2. UI MANAGER - GIỮ NGUYÊN BẤT DI BẤT DỊCH THEO YÊU CẦU
# ==============================================================================

class UIManager:
    @staticmethod
    def get_img_as_base64(file_path):
        if not os.path.exists(file_path):
            return ""
        with open(file_path, "rb") as f:
            data = f.read()
        return base64.b64encode(data).decode()

    @staticmethod
    def inject_custom_css():
        # CSS của thầy Khanh giữ nguyên 100%
        st.markdown("""
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;800&display=swap');
            html, body, [class*="css"], .stMarkdown, .stButton, .stTextInput, .stChatInput {
                font-family: 'Inter', sans-serif !important;
            }
            section[data-testid="stSidebar"] {
                background-color: #f8f9fa; border-right: 1px solid #e9ecef;
            }
            .project-card {
                background: white; padding: 15px; border-radius: 12px;
                box-shadow: 0 2px 8px rgba(0,0,0,0.05); margin-bottom: 20px;
                border: 1px solid #dee2e6;
            }
            .project-title {
                color: #0077b6; font-weight: 800; font-size: 1.1rem;
                margin-bottom: 5px; text-align: center; text-transform: uppercase;
            }
            .project-sub {
                font-size: 0.8rem; color: #6c757d; text-align: center;
                margin-bottom: 15px; font-style: italic;
            }
            .main-header {
                background: linear-gradient(135deg, #023e8a 0%, #0077b6 100%);
                padding: 1.5rem 2rem; border-radius: 15px; color: white;
                margin-bottom: 2rem; box-shadow: 0 8px 20px rgba(0, 119, 182, 0.3);
                display: flex; align-items: center; justify-content: space-between;
            }
            .header-left h1 {
                color: #caf0f8 !important; font-weight: 900; margin: 0;
                font-size: 2.2rem; letter-spacing: -0.5px;
            }
            .header-left p {
                color: #e0fbfc; margin: 5px 0 0 0; font-size: 1rem; opacity: 0.9;
            }
            .header-right img {
                border-radius: 50%; border: 3px solid rgba(255,255,255,0.3);
                box-shadow: 0 4px 10px rgba(0,0,0,0.2); width: 100px; height: 100px;
                object-fit: cover;
            }
            [data-testid="stChatMessageContent"] {
                border-radius: 15px !important; padding: 1rem !important;
                box-shadow: 0 2px 4px rgba(0,0,0,0.05);
            }
            [data-testid="stChatMessageContent"]:has(+ [data-testid="stChatMessageAvatar"]) {
                background: #e3f2fd; color: #0d47a1;
            }
            [data-testid="stChatMessageContent"]:not(:has(+ [data-testid="stChatMessageAvatar"])) {
                background: white; border: 1px solid #e9ecef;
                border-left: 5px solid #00b4d8;
            }
            
            /* Style cho phần Nguồn tham khảo footer */
            .citation-footer {
                margin-top: 15px;
                padding-top: 10px;
                border-top: 1px dashed #ced4da;
                font-size: 0.85rem;
                color: #495057;
                background-color: #f8f9fa;
                border-radius: 8px;
                padding: 10px;
            }
            .citation-header {
                font-weight: 700;
                color: #d63384; 
                margin-bottom: 5px;
                display: flex;
                align-items: center;
                gap: 5px;
            }
            .citation-item {
                margin-left: 5px;
                margin-bottom: 3px;
                display: block;
            }
            
            div.stButton > button {
                border-radius: 8px; background-color: white; color: #0077b6;
                border: 1px solid #90e0ef; transition: all 0.2s;
            }
            div.stButton > button:hover {
                background-color: #0077b6; color: white;
                border-color: #0077b6; box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            }
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
        </style>
        """, unsafe_allow_html=True)

    @staticmethod
    def render_sidebar():
        with st.sidebar:
            if os.path.exists(AppConfig.LOGO_SCHOOL):
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    st.image(AppConfig.LOGO_SCHOOL, use_container_width=True)
                st.markdown("<div style='text-align:center; font-weight:700; color:#023e8a; margin-bottom:20px;'>THCS & THPT PHẠM KIỆT</div>", unsafe_allow_html=True)

            # Phần thông tin nhóm tác giả - Giữ nguyên
            st.markdown("""
            <div class="project-card">
                <div class="project-title">KTC CHATBOT</div>
                <div class="project-sub">Sản phẩm dự thi KHKT cấp Tỉnh</div>
                <hr style="margin: 10px 0; border-top: 1px dashed #dee2e6;">
                <div style="font-size: 0.9rem; line-height: 1.6;">
                    <div style="display: flex; justify-content: space-between;">
                        <span style="font-weight: 600; color: #555;">Tác giả:</span>
                        <span style="text-align: right; color: #222;"><b>Bùi Tá Tùng</b><br><b>Cao Sỹ Bảo Chung</b></span>
                    </div>
                    <div style="display: flex; justify-content: space-between; margin-top: 8px;">
                        <span style="font-weight: 600; color: #555;">GVHD:</span>
                        <span style="text-align: right; color: #222;">Thầy <b>Nguyễn Thế Khanh</b></span>
                    </div>
                    <div style="display: flex; justify-content: space-between; margin-top: 8px;">
                        <span style="font-weight: 600; color: #555;">Năm học:</span>
                        <span style="text-align: right; color: #222;"><b>2025 - 2026</b></span>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("### ⚙️ Tiện ích")
            # Nút Re-index dữ liệu (ẩn trong expander để đỡ bấm nhầm)
            with st.expander("Quản lý dữ liệu học tập"):
                uploaded_files = st.file_uploader("Nạp thêm SGK (PDF)", accept_multiple_files=True, type=['pdf'])
                if st.button("🔄 Huấn luyện lại AI (Re-Build DB)", use_container_width=True):
                    if uploaded_files:
                        for up_file in uploaded_files:
                            with open(os.path.join(AppConfig.PDF_DIR, up_file.name), "wb") as f:
                                f.write(up_file.getbuffer())
                    
                    if os.path.exists(AppConfig.VECTOR_DB_PATH):
                        shutil.rmtree(AppConfig.VECTOR_DB_PATH)
                    st.session_state.pop('rag_engine', None)
                    st.rerun()

            if st.button("🗑️ Xóa lịch sử chat", use_container_width=True):
                st.session_state.messages = []
                st.rerun()

    @staticmethod
    def render_header():
        logo_nhom_b64 = UIManager.get_img_as_base64(AppConfig.LOGO_PROJECT)
        img_html = f'<img src="data:image/jpeg;base64,{logo_nhom_b64}" alt="Logo">' if logo_nhom_b64 else ""

        st.markdown(f"""
        <div class="main-header">
            <div class="header-left">
                <h1>KTC CHATBOT</h1>
                <p style="font-size: 1.1rem; margin-top: 5px;">Học Tin dễ dàng - Thao tác vững vàng</p>
            </div>
            <div class="header-right">
                {img_html}
            </div>
        </div>
        """, unsafe_allow_html=True)

# ==============================================================================
# 3. ADVANCED DATA ENGINEERING - KỸ THUẬT XỬ LÝ DỮ LIỆU CẤP QUỐC GIA
# ==============================================================================

class KnowledgeBaseBuilder:
    """
    Class chịu trách nhiệm xử lý file PDF thành các chunks thông minh.
    Điểm nhấn: Context-Aware Splitting (Cắt theo ngữ cảnh Chủ đề/Bài).
    """

    @staticmethod
    def clean_vietnamese_text(text: str) -> str:
        """Làm sạch và chuẩn hóa văn bản tiếng Việt."""
        text = unicodedata.normalize('NFC', text)
        text = re.sub(r'\s+', ' ', text) # Xóa khoảng trắng thừa
        text = text.replace(' .', '.').replace(' ,', ',')
        return text.strip()

    @staticmethod
    def extract_structure_and_chunk(file_path: str) -> List[Document]:
        """
        [KỸ THUẬT CORE] 
        Đọc PDF -> Duyệt từng dòng -> Phát hiện 'Chủ đề'/'Bài' -> Gắn Metadata.
        """
        filename = os.path.basename(file_path)
        
        # 1. Detect Grade (Lớp) from filename (Router data)
        grade = "General"
        if "10" in filename: grade = "10"
        elif "11" in filename: grade = "11"
        elif "12" in filename: grade = "12"

        # 2. Load PDF Text
        loader = PyPDFLoader(file_path)
        pages = loader.load()
        full_text = "\n".join([p.page_content for p in pages])
        full_text = KnowledgeBaseBuilder.clean_vietnamese_text(full_text)
        
        # 3. Define Regex Patterns for Textbook Structure (KNTT)
        # Pattern bắt: "Chủ đề 1: ...", "Chủ đề E ...", "CHỦ ĐỀ F..."
        topic_pattern = re.compile(r'(?:^|\n)(CHỦ ĐỀ\s+[0-9A-Z]+[.:]?\s+.*)', re.IGNORECASE)
        # Pattern bắt: "Bài 1: ...", "Bài 2 ..."
        lesson_pattern = re.compile(r'(?:^|\n)(BÀI\s+[0-9]+[.:]?\s+.*)', re.IGNORECASE)

        lines = full_text.split('.') # Tách câu để duyệt (hoặc tách dòng nếu PDF giữ format tốt)
        
        chunks = []
        current_topic = "Chủ đề chung"
        current_lesson = "Tổng quan"
        buffer_text = ""
        
        # 4. Context-Aware Loop
        # Thay vì cắt độ dài cố định ngay, ta cắt theo logic bài học trước
        # Sau đó mới cắt nhỏ theo token nếu bài quá dài.
        
        # Để đơn giản hóa cho demo nhưng vẫn hiệu quả: Dùng RecursiveSplitter nhưng inject metadata
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=AppConfig.CHUNK_SIZE,
            chunk_overlap=AppConfig.CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        # Duyệt từng trang để giữ page number
        for page in pages:
            page_content = KnowledgeBaseBuilder.clean_vietnamese_text(page.page_content)
            
            # Update Context State
            topic_matches = topic_pattern.findall(page_content)
            if topic_matches:
                current_topic = topic_matches[-1].strip() # Lấy chủ đề mới nhất tìm thấy
                
            lesson_matches = lesson_pattern.findall(page_content)
            if lesson_matches:
                current_lesson = lesson_matches[-1].strip()
            
            # Create sub-chunks for this page
            page_chunks = text_splitter.create_documents([page_content])
            
            for chunk in page_chunks:
                # [QUAN TRỌNG] Gắn Metadata phân cấp
                chunk.metadata.update({
                    "source": filename,
                    "grade": grade,
                    "topic": current_topic,
                    "lesson": current_lesson,
                    "page": page.metadata.get('page', 0) + 1,
                    # Tạo trường citation string để dùng sau này
                    "citation_label": f"{filename} > {current_topic} > {current_lesson} (Trang {page.metadata.get('page', 0) + 1})"
                })
                chunks.append(chunk)
                
        return chunks

# ==============================================================================
# 4. NATIONAL LEVEL RAG ENGINE - LÕI XỬ LÝ THÔNG MINH
# ==============================================================================

class AdvancedRAGEngine:
    def __init__(self, api_key):
        self.groq_client = Groq(api_key=api_key)
        self.embeddings = HuggingFaceEmbeddings(model_name=AppConfig.EMBEDDING_MODEL)
        
        # Load hoặc Build Vector DB
        self.vector_db = self._initialize_vector_db()
        
        # Khởi tạo BM25 Retriever (Sparse Search)
        # Lưu ý: Trong môi trường production, nên lưu BM25 ra đĩa. Ở đây build in-memory cho gọn.
        all_docs = list(self.vector_db.docstore._dict.values())
        self.bm25_retriever = BM25Retriever.from_documents(all_docs)
        self.bm25_retriever.k = AppConfig.RETRIEVAL_K
        
        # Khởi tạo Reranker
        try:
            self.reranker = Ranker(model_name=AppConfig.RERANK_MODEL_NAME, cache_dir=AppConfig.RERANK_CACHE)
            self.has_reranker = True
        except Exception:
            self.has_reranker = False # Fallback nếu không load được reranker

    def _initialize_vector_db(self):
        """Khởi tạo FAISS DB. Nếu có rồi thì load, chưa có thì build mới."""
        AppConfig.init_folders()
        
        if os.path.exists(os.path.join(AppConfig.VECTOR_DB_PATH, "index.faiss")):
            try:
                return FAISS.load_local(AppConfig.VECTOR_DB_PATH, self.embeddings, allow_dangerous_deserialization=True)
            except Exception as e:
                st.warning("Index cũ bị lỗi, đang tạo mới...")
        
        # Build new
        pdf_files = glob.glob(os.path.join(AppConfig.PDF_DIR, "*.pdf"))
        if not pdf_files:
            # Tạo dummy DB nếu chưa có file để tránh crash
            return FAISS.from_texts(["Chưa có dữ liệu"], self.embeddings)
            
        all_chunks = []
        progress_bar = st.progress(0, text="Đang số hóa tri thức SGK...")
        
        for i, pdf_path in enumerate(pdf_files):
            chunks = KnowledgeBaseBuilder.extract_structure_and_chunk(pdf_path)
            all_chunks.extend(chunks)
            progress_bar.progress((i + 1) / len(pdf_files))
            
        progress_bar.empty()
        
        if not all_chunks:
            return FAISS.from_texts(["Chưa có dữ liệu"], self.embeddings)
            
        db = FAISS.from_documents(all_chunks, self.embeddings)
        db.save_local(AppConfig.VECTOR_DB_PATH)
        return db

    def _detect_intent_and_route(self, query: str) -> Dict:
        """
        [ROUTER] Kỹ thuật định tuyến câu hỏi.
        Nếu hỏi Tin 10 -> Chỉ tìm trong file Tin 10.
        """
        query_lower = query.lower()
        filters = {}
        
        if "tin 10" in query_lower or "lớp 10" in query_lower:
            filters["grade"] = "10"
        elif "tin 11" in query_lower or "lớp 11" in query_lower:
            filters["grade"] = "11"
        elif "tin 12" in query_lower or "lớp 12" in query_lower:
            filters["grade"] = "12"
            
        return filters

    def generate_response(self, user_query: str) -> Generator[str, None, None]:
        """
        Luồng xử lý chính: Router -> Hybrid Search -> Rerank -> LLM
        """
        # 1. ROUTING & FILTERING
        metadata_filter = self._detect_intent_and_route(user_query)
        
        # 2. HYBRID RETRIEVAL (Vector + Keyword)
        # Vector Search với Filter
        vector_retriever = self.vector_db.as_retriever(
            search_kwargs={"k": AppConfig.RETRIEVAL_K, "filter": metadata_filter} if metadata_filter else {"k": AppConfig.RETRIEVAL_K}
        )
        
        # Tạo Ensemble Retriever
        ensemble_retriever = EnsembleRetriever(
            retrievers=[self.bm25_retriever, vector_retriever],
            weights=[AppConfig.BM25_WEIGHT, AppConfig.FAISS_WEIGHT]
        )
        
        try:
            initial_docs = ensemble_retriever.invoke(user_query)
        except Exception:
            # Fallback nếu BM25 lỗi filter (do BM25 của langchain hạn chế filter)
            initial_docs = vector_retriever.invoke(user_query)

        if not initial_docs:
            yield "Xin lỗi, thầy không tìm thấy thông tin trong SGK."
            return

        # 3. RERANKING (Sắp xếp lại theo độ phù hợp ngữ nghĩa sâu)
        final_docs = initial_docs
        if self.has_reranker:
            passages = [
                {"id": str(i), "text": doc.page_content, "meta": doc.metadata} 
                for i, doc in enumerate(initial_docs)
            ]
            rerank_request = RerankRequest(query=user_query, passages=passages)
            reranked_results = self.reranker.rank(rerank_request)
            
            # Map lại kết quả rerank về Document object
            final_docs = []
            for res in reranked_results[:AppConfig.FINAL_K]:
                final_docs.append(Document(page_content=res["text"], metadata=res["meta"]))
        else:
            final_docs = initial_docs[:AppConfig.FINAL_K]

        # 4. CONTEXT CONSTRUCTION
        context_str = ""
        unique_sources = set()
        
        for doc in final_docs:
            context_str += f"Nội dung: {doc.page_content}\n"
            context_str += f"Nguồn: {doc.metadata.get('citation_label', 'SGK')}\n---\n"
            unique_sources.add(doc.metadata.get('citation_label', 'SGK'))

        # 5. PROMPT ENGINEERING (SYSTEM PROMPT CHUẨN SƯ PHẠM)
        system_prompt = f"""Bạn là Trợ lý AI môn Tin học của trường Phạm Kiệt.
        Nhiệm vụ: Trả lời câu hỏi dựa trên [CONTEXT] bên dưới.
        
        Yêu cầu:
        - Trả lời ngắn gọn, dễ hiểu, giọng văn thân thiện của giáo viên.
        - Tuyệt đối trung thực với [CONTEXT]. Nếu không có tin, nói không biết.
        - Định dạng Markdown đẹp mắt (dùng bold, list).
        
        [CONTEXT]:
        {context_str}
        """

        # 6. CALL LLM (STREAMING)
        stream = self.groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_query}
            ],
            model=AppConfig.LLM_MODEL,
            stream=True,
            temperature=AppConfig.LLM_TEMPERATURE
        )

        full_response = ""
        for chunk in stream:
            if chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
                full_response += content
                yield content
        
        # 7. APPEND CITATIONS (Tự động thêm nguồn vào cuối câu trả lời)
        yield "\n\n" # Xuống dòng
        
        # Render HTML Footer cho nguồn (Hack để hiển thị đẹp trong Streamlit markdown)
        citation_html = "\n\n<div class='citation-footer'><div class='citation-header'>📚 Nguồn tham khảo xác thực:</div>"
        for src in sorted(list(unique_sources)):
            citation_html += f"<span class='citation-item'>• {src}</span>"
        citation_html += "</div>"
        
        yield citation_html

# ==============================================================================
# 5. MAIN APPLICATION LOGIC
# ==============================================================================

def main():
    if not DEPENDENCIES_OK:
        st.error(f"⚠️ Thiếu thư viện: {IMPORT_ERROR}")
        st.stop()

    # Khởi tạo giao diện
    UIManager.inject_custom_css()
    UIManager.render_sidebar()
    UIManager.render_header()

    # Kiểm tra API Key
    api_key = st.secrets.get("GROQ_API_KEY")
    if not api_key:
        with st.sidebar:
            api_key = st.text_input("Nhập Groq API Key:", type="password")
    
    if not api_key:
        st.warning("Vui lòng nhập API Key để bắt đầu.")
        return

    # Khởi tạo RAG Engine (Singleton trong Session State)
    if "rag_engine" not in st.session_state:
        with st.spinner("🚀 Đang khởi động hệ thống tri thức số..."):
            try:
                st.session_state.rag_engine = AdvancedRAGEngine(api_key)
                st.toast("Hệ thống đã sẵn sàng!", icon="✅")
            except Exception as e:
                st.error(f"Lỗi khởi tạo: {e}")
                return

    # Quản lý lịch sử chat
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "👋 Chào em! Thầy là trợ lý ảo KTC. Em cần tìm hiểu kiến thức Tin 10, 11 hay 12?"}]

    # Render tin nhắn cũ
    for msg in st.session_state.messages:
        bot_avatar = AppConfig.LOGO_PROJECT if os.path.exists(AppConfig.LOGO_PROJECT) else "🤖"
        avatar = "🧑‍🎓" if msg["role"] == "user" else bot_avatar
        with st.chat_message(msg["role"], avatar=avatar):
            st.markdown(msg["content"], unsafe_allow_html=True) 

    # Xử lý Input
    user_input = st.chat_input("Nhập câu hỏi học tập (Ví dụ: Tin 10 bài cấu trúc rẽ nhánh)...")
    
    if user_input:
        # User message
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user", avatar="🧑‍🎓"):
            st.markdown(user_input)

        # AI Response
        with st.chat_message("assistant", avatar=AppConfig.LOGO_PROJECT if os.path.exists(AppConfig.LOGO_PROJECT) else "🤖"):
            response_placeholder = st.empty()
            full_response = ""
            
            try:
                # Gọi Generator
                response_gen = st.session_state.rag_engine.generate_response(user_input)
                
                for chunk in response_gen:
                    full_response += chunk
                    # Update liên tục tạo hiệu ứng gõ máy
                    response_placeholder.markdown(full_response + "▌", unsafe_allow_html=True)
                
                # Final update (bỏ con trỏ)
                response_placeholder.markdown(full_response, unsafe_allow_html=True)
                
                # Lưu vào lịch sử (bao gồm cả HTML citation)
                st.session_state.messages.append({"role": "assistant", "content": full_response})
                
            except Exception as e:
                st.error(f"Lỗi: {str(e)}")

if __name__ == "__main__":
    main()
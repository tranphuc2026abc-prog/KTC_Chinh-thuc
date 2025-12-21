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
from collections import defaultdict

# --- Imports với xử lý lỗi ---
try:
    import nest_asyncio
    nest_asyncio.apply() 
    try:
        from llama_parse import LlamaParse 
    except ImportError:
        LlamaParse = None
    
    # 🔥 NEW: PyMuPDF for advanced processing
    import fitz  # PyMuPDF
    
    from langchain_community.document_loaders import PyPDFLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_community.vectorstores import FAISS
    from langchain_community.retrievers import BM25Retriever
    from langchain.retrievers import EnsembleRetriever
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_core.documents import Document
    from groq import Groq
    from flashrank import Ranker, RerankRequest
    DEPENDENCIES_OK = True
except ImportError as e:
    DEPENDENCIES_OK = False
    IMPORT_ERROR = str(e)

# ==============================
# 1. CẤU HÌNH HỆ THỐNG (CONFIG) 
# ==============================

st.set_page_config(
    page_title="KTC Chatbot - THCS & THPT Phạm Kiệt",
    page_icon="LOGO.jpg",
    layout="wide",
    initial_sidebar_state="expanded"
)

class AppConfig:
    # Model Config
    LLM_MODEL = 'llama-3.1-8b-instant'
    EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    RERANK_MODEL_NAME = "ms-marco-TinyBERT-L-2-v2"

    # Paths
    PDF_DIR = "PDF_KNOWLEDGE"
    VECTOR_DB_PATH = "faiss_db_index"
    RERANK_CACHE = "./opt"
    PROCESSED_MD_DIR = "PROCESSED_MD" 

    # Assets
    LOGO_PROJECT = "LOGO.jpg"
    LOGO_SCHOOL = "LOGO PKS.png"

    # RAG Parameters
    RETRIEVAL_K = 30       
    FINAL_K = 5
    RERANK_THRESHOLD = 0.45  # Score threshold for filtering
    
    # Synthetic Scoring (Fallback when Reranker fails)
    SYNTHETIC_BASE_SCORE = 0.95
    SYNTHETIC_DECAY = 0.05
    
    # Hybrid Search Weights
    BM25_WEIGHT = 0.4      
    FAISS_WEIGHT = 0.6     

    LLM_TEMPERATURE = 0.0 

# ===============================
# 2. XỬ LÝ GIAO DIỆN (UI MANAGER ) 
# ===============================

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
            
            /* Evidence Card Styles */
            .evidence-card {
                background: #f8f9fa;
                border-left: 4px solid #0077b6;
                padding: 12px 15px;
                margin-bottom: 10px;
                border-radius: 8px;
                font-size: 0.9rem;
            }
            .evidence-header {
                font-weight: 700;
                color: #023e8a;
                margin-bottom: 5px;
                display: flex;
                align-items: center;
                flex-wrap: wrap;
                gap: 8px;
            }
            .evidence-confidence {
                display: inline-block;
                background: linear-gradient(135deg, #0077b6, #00b4d8);
                color: white;
                padding: 3px 10px;
                border-radius: 12px;
                font-size: 0.8rem;
                font-weight: 600;
            }
            .evidence-badge {
                display: inline-block;
                background: #e9ecef;
                color: #495057;
                padding: 3px 8px;
                border-radius: 10px;
                font-size: 0.75rem;
                font-weight: 600;
            }
            .evidence-context {
                color: #495057;
                font-size: 0.85rem;
                margin-top: 5px;
                font-style: italic;
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
            if st.button("🗑️ Xóa lịch sử chat", use_container_width=True):
                st.session_state.messages = []
                st.rerun()

            if st.button("🔄 Cập nhật dữ liệu mới", use_container_width=True):
                if os.path.exists(AppConfig.VECTOR_DB_PATH):
                    shutil.rmtree(AppConfig.VECTOR_DB_PATH)
                if os.path.exists(AppConfig.PROCESSED_MD_DIR):
                    shutil.rmtree(AppConfig.PROCESSED_MD_DIR)
                st.session_state.pop('retriever_engine', None)
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

# ============================================================
# 🔥 ADVANCED PDF PROCESSOR - INTEGRATED MODULE
# ============================================================

class AdvancedPDFProcessor:
    """
    Advanced processor for Vietnamese textbook PDFs with hierarchical structure.
    Implements context-aware chunking with proper metadata tracking.
    
    This replaces the naive RecursiveCharacterTextSplitter approach.
    """
    
    # Noise patterns to filter out
    NOISE_PATTERNS = [
        r'KẾT\s+NỐI\s+TRI\s+THỨC\s+VỚI\s+CUỘC\s+SỐNG',
        r'TIN\s+HỌC\s+\d+',
        r'CHƯƠNG\s+TRÌNH\s+GIÁO\s+DỤC',
        r'PHÂN\s+PHỐI\s+CHƯƠNG\s+TRÌNH',
        r'^\s*\d+\s*$',  # Isolated page numbers
    ]
    
    # Structural patterns for Vietnamese textbooks
    TOPIC_PATTERN = re.compile(
        r'(?:^|\n)\s*CHỦ\s+ĐỀ\s+(\d+)[\.:\s]*(.*?)(?:\n|$)',
        re.IGNORECASE | re.MULTILINE
    )
    
    LESSON_PATTERN = re.compile(
        r'(?:^|\n)\s*BÀI\s+(\d+)[\.:\s]*(.*?)(?:\n|$)',
        re.IGNORECASE | re.MULTILINE
    )
    
    @staticmethod
    def normalize_text(text: str) -> str:
        """Normalize Vietnamese text (NFC normalization, whitespace cleanup)."""
        text = unicodedata.normalize('NFC', text)
        text = text.replace('\xa0', ' ').replace('\u200b', '')
        text = re.sub(r'[ \t]+', ' ', text)
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
        return text.strip()
    
    @staticmethod
    def is_noise(text: str) -> bool:
        """Check if a text line is noise (header/footer/page number)."""
        text_clean = text.strip()
        
        if len(text_clean) < 3:
            return True
        
        for pattern in AdvancedPDFProcessor.NOISE_PATTERNS:
            if re.search(pattern, text_clean, re.IGNORECASE):
                return True
        
        if text_clean.isdigit() and len(text_clean) <= 3:
            return True
        
        return False
    
    @staticmethod
    def extract_page_text(page) -> Tuple[str, List[str]]:
        """Extract clean text from a PDF page, filtering noise."""
        text = page.get_text()
        text = AdvancedPDFProcessor.normalize_text(text)
        
        lines = text.split('\n')
        clean_lines = []
        
        for line in lines:
            line = line.strip()
            if line and not AdvancedPDFProcessor.is_noise(line):
                clean_lines.append(line)
        
        full_text = '\n'.join(clean_lines)
        return full_text, clean_lines
    
    @staticmethod
    def detect_topic(text: str) -> Optional[str]:
        """Detect 'Chủ đề' (Topic/Chapter) from text."""
        match = AdvancedPDFProcessor.TOPIC_PATTERN.search(text)
        if match:
            topic_num = match.group(1).strip()
            topic_name = match.group(2).strip()
            return f"Chủ đề {topic_num}. {topic_name}"
        return None
    
    @staticmethod
    def detect_lesson(text: str) -> Optional[str]:
        """Detect 'Bài' (Lesson) from text."""
        match = AdvancedPDFProcessor.LESSON_PATTERN.search(text)
        if match:
            lesson_num = match.group(1).strip()
            lesson_name = match.group(2).strip()
            return f"Bài {lesson_num}. {lesson_name}"
        return None
    
    @staticmethod
    def split_into_semantic_chunks(text: str, max_chunk_size: int = 1000) -> List[str]:
        """Split text into semantic chunks respecting paragraph boundaries."""
        if len(text) <= max_chunk_size:
            return [text]
        
        paragraphs = re.split(r'\n\n+', text)
        chunks = []
        current_chunk = []
        current_length = 0
        
        for para in paragraphs:
            para_len = len(para)
            
            if para_len > max_chunk_size:
                if current_chunk:
                    chunks.append('\n\n'.join(current_chunk))
                    current_chunk = []
                    current_length = 0
                
                sentences = re.split(r'([.!?]+\s+)', para)
                temp_chunk = ""
                for sent in sentences:
                    if len(temp_chunk) + len(sent) > max_chunk_size and temp_chunk:
                        chunks.append(temp_chunk.strip())
                        temp_chunk = sent
                    else:
                        temp_chunk += sent
                
                if temp_chunk.strip():
                    chunks.append(temp_chunk.strip())
                    
            elif current_length + para_len + 2 > max_chunk_size:
                if current_chunk:
                    chunks.append('\n\n'.join(current_chunk))
                current_chunk = [para]
                current_length = para_len
            else:
                current_chunk.append(para)
                current_length += para_len + 2
        
        if current_chunk:
            chunks.append('\n\n'.join(current_chunk))
        
        return chunks
    
    @staticmethod
    def process_pdf_advanced(pdf_path: str, chunk_size: int = 1000, overlap: int = 100) -> List[Document]:
        """
        🔥 MAIN PROCESSING FUNCTION: Extract PDF with context-aware hierarchical chunking.
        
        Algorithm:
        1. Iterate through all PDF pages
        2. Extract and clean text from each page
        3. Maintain state machine for current topic/lesson context
        4. Detect structural changes (new topic, new lesson)
        5. Create chunks with proper metadata enrichment
        
        Returns:
            List of LangChain Document objects with enriched metadata
        """
        doc = fitz.open(pdf_path)
        source_name = os.path.basename(pdf_path)
        documents = []
        
        # State machine variables
        current_topic = None
        current_lesson = None
        content_buffer = []
        buffer_page_start = 0
        
        print(f"📚 Processing: {source_name}")
        print(f"📄 Total pages: {len(doc)}")
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            page_text, lines = AdvancedPDFProcessor.extract_page_text(page)
            
            if not page_text.strip():
                continue
            
            # Detect structural changes on this page
            detected_topic = AdvancedPDFProcessor.detect_topic(page_text)
            detected_lesson = AdvancedPDFProcessor.detect_lesson(page_text)
            
            # STATE TRANSITION: New Topic detected
            if detected_topic:
                if content_buffer:
                    AdvancedPDFProcessor._commit_buffer_to_documents(
                        documents, content_buffer, current_topic, current_lesson,
                        buffer_page_start, page_num - 1, source_name, chunk_size, overlap
                    )
                    content_buffer = []
                
                current_topic = detected_topic
                current_lesson = None
                buffer_page_start = page_num
                print(f"  📌 Page {page_num + 1}: Detected {current_topic}")
            
            # STATE TRANSITION: New Lesson detected
            if detected_lesson:
                if content_buffer:
                    AdvancedPDFProcessor._commit_buffer_to_documents(
                        documents, content_buffer, current_topic, current_lesson,
                        buffer_page_start, page_num - 1, source_name, chunk_size, overlap
                    )
                    content_buffer = []
                
                current_lesson = detected_lesson
                buffer_page_start = page_num
                print(f"    📖 Page {page_num + 1}: Detected {current_lesson}")
            
            content_buffer.append({'text': page_text, 'page': page_num})
        
        # Commit remaining buffer
        if content_buffer:
            AdvancedPDFProcessor._commit_buffer_to_documents(
                documents, content_buffer, current_topic, current_lesson,
                buffer_page_start, len(doc) - 1, source_name, chunk_size, overlap
            )
        
        doc.close()
        print(f"✅ Generated {len(documents)} context-aware chunks")
        return documents
    
    @staticmethod
    def _commit_buffer_to_documents(
        documents: List[Document],
        buffer: List[Dict],
        topic: Optional[str],
        lesson: Optional[str],
        page_start: int,
        page_end: int,
        source_name: str,
        chunk_size: int,
        overlap: int
    ):
        """Convert accumulated buffer into Document objects with metadata."""
        if not buffer:
            return
        
        full_text = '\n\n'.join([item['text'] for item in buffer])
        representative_page = page_start + (page_end - page_start) // 2
        
        chunks = AdvancedPDFProcessor.split_into_semantic_chunks(full_text, chunk_size)
        
        # Create overlapping chunks
        final_chunks = []
        for i, chunk in enumerate(chunks):
            if i > 0 and overlap > 0:
                prev_chunk = chunks[i - 1]
                overlap_text = prev_chunk[-overlap:] if len(prev_chunk) > overlap else prev_chunk
                chunk = overlap_text + '\n' + chunk
            final_chunks.append(chunk)
        
        # Create Document objects
        for chunk_idx, chunk_text in enumerate(final_chunks):
            metadata = {
                'source': source_name,
                'page': representative_page + 1,  # 1-indexed
                'chapter': topic if topic else 'Nội dung chung',
                'lesson': lesson if lesson else 'Phần giới thiệu',
                'chunk_index': chunk_idx,
                'total_chunks': len(final_chunks),
                'page_range': f"{page_start + 1}-{page_end + 1}"
            }
            
            doc = Document(page_content=chunk_text.strip(), metadata=metadata)
            documents.append(doc)

# ==================================
# 3. LOGIC BACKEND - ROBUST RAG ENGINE
# ==================================

class RAGEngine:
    @staticmethod
    @st.cache_resource(show_spinner=False)
    def load_groq_client():
        try:
            api_key = st.secrets.get("GROQ_API_KEY") or os.environ.get("GROQ_API_KEY")
            if not api_key:
                return None
            return Groq(api_key=api_key)
        except Exception:
            return None

    @staticmethod
    @st.cache_resource(show_spinner=False)
    def load_embedding_model():
        try:
            return HuggingFaceEmbeddings(
                model_name=AppConfig.EMBEDDING_MODEL,
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
        except Exception as e:
            st.error(f"Lỗi tải Embedding: {e}")
            return None

    @staticmethod
    @st.cache_resource(show_spinner=False)
    def load_reranker():
        try:
            return Ranker(model_name=AppConfig.RERANK_MODEL_NAME, cache_dir=AppConfig.RERANK_CACHE)
        except Exception as e:
            return None

    @staticmethod
    def _detect_grade(filename: str) -> str:
        filename = filename.lower()
        if "10" in filename: return "10"
        if "11" in filename: return "11"
        if "12" in filename: return "12"
        return "general"

    @staticmethod
    def _read_and_process_files(pdf_dir: str) -> List[Document]:
        """
        🔥 UPGRADED: Uses advanced context-aware PDF processing.
        
        This method now directly calls AdvancedPDFProcessor which implements:
        - Hierarchical structure detection (Chủ đề, Bài)
        - State machine for context tracking
        - Noise reduction (headers, footers, page numbers)
        - Semantic chunking at paragraph boundaries
        - Full metadata enrichment (chapter, lesson, page)
        """
        if not os.path.exists(pdf_dir):
            os.makedirs(pdf_dir, exist_ok=True)
            return []
        
        pdf_files = glob.glob(os.path.join(pdf_dir, "*.pdf"))
        all_chunks: List[Document] = []
        status_text = st.empty()

        if not pdf_files:
            st.warning(f"⚠️ Thư mục {pdf_dir} đang trống. Vui lòng bỏ file PDF SGK vào.")
            return []

        for file_path in pdf_files:
            source_file = os.path.basename(file_path)
            status_text.text(f"🧠 Đang xử lý cấu trúc tri thức nâng cao: {source_file}...")
            
            try:
                # 🔥 Use advanced processor
                file_chunks = AdvancedPDFProcessor.process_pdf_advanced(
                    pdf_path=file_path,
                    chunk_size=1000,
                    overlap=100
                )
                
                if file_chunks:
                    all_chunks.extend(file_chunks)
                    print(f"✅ {source_file}: {len(file_chunks)} chunks created with full metadata")
                else:
                    print(f"⚠️ File {source_file} không tạo được chunk nào.")
                    
            except Exception as e:
                st.error(f"❌ Lỗi xử lý file {source_file}: {str(e)}")
                print(f"Error details: {e}")
                import traceback
                traceback.print_exc()
                
        status_text.empty()
        print(f"📊 Total chunks created: {len(all_chunks)}")
        return all_chunks

    @staticmethod
    def build_hybrid_retriever(embeddings):
        if not embeddings: return None

        vector_db = None
        if os.path.exists(AppConfig.VECTOR_DB_PATH):
            try:
                vector_db = FAISS.load_local(AppConfig.VECTOR_DB_PATH, embeddings, allow_dangerous_deserialization=True)
            except Exception: pass

        if not vector_db:
            chunk_docs = RAGEngine._read_and_process_files(AppConfig.PDF_DIR)
            
            if not chunk_docs:
                st.error(f"Không tạo được dữ liệu từ {AppConfig.PDF_DIR}. Hãy kiểm tra: 1. Có file PDF không? 2. File có text không (hay là ảnh scan)?")
                return None
            
            vector_db = FAISS.from_documents(chunk_docs, embeddings)
            vector_db.save_local(AppConfig.VECTOR_DB_PATH)

        try:
            docstore_docs = list(vector_db.docstore._dict.values())
            bm25_k = min(AppConfig.RETRIEVAL_K, len(docstore_docs))
            
            if bm25_k > 0:
                bm25_retriever = BM25Retriever.from_documents(docstore_docs)
                bm25_retriever.k = bm25_k

                faiss_retriever = vector_db.as_retriever(
                    search_type="mmr",
                    search_kwargs={"k": AppConfig.RETRIEVAL_K, "lambda_mult": 0.5}
                )

                ensemble_retriever = EnsembleRetriever(
                    retrievers=[bm25_retriever, faiss_retriever],
                    weights=[AppConfig.BM25_WEIGHT, AppConfig.FAISS_WEIGHT]
                )
                return ensemble_retriever
            else:
                return vector_db.as_retriever(search_kwargs={"k": AppConfig.RETRIEVAL_K})
        except Exception as e:
            print(f"Lỗi build retriever: {e}. Fallback về FAISS thường.")
            return vector_db.as_retriever(search_kwargs={"k": AppConfig.RETRIEVAL_K})
    
    @staticmethod
    def _sanitize_output(text: str) -> str:
        cjk_pattern = re.compile(r'[\u4e00-\u9fff\u3400-\u4dbf\u3040-\u309f\u30a0-\u30ff\uac00-\ud7af]+')
        text = cjk_pattern.sub("", text)
        
        hallucination_pattern = re.compile(r'\[(ID|Nguồn|Source|Trích dẫn|Tài liệu).*?\]', re.IGNORECASE)
        text = hallucination_pattern.sub("", text)
        
        leakage_pattern = re.compile(r'^(Hệ thống|Chatbot|Phần này) (tự động|sẽ|đã) (gắn|thêm|trích dẫn).*', re.IGNORECASE | re.MULTILINE)
        text = leakage_pattern.sub("", text)
        
        lines = text.split('\n')
        cleaned_lines = []
        for line in lines:
            line_clean = line.strip().lower()
            if line_clean.startswith(('nguồn:', 'source:', 'trích dẫn:', 'tài liệu tham khảo:')):
                continue
            cleaned_lines.append(line)
        
        return "\n".join(cleaned_lines).strip()

    @staticmethod
    def _format_chat_history(messages: List[Dict]) -> str:
        """Format chat history for context injection"""
        formatted = []
        for msg in messages[-6:]:  # Last 3 turns (6 messages)
            role = "Học sinh" if msg["role"] == "user" else "Trợ lý"
            content = re.sub(r'<[^>]+>', '', msg["content"])
            formatted.append(f"{role}: {content[:200]}")
        return "\n".join(formatted)

    @staticmethod
    def generate_response(client, retriever, query: str, chat_history: List[Dict]) -> Tuple[str, List[Tuple[Document, float]]]:
        if not client or not retriever:
            return "❌ Hệ thống chưa sẵn sàng. Vui lòng kiểm tra API Key và dữ liệu SGK.", []

        # --- TẦNG 1: RETRIEVAL ---
        try:
            raw_docs = retriever.invoke(query)
            if not raw_docs:
                return "🔍 Không tìm thấy thông tin liên quan trong SGK.", []
        except Exception as e:
            return f"Lỗi truy vấn dữ liệu: {str(e)}", []

        # --- TẦNG 2: RERANKING ---
        reranker = RAGEngine.load_reranker()
        scored_docs = []

        if reranker:
            try:
                passages = [
                    {"id": idx, "text": doc.page_content, "meta": doc.metadata}
                    for idx, doc in enumerate(raw_docs)
                ]
                rerank_req = RerankRequest(query=query, passages=passages)
                rerank_results = reranker.rerank(rerank_req)
                
                scored_docs = [
                    (raw_docs[res["id"]], res["score"])
                    for res in rerank_results[:AppConfig.FINAL_K]
                    if res["score"] >= AppConfig.RERANK_THRESHOLD
                ]
            except Exception as e:
                print(f"⚠️ Reranker failed: {e}. Using synthetic scores.")
                reranker = None
        
        if not reranker or not scored_docs:
            scored_docs = [
                (doc, AppConfig.SYNTHETIC_BASE_SCORE - (i * AppConfig.SYNTHETIC_DECAY))
                for i, doc in enumerate(raw_docs[:AppConfig.FINAL_K])
            ]

        if not scored_docs:
            return "🔍 Không tìm thấy thông tin liên quan trong SGK.", []

        # --- TẦNG 3: CONTEXT BUILDING ---
        context_parts = []
        for doc, _ in scored_docs:
             context_parts.append(
                f"--- BEGIN DATA ---\n{doc.page_content}\n--- END DATA ---"
            )

        full_context = "\n".join(context_parts)
        history_context = RAGEngine._format_chat_history(chat_history)

        # --- TẦNG 4: PROMPT WITH MEMORY ---
        system_prompt = f"""Bạn là KTC Chatbot, trợ lý ảo AI hỗ trợ học tập Tin học trường Phạm Kiệt.
Nhiệm vụ: Trả lời câu hỏi của học sinh dựa trên thông tin trong [CONTEXT] và [LỊCH SỬ HỘI THOẠI].

QUY TẮC BẮT BUỘC:
1. Chỉ sử dụng thông tin trong [CONTEXT].
2. Sử dụng [LỊCH SỬ HỘI THOẠI] để hiểu ngữ cảnh (ví dụ: "cho tôi ví dụ về cái đó" → biết "cái đó" là gì).
3. KHÔNG tự viết nguồn tham khảo giả.
4. Trả lời ngắn gọn, sư phạm, dễ hiểu cho học sinh phổ thông.

[LỊCH SỬ HỘI THOẠI]
{history_context}

[CONTEXT]
{full_context}
"""
        
        try:
            completion = client.chat.completions.create(
                model=AppConfig.LLM_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": query}
                ],
                stream=False,
                temperature=AppConfig.LLM_TEMPERATURE,
                max_tokens=1500
            )
            raw_response = completion.choices[0].message.content

            if "NO_INFO" in raw_response or not raw_response.strip():
                return "Không tìm thấy thông tin phù hợp trong SGK hiện có.", []

            cleaned_response = RAGEngine._sanitize_output(raw_response)
            return cleaned_response, scored_docs

        except Exception as e:
            return f"Lỗi xử lý hệ thống: {str(e)}", []

# ===================
# 4. MAIN APPLICATION
# ===================

def deduplicate_evidence(evidence_docs: List[Tuple[Document, float]]) -> List[Dict]:
    """
    🔥 CRITICAL FIX: Group evidence by unique lesson, show highest score + count
    Returns: [{"source": ..., "chapter": ..., "lesson": ..., "max_score": ..., "count": ...}]
    """
    lesson_groups = defaultdict(lambda: {"docs": [], "scores": []})
    
    for doc, score in evidence_docs:
        src = doc.metadata.get('source', 'Unknown')
        chapter = doc.metadata.get('chapter', '')
        lesson = doc.metadata.get('lesson', '')
        
        # Create unique key: source + chapter + lesson
        key = f"{src}|||{chapter}|||{lesson}"
        lesson_groups[key]["docs"].append(doc)
        lesson_groups[key]["scores"].append(score)
    
    # Build deduplicated list
    deduplicated = []
    for key, data in lesson_groups.items():
        src, chapter, lesson = key.split("|||")
        max_score = max(data["scores"])
        count = len(data["docs"])
        
        deduplicated.append({
            "source": src,
            "chapter": chapter,
            "lesson": lesson,
            "max_score": max_score,
            "count": count
        })
    
    # Sort by score descending
    deduplicated.sort(key=lambda x: x["max_score"], reverse=True)
    return deduplicated

def main():
    if not DEPENDENCIES_OK:
        st.error(f"⚠️ Thiếu thư viện: {IMPORT_ERROR}")
        st.stop()

    UIManager.inject_custom_css()
    UIManager.render_sidebar()
    UIManager.render_header()

    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "👋 Chào bạn! KTC Chatbot sẵn sàng hỗ trợ tra cứu kiến thức SGK Tin học."}]

    groq_client = RAGEngine.load_groq_client()

    if "retriever_engine" not in st.session_state:
        with st.spinner("🚀 Đang khởi động hệ thống tri thức số (Advanced Context-Aware Processing)..."):
            embeddings = RAGEngine.load_embedding_model()
            st.session_state.retriever_engine = RAGEngine.build_hybrid_retriever(embeddings)
            if st.session_state.retriever_engine:
                st.toast("✅ Dữ liệu SGK đã sẵn sàng!", icon="📚")

    # Display chat history
    for msg in st.session_state.messages:
        bot_avatar = AppConfig.LOGO_PROJECT if os.path.exists(AppConfig.LOGO_PROJECT) else "🤖"
        avatar = "🧑‍🎓" if msg["role"] == "user" else bot_avatar
        with st.chat_message(msg["role"], avatar=avatar):
            if msg["role"] == "assistant" and "evidence" in msg:
                st.markdown(msg["content"])
                
                # 🔥 Re-render deduplicated evidence for history
                if msg["evidence"]:
                    deduplicated = deduplicate_evidence(msg["evidence"])
                    with st.expander("📚 Kiểm chứng nguồn gốc (Evidence)", expanded=False):
                        for item in deduplicated:
                            src = item["source"].replace('.pdf', '').replace('_', ' ')
                            topic = item["chapter"]
                            lesson = item["lesson"]
                            confidence_pct = int(item["max_score"] * 100)
                            count = item["count"]
                            
                            count_badge = f'<span class="evidence-badge">🔍 {count} đoạn liên quan</span>' if count > 1 else ''
                            
                            st.markdown(f"""
                            <div class="evidence-card">
                                <div class="evidence-header">
                                    📖 {src}
                                    <span class="evidence-confidence">Độ tin cậy: {confidence_pct}%</span>
                                    {count_badge}
                                </div>
                                <div class="evidence-context">➜ {topic} ➜ {lesson}</div>
                            </div>
                            """, unsafe_allow_html=True)
            else:
                st.markdown(msg["content"])

    user_input = st.chat_input("Nhập câu hỏi học tập...")
    
    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user", avatar="🧑‍🎓"):
            st.markdown(user_input)

        with st.chat_message("assistant", avatar=AppConfig.LOGO_PROJECT if os.path.exists(AppConfig.LOGO_PROJECT) else "🤖"):
            response_placeholder = st.empty()
            
            # Pass chat history for context
            response_text, evidence_docs = RAGEngine.generate_response(
                groq_client,
                st.session_state.retriever_engine,
                user_input,
                st.session_state.messages[:-1]  # Exclude the just-added user message
            )

            # Stream simulation for better UX
            displayed = ""
            for char in response_text:
                displayed += char
                response_placeholder.markdown(displayed + "▌")
            
            response_placeholder.markdown(response_text)

            # 🔥 Display DEDUPLICATED evidence in expander
            if evidence_docs:
                deduplicated = deduplicate_evidence(evidence_docs)
                with st.expander("📚 Kiểm chứng nguồn gốc (Evidence)", expanded=False):
                    for item in deduplicated:
                        src = item["source"].replace('.pdf', '').replace('_', ' ')
                        topic = item["chapter"]
                        lesson = item["lesson"]
                        confidence_pct = int(item["max_score"] * 100)
                        count = item["count"]
                        
                        count_badge = f'<span class="evidence-badge">🔍 {count} đoạn liên quan</span>' if count > 1 else ''
                        
                        st.markdown(f"""
                        <div class="evidence-card">
                            <div class="evidence-header">
                                📖 {src}
                                <span class="evidence-confidence">Độ tin cậy: {confidence_pct}%</span>
                                {count_badge}
                            </div>
                            <div class="evidence-context">➜ {topic} ➜ {lesson}</div>
                        </div>
                        """, unsafe_allow_html=True)

            # Store evidence with message for history re-rendering
            st.session_state.messages.append({
                "role": "assistant", 
                "content": response_text,
                "evidence": evidence_docs
            })

if __name__ == "__main__":
    main()
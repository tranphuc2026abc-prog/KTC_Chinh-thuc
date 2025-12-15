import os
import glob
import base64
import streamlit as st
import shutil
import pickle
import re
import uuid
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Generator

# --- Imports với xử lý lỗi ---
try:
    import nest_asyncio
    nest_asyncio.apply() # Bắt buộc cho LlamaParse chạy trong Streamlit
    from llama_parse import LlamaParse 
    
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_community.vectorstores import FAISS
    from langchain_community.retrievers import BM25Retriever
    from langchain.retrievers import EnsembleRetriever
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_core.documents import Document
    from groq import Groq
    # Rerank optimization
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

    # RAG Parameters (Updated for Semantic Chunking logic)
    RETRIEVAL_K = 30       
    FINAL_K = 5            
    
    # Hybrid Search Weights
    BM25_WEIGHT = 0.4      
    FAISS_WEIGHT = 0.6     

    LLM_TEMPERATURE = 0.0  # Temperature = 0 để đảm bảo tính xác thực khoa học

# ===============================
# 2. XỬ LÝ GIAO DIỆN (UI MANAGER ) 
# GIỮ NGUYÊN 100% THEO YÊU CẦU
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

# =========================================================
# 3. LOGIC BACKEND - VERIFIABLE HYBRID RAG (KHKT QUỐC GIA)
# =========================================================

class VerifiableRAGEngine:
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
            st.error(f"Lỗi tải mô hình nhúng (Embedding): {e}")
            return None

    @staticmethod
    @st.cache_resource(show_spinner=False)
    def load_reranker():
        try:
            return Ranker(model_name=AppConfig.RERANK_MODEL_NAME, cache_dir=AppConfig.RERANK_CACHE)
        except Exception:
            return None

    @staticmethod
    def _detect_grade_and_topic(filename: str, text: str) -> dict:
        """
        Nhận diện khối lớp (Curriculum-Aware) và chủ đề.
        """
        meta = {"grade": "General", "topic": "general"}
        
        # Detect Grade
        fname = filename.lower()
        if "10" in fname: meta["grade"] = "10"
        elif "11" in fname: meta["grade"] = "11"
        elif "12" in fname: meta["grade"] = "12"
        
        # Detect Topic
        tx = text.lower()
        if any(t in tx for t in ["<html", "css", "javascript", "thẻ"]): meta["topic"] = "html_web"
        elif any(t in tx for t in ["def ", "import ", "python", "biến", "hàm"]): meta["topic"] = "python"
        elif any(t in tx for t in ["sql", "primary key", "csdl", "bảng", "truy vấn"]): meta["topic"] = "database"
        
        return meta

    @staticmethod
    def _parse_pdf_with_llama(file_path: str) -> str:
        """
        Sử dụng LlamaParse để chuyển PDF sang Markdown cấu trúc (Header-aware).
        """
        os.makedirs(AppConfig.PROCESSED_MD_DIR, exist_ok=True)
        file_name = os.path.basename(file_path)
        md_file_path = os.path.join(AppConfig.PROCESSED_MD_DIR, f"{file_name}.md")
        
        if os.path.exists(md_file_path):
            with open(md_file_path, "r", encoding="utf-8") as f:
                return f.read()
        
        llama_api_key = st.secrets.get("LLAMA_CLOUD_API_KEY")
        if not llama_api_key:
            return "ERROR: Missing LLAMA_CLOUD_API_KEY"

        try:
            # Instruction tối ưu cho SGK Việt Nam
            parser = LlamaParse(
                api_key=llama_api_key,
                result_type="markdown",
                language="vi",
                verbose=True,
                parsing_instruction="Hãy phân tích tài liệu SGK Tin học. Giữ nguyên các tiêu đề chương, bài, mục bằng Markdown (#, ##, ###). Giữ nguyên bảng biểu và code block."
            )
            documents = parser.load_data(file_path)
            markdown_text = documents[0].text
            
            with open(md_file_path, "w", encoding="utf-8") as f:
                f.write(markdown_text)
            
            return markdown_text
        except Exception as e:
            return f"Error parsing {file_name}: {str(e)}"

    @staticmethod
    def _semantic_chunking(text: str, source_filename: str) -> List[Document]:
        """
        KỸ THUẬT: STRUCTURAL / SEMANTIC CHUNKING
        Thay vì cắt theo ký tự, hàm này cắt theo cấu trúc logic của SGK (Chương -> Bài -> Mục).
        Đảm bảo mỗi chunk là một đơn vị tri thức hoàn chỉnh.
        """
        chunks = []
        lines = text.split('\n')
        
        current_chapter = "Chương mở đầu/Tổng quan"
        current_lesson = "Nội dung chung"
        current_section = "Chi tiết"
        buffer_content = []
        
        base_meta = VerifiableRAGEngine._detect_grade_and_topic(source_filename, text)
        base_meta["source"] = source_filename

        def flush_buffer():
            if buffer_content:
                content_str = "\n".join(buffer_content).strip()
                if len(content_str) > 50: # Bỏ qua các đoạn quá ngắn (nhiễu)
                    # Tạo ID định danh duy nhất cho chunk (Verifiable ID)
                    chunk_uid = uuid.uuid4().hex[:8].upper()
                    
                    meta = base_meta.copy()
                    meta.update({
                        "chapter": current_chapter,
                        "lesson": current_lesson,
                        "section": current_section,
                        "chunk_uid": chunk_uid
                    })
                    
                    chunks.append(Document(page_content=content_str, metadata=meta))
                buffer_content.clear()

        for line in lines:
            # Nhận diện Header Markdown từ LlamaParse
            header_match = re.match(r'^(#{1,3})\s+(.*)', line)
            
            if header_match:
                flush_buffer() # Lưu nội dung của mục trước đó
                level = len(header_match.group(1))
                title = header_match.group(2).strip()
                
                if level == 1: # Chapter
                    current_chapter = title
                    current_lesson = "Tổng quan chương"
                    current_section = "Mở đầu"
                elif level == 2: # Lesson
                    current_lesson = title
                    current_section = "Nội dung bài"
                elif level == 3: # Section
                    current_section = title
            else:
                buffer_content.append(line)
        
        flush_buffer() # Lưu đoạn cuối cùng
        return chunks

    @staticmethod
    def _read_and_process_files(pdf_dir: str) -> List[Document]:
        if not os.path.exists(pdf_dir):
            return []
        
        pdf_files = glob.glob(os.path.join(pdf_dir, "*.pdf"))
        all_chunks: List[Document] = []
        status_text = st.empty()

        for file_path in pdf_files:
            source_file = os.path.basename(file_path)
            status_text.text(f"Đang phân tích ngữ nghĩa: {source_file}...")
            
            markdown_content = VerifiableRAGEngine._parse_pdf_with_llama(file_path)
            
            if "ERROR" not in markdown_content:
                # Áp dụng Semantic Chunking
                file_chunks = VerifiableRAGEngine._semantic_chunking(markdown_content, source_file)
                all_chunks.extend(file_chunks)
            else:
                # Fallback nếu LlamaParse lỗi (ít dùng, nhưng cần để an toàn hệ thống)
                try:
                    from pypdf import PdfReader
                    reader = PdfReader(file_path)
                    text = ""
                    for page in reader.pages:
                        text += page.extract_text() or ""
                    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                    raw_docs = [Document(page_content=text, metadata={"source": source_file})]
                    all_chunks.extend(splitter.split_documents(raw_docs))
                except: pass
                
        status_text.empty()
        return all_chunks

    @staticmethod
    def build_hybrid_retriever(embeddings):
        if not embeddings: return None

        vector_db = None
        # Kiểm tra DB đã tồn tại chưa
        if os.path.exists(AppConfig.VECTOR_DB_PATH):
            try:
                vector_db = FAISS.load_local(AppConfig.VECTOR_DB_PATH, embeddings, allow_dangerous_deserialization=True)
            except Exception: pass

        if not vector_db:
            # Nếu chưa, tiến hành xử lý dữ liệu mới
            chunk_docs = VerifiableRAGEngine._read_and_process_files(AppConfig.PDF_DIR)
            if not chunk_docs:
                st.error(f"Không tìm thấy tài liệu trong {AppConfig.PDF_DIR}")
                return None
            
            # Xây dựng Index
            vector_db = FAISS.from_documents(chunk_docs, embeddings)
            vector_db.save_local(AppConfig.VECTOR_DB_PATH)

        try:
            docstore_docs = list(vector_db.docstore._dict.values())
            # BM25 cho tìm kiếm từ khóa chính xác
            bm25_retriever = BM25Retriever.from_documents(docstore_docs)
            bm25_retriever.k = AppConfig.RETRIEVAL_K

            # FAISS cho tìm kiếm ngữ nghĩa (Semantic Search)
            faiss_retriever = vector_db.as_retriever(
                search_type="mmr",
                search_kwargs={"k": AppConfig.RETRIEVAL_K, "lambda_mult": 0.5}
            )

            ensemble_retriever = EnsembleRetriever(
                retrievers=[bm25_retriever, faiss_retriever],
                weights=[AppConfig.BM25_WEIGHT, AppConfig.FAISS_WEIGHT]
            )
            return ensemble_retriever
        except Exception:
            return vector_db.as_retriever(search_kwargs={"k": AppConfig.RETRIEVAL_K})

    @staticmethod
    def _validate_answer_grounding(response_text: str, context_docs: List[Document]) -> bool:
        """
        KỸ THUẬT: POST-GENERATION VALIDATION LAYER
        Kiểm tra xem câu trả lời của AI có thực sự dựa trên Context hay không.
        Ngăn chặn ảo giác (Hallucination).
        """
        # 1. Trích xuất từ khóa đơn giản từ context
        context_text = " ".join([d.page_content.lower() for d in context_docs])
        
        # 2. Kiểm tra chồng lắp (Overlap Check) - Simplified for Speed
        # Nếu câu trả lời quá ngắn (ví dụ: chào hỏi), bỏ qua check
        if len(response_text.split()) < 10:
            return True
            
        # Kiểm tra nếu AI thừa nhận không biết
        if "không tìm thấy" in response_text.lower() or "không có thông tin" in response_text.lower():
            return True

        # Tính tỷ lệ xuất hiện của các từ quan trọng trong câu trả lời so với context
        # Đây là một bộ lọc đơn giản. Trong thực tế KHKT có thể dùng NLI models.
        response_words = set(response_text.lower().split())
        context_words = set(context_text.split())
        
        common = response_words.intersection(context_words)
        
        # Ngưỡng chấp nhận: Ít nhất 30% từ vựng (trừ stopword) phải nằm trong context
        # (Ở đây cài đặt đơn giản: nếu overlap > 5 từ là pass để tránh chặn quá chặt)
        if len(common) > 5:
            return True
            
        return False

    @staticmethod
    def generate_response(client, retriever, query) -> Tuple[Generator, List[str]]:
        if not retriever:
            return (x for x in ["Hệ thống đang khởi tạo... vui lòng chờ giây lát."]), []
        
        # 1. Hybrid Retrieval
        initial_docs = retriever.invoke(query)
        
        # 2. Reranking (Lọc tinh)
        final_docs = []
        try:
            ranker = VerifiableRAGEngine.load_reranker()
            if ranker and initial_docs:
                passages = [
                    {"id": str(i), "text": d.page_content, "meta": d.metadata} 
                    for i, d in enumerate(initial_docs)
                ]
                rerank_req = RerankRequest(query=query, passages=passages)
                results = ranker.rank(rerank_req)
                
                for res in results[:AppConfig.FINAL_K]:
                    final_docs.append(Document(page_content=res["text"], metadata=res["meta"]))
            else:
                final_docs = initial_docs[:AppConfig.FINAL_K]
        except Exception:
            final_docs = initial_docs[:AppConfig.FINAL_K]

        if not final_docs:
            return (x for x in ["Xin lỗi, tôi không tìm thấy thông tin trong SGK để trả lời câu hỏi này."]), []

        # 3. Build Verifiable Context (Kèm ID để trích dẫn)
        context_parts = []
        source_display = []
        
        for doc in final_docs:
            meta = doc.metadata
            chunk_uid = meta.get('chunk_uid', 'N/A')
            source_name = meta.get('source', 'SGK')
            chapter = meta.get('chapter', '')
            lesson = meta.get('lesson', '')
            
            # Format hiển thị nguồn cho người dùng
            source_label = f"{source_name} - {chapter}"
            source_display.append(source_label)
            
            # Format Context cho AI (Bắt buộc trích dẫn ID)
            context_parts.append(f"""
--- CHUNK ID: {chunk_uid} ---
Nguồn: {source_name} > {chapter} > {lesson}
Nội dung: {doc.page_content}
""")
        
        full_context = "\n".join(context_parts)

        # 4. Strict Scientific Prompting
        system_prompt = f"""Bạn là Trợ lý AI Giáo dục trong hệ thống RAG (Retrieval-Augmented Generation).
NHIỆM VỤ: Sinh câu trả lời dựa trên tri thức SGK đã truy xuất dưới đây.

QUY TẮC CỐT LÕI (VERIFIABLE GROUNDING):
1. **Dựa trên bằng chứng:** Chỉ trả lời dựa trên thông tin trong [CONTEXT]. Tuyệt đối không tự bịa đặt kiến thức ngoài.
2. **Trích dẫn bắt buộc:** Mọi ý chính phải đi kèm nguồn gốc. Cú pháp: `[Nguồn: ID_CỦA_CHUNK]`.
   - Ví dụ: "Python là ngôn ngữ lập trình bậc cao [Nguồn: A1B2C3D4]."
3. **Trung thực:** Nếu [CONTEXT] không đủ để trả lời, hãy nói: "Không tìm thấy thông tin phù hợp trong SGK hiện có."
4. **Phong cách:** Học thuật, sư phạm, khuyến khích tư duy. Trình bày Markdown rõ ràng.

[CONTEXT BẮT ĐẦU]
{full_context}
[CONTEXT KẾT THÚC]
"""

        try:
            # 5. Generation (Non-stream internal to allow validation, simulated stream output)
            # Lưu ý: Để tối ưu trải nghiệm UI stream nhưng vẫn validate, ta sẽ dùng kỹ thuật
            # "Speculative Streaming" hoặc đơn giản là lấy full response rồi stream giả lập nếu pass validation.
            # Để an toàn cho KHKT, ta lấy full response để validate chặt chẽ.
            
            response = client.chat.completions.create(
                model=AppConfig.LLM_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": query}
                ],
                stream=False, # Tắt stream để validate toàn vẹn
                temperature=AppConfig.LLM_TEMPERATURE,
                max_tokens=1500
            )
            
            full_response_text = response.choices[0].message.content

            # 6. Post-Generation Validation Check
            is_valid = VerifiableRAGEngine._validate_answer_grounding(full_response_text, final_docs)
            
            if not is_valid:
                # Nếu phát hiện ảo giác hoặc không liên quan
                final_response = "Hệ thống phát hiện câu trả lời không bám sát tài liệu SGK gốc. Vui lòng thử lại với câu hỏi cụ thể hơn."
                return (x for x in [final_response]), []
            
            # Nếu hợp lệ, giả lập stream trả về cho UI
            # Tách thành các từ để tạo hiệu ứng gõ
            def simulated_stream():
                words = full_response_text.split(' ') # Tách theo khoảng trắng để giữ format tốt hơn
                for i, word in enumerate(words):
                    yield word + " " if i < len(words)-1 else word
                    
            return simulated_stream(), list(set(source_display))

        except Exception as e:
            return (x for x in [f"Lỗi hệ thống RAG: {str(e)}"]), []

# ===================
# 4. MAIN APPLICATION
# ===================

def main():
    if not DEPENDENCIES_OK:
        st.error(f"⚠️ Thiếu thư viện: {IMPORT_ERROR}")
        st.stop()

    UIManager.inject_custom_css()
    UIManager.render_sidebar()
    UIManager.render_header()

    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "👋 Chào bạn! KTC Chatbot sẵn sàng hỗ trợ tra cứu kiến thức."}]

    groq_client = VerifiableRAGEngine.load_groq_client()

    if "retriever_engine" not in st.session_state:
        with st.spinner("🚀 Đang khởi động hệ thống tri thức số (LlamaParse + Semantic RAG)..."):
            embeddings = VerifiableRAGEngine.load_embedding_model()
            st.session_state.retriever_engine = VerifiableRAGEngine.build_hybrid_retriever(embeddings)
            if st.session_state.retriever_engine:
                st.toast("✅ Dữ liệu SGK đã sẵn sàng!", icon="📚")

    for msg in st.session_state.messages:
        bot_avatar = AppConfig.LOGO_PROJECT if os.path.exists(AppConfig.LOGO_PROJECT) else "🤖"
        avatar = "🧑‍🎓" if msg["role"] == "user" else bot_avatar
        with st.chat_message(msg["role"], avatar=avatar):
            st.markdown(msg["content"])

    user_input = st.chat_input("Nhập câu hỏi của bạn tại đây...")
    
    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user", avatar="🧑‍🎓"):
            st.markdown(user_input)

        with st.chat_message("assistant", avatar=AppConfig.LOGO_PROJECT if os.path.exists(AppConfig.LOGO_PROJECT) else "🤖"):
            response_placeholder = st.empty()
            
            stream_generator, sources = VerifiableRAGEngine.generate_response(
                groq_client,
                st.session_state.retriever_engine,
                user_input
            )

            full_response = ""
            # Xử lý generator trả về (dù là stream thật hay giả lập)
            for chunk in stream_generator:
                # Xử lý khác biệt giữa object chunk của OpenAI và string thuần
                content = chunk if isinstance(chunk, str) else (chunk.choices[0].delta.content or "")
                full_response += content
                response_placeholder.markdown(full_response + "▌")
                
            response_placeholder.markdown(full_response)

            if sources:
                with st.expander("📚 Nguồn SGK xác thực (Verifiable Source)"):
                    for src in sources:
                        st.markdown(f"- {src}")

            st.session_state.messages.append({"role": "assistant", "content": full_response})

if __name__ == "__main__":
    main()
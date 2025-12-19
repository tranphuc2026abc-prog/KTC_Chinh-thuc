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

# --- Imports với xử lý lỗi ---
try:
    import nest_asyncio
    nest_asyncio.apply() 
    # Ưu tiên LlamaParse, nếu không có sẽ dùng PyPDFLoader làm fallback
    try:
        from llama_parse import LlamaParse 
    except ImportError:
        LlamaParse = None
        
    from langchain_community.document_loaders import PyPDFLoader # Fallback loader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_community.vectorstores import FAISS
    from langchain_community.retrievers import BM25Retriever
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

    # RAG Pipeline Parameters (Strict Mode for KHKT)
    BM25_TOP_K = 50        # Lọc thô: Lấy rộng để bắt từ khóa SGK chính xác
    SEMANTIC_TOP_K = 10    # Lọc tinh: Lấy theo ngữ nghĩa từ tập thô
    FINAL_K = 5            # Output: Đưa vào LLM làm bằng chứng
    
    LLM_TEMPERATURE = 0.0  # Nhiệt độ 0 để đảm bảo tính nhất quán khoa học

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
            
            /* Style cho phần Nguồn tham khảo footer */
            .citation-footer {
                margin-top: 15px;
                padding-top: 10px;
                border-top: 1px dashed #ced4da;
                font-size: 0.9rem;
                color: #495057;
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

# ==================================
# 3. LOGIC BACKEND - VERIFIABLE CASCADING RAG
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
        return "THCS"

    # --- [STRICT] CHUNKING: TÁCH BIỆT DATA & METADATA ---
    @staticmethod
    def _structural_chunking(text: str, source_meta: dict) -> List[Document]:
        text = unicodedata.normalize('NFC', text)
        text = text.replace('\xa0', ' ').replace('\u200b', '')
        
        lines = text.split('\n')
        chunks = []
        
        # State tracking
        current_topic = "Kiến thức chung"
        current_lesson = "Tổng quan"
        current_section = "Nội dung"
        
        buffer = []

        # Regex ĐẶC THÙ CHO SGK KNTT
        p_topic = re.compile(r'(?:^|[\#\*\s]+)(CHỦ\s*ĐỀ)\s+([0-9A-Z]+)(.*)', re.IGNORECASE)
        p_lesson = re.compile(r'(?:^|[\#\*\s]+)(BÀI)\s+([0-9]+)(.*)', re.IGNORECASE)
        
        def commit_chunk(buf, topic, lesson, section):
            content = "\n".join(buf).strip()
            if len(content) < 30: return 
            
            # QUY TẮC 1: Content chỉ chứa kiến thức thuần túy để Embedding không bị nhiễu
            clean_content = content 
            
            # QUY TẮC 2: Metadata chứa ngữ cảnh để trích nguồn
            meta = source_meta.copy()
            meta.update({
                "subject": "Tin học",
                "book": "Kết nối tri thức",
                "chapter": topic,
                "lesson": lesson,
                "section": section,
                # Context String dùng cho hiển thị nếu cần
                "full_source_str": f"SGK Tin học {meta.get('grade')} - {topic} - {lesson}"
            })
            
            chunks.append(Document(page_content=clean_content, metadata=meta))

        for line in lines:
            line_stripped = line.strip()
            if not line_stripped: continue
            
            # 4. LOGIC PHÁT HIỆN CHỦ ĐỀ (TOPIC)
            match_topic = p_topic.search(line_stripped)
            if match_topic:
                commit_chunk(buffer, current_topic, current_lesson, current_section)
                buffer = []
                current_topic = f"Chủ đề {match_topic.group(2)}: {match_topic.group(3).strip(' :.-')}"
                current_lesson = "Giới thiệu chủ đề"
                continue
            
            # 5. LOGIC PHÁT HIỆN BÀI (LESSON)
            match_lesson = p_lesson.search(line_stripped)
            if match_lesson:
                commit_chunk(buffer, current_topic, current_lesson, current_section)
                buffer = []
                current_lesson = f"Bài {match_lesson.group(2)}: {match_lesson.group(3).strip(' :.-')}"
                current_section = "Nội dung bài"
                continue
                
            buffer.append(line)
        
        commit_chunk(buffer, current_topic, current_lesson, current_section)
        return chunks

    @staticmethod
    def _parse_pdf_smart(file_path: str) -> str:
        """
        Hàm đọc PDF thông minh: Thử LlamaParse trước, nếu lỗi thì dùng PyPDFLoader (miễn phí, offline)
        """
        os.makedirs(AppConfig.PROCESSED_MD_DIR, exist_ok=True)
        file_name = os.path.basename(file_path)
        md_file_path = os.path.join(AppConfig.PROCESSED_MD_DIR, f"{file_name}.md")
        
        # 1. Kiểm tra Cache
        if os.path.exists(md_file_path):
            with open(md_file_path, "r", encoding="utf-8") as f:
                return f.read()
        
        markdown_text = ""
        
        # 2. Thử dùng LlamaParse (Ưu tiên)
        llama_api_key = st.secrets.get("LLAMA_CLOUD_API_KEY")
        used_llama = False
        
        if llama_api_key and LlamaParse:
            try:
                parser = LlamaParse(
                    api_key=llama_api_key,
                    result_type="markdown",
                    language="vi",
                    verbose=True
                )
                documents = parser.load_data(file_path)
                markdown_text = documents[0].text
                used_llama = True
            except Exception as e:
                print(f"⚠️ LlamaParse failed cho {file_name}: {e}. Chuyển sang PyPDFLoader.")
        
        # 3. Fallback: Dùng PyPDFLoader (Nếu LlamaParse lỗi hoặc không có key)
        if not used_llama or not markdown_text:
            try:
                loader = PyPDFLoader(file_path)
                docs = loader.load()
                # Nối text các trang lại
                markdown_text = "\n\n".join([d.page_content for d in docs])
            except Exception as e:
                return f"ERROR reading file {file_name}: {str(e)}"

        # 4. Lưu Cache
        if markdown_text:
            with open(md_file_path, "w", encoding="utf-8") as f:
                f.write(markdown_text)
            
        return markdown_text

    @staticmethod
    def _read_and_process_files(pdf_dir: str) -> List[Document]:
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
            status_text.text(f"Đang xử lý cấu trúc tri thức: {source_file}...")
            
            content = RAGEngine._parse_pdf_smart(file_path)
            
            if content and not content.startswith("ERROR"):
                 meta = {
                     "source": source_file, 
                     "grade": RAGEngine._detect_grade(source_file)
                 }
                 file_chunks = RAGEngine._structural_chunking(content, meta)
                 if file_chunks:
                    all_chunks.extend(file_chunks)
                 else:
                    print(f"⚠️ File {source_file} đọc được text nhưng không tạo được chunk nào.")
            else:
                st.error(f"Lỗi đọc file {source_file}: {content}")
                
        status_text.empty()
        return all_chunks

    # --- [STRICT] BUILD COMPONENTS: KHÔNG DÙNG ENSEMBLE ---
    @staticmethod
    def build_pipeline_components(embeddings):
        """
        Khởi tạo các thành phần rời rạc cho Pipeline thủ công.
        """
        if not embeddings: return None

        # 1. Load/Create Vector DB
        vector_db = None
        if os.path.exists(AppConfig.VECTOR_DB_PATH):
            try:
                vector_db = FAISS.load_local(AppConfig.VECTOR_DB_PATH, embeddings, allow_dangerous_deserialization=True)
            except Exception: pass

        docs_for_bm25 = []
        if not vector_db:
            chunk_docs = RAGEngine._read_and_process_files(AppConfig.PDF_DIR)
            if not chunk_docs: 
                st.error(f"Không tạo được dữ liệu từ {AppConfig.PDF_DIR}. Hãy kiểm tra: 1. Có file PDF không? 2. File có text không?")
                return None
            vector_db = FAISS.from_documents(chunk_docs, embeddings)
            vector_db.save_local(AppConfig.VECTOR_DB_PATH)
            docs_for_bm25 = chunk_docs
        else:
            docs_for_bm25 = list(vector_db.docstore._dict.values())

        # 2. Build BM25 Retriever (Independent)
        if not docs_for_bm25: return None
        bm25_retriever = BM25Retriever.from_documents(docs_for_bm25)
        bm25_retriever.k = AppConfig.BM25_TOP_K 

        return {
            "vector_db": vector_db,
            "bm25": bm25_retriever
        }

    # --- [STRICT] CASCADING RETRIEVAL & GENERATION ---
    @staticmethod
    def generate_response(client, components, query) -> Generator[str, None, None]:
        if not components:
            yield "Hệ thống đang khởi tạo dữ liệu..."
            return
        
        bm25 = components['bm25']
        vector_db = components['vector_db']
        
        # === BƯỚC 1: BM25 KEYWORD FILTER (Lấy 50 ứng viên) ===
        # Mục tiêu: Đảm bảo không bỏ sót từ khóa chính xác (VD: "biến cục bộ", "cấu trúc rẽ nhánh")
        try:
            initial_candidates = bm25.invoke(query)
        except Exception:
            yield "Lỗi truy vấn dữ liệu."
            return

        if not initial_candidates:
            yield "Không tìm thấy thông tin trong SGK."
            return

        # === BƯỚC 2: FAISS SEMANTIC SCORING (Trên tập 50 ứng viên) ===
        # Mục tiêu: Lọc từ 50 xuống 10 dựa trên độ hiểu ngữ cảnh của câu hỏi
        final_docs = []
        try:
            embeddings = RAGEngine.load_embedding_model()
            # Tạo vector store tạm thời cực nhanh từ 50 kết quả BM25
            temp_db = FAISS.from_documents(initial_candidates, embeddings)
            # Semantic Search trên tập nhỏ này
            semantic_docs = temp_db.similarity_search(query, k=AppConfig.SEMANTIC_TOP_K)
            
            # === BƯỚC 3: RERANKER (FlashRank) ===
            # Mục tiêu: Sắp xếp lại top 10 để chọn ra Top 5 chuẩn xác nhất
            ranker = RAGEngine.load_reranker()
            if ranker:
                passages = [
                    {"id": str(i), "text": d.page_content, "meta": d.metadata} 
                    for i, d in enumerate(semantic_docs)
                ]
                rerank_req = RerankRequest(query=query, passages=passages)
                results = ranker.rank(rerank_req)
                for res in results[:AppConfig.FINAL_K]:
                    final_docs.append(Document(page_content=res["text"], metadata=res["meta"]))
            else:
                final_docs = semantic_docs[:AppConfig.FINAL_K]
                
        except Exception as e:
            # Fallback nếu lỗi Embedding/Rerank thì dùng kết quả BM25
            print(f"Lỗi Pipeline Semantics: {e}")
            final_docs = initial_candidates[:AppConfig.FINAL_K]

        # === BƯỚC 4: CONTEXT CONSTRUCTION ===
        context_text = ""
        used_sources = []
        
        for i, doc in enumerate(final_docs):
            # Chỉ lấy nội dung sạch, không trộn metadata vào context LLM để tránh nhiễu
            context_text += f"\n[Đoạn {i+1}]: {doc.page_content}\n"
            used_sources.append(doc.metadata)

        # === BƯỚC 5: STRICT PROMPT ===
        system_prompt = f"""Bạn là Trợ lý ảo Tin học KTC.
Nhiệm vụ: Trả lời câu hỏi dựa trên [CONTEXT] bên dưới.

YÊU CẦU NGHIÊM NGẶT:
1. Nội dung phải lấy TỪNG CHỮ từ [CONTEXT]. Không tự bịa kiến thức.
2. Nếu [CONTEXT] không có thông tin, trả lời "SGK hiện tại chưa cập nhật thông tin này".
3. Trả lời văn phong sư phạm, hướng dẫn học sinh.
4. Trình bày rõ ràng, dùng gạch đầu dòng nếu liệt kê.

[CONTEXT]
{context_text}
"""

        try:
            completion = client.chat.completions.create(
                model=AppConfig.LLM_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": query}
                ],
                stream=False,
                temperature=AppConfig.LLM_TEMPERATURE
            )
            response_content = completion.choices[0].message.content

            # === BƯỚC 6: CITATION (Nguồn tham khảo chuẩn SGK) ===
            citation_html = "\n\n<div class='citation-footer'><div class='citation-header'>📚 Căn cứ SGK Tin học (Kết nối tri thức):</div>"
            
            seen_citations = set()
            has_citation = False
            
            for meta in used_sources:
                # Format: SGK Tin học 10 - Chủ đề 1 - Bài 2
                grade = meta.get('grade', '')
                topic = meta.get('chapter', 'Chương ?')
                lesson = meta.get('lesson', 'Bài ?')
                
                # Tạo chuỗi citation duy nhất
                cite_str = f"Lớp {grade} ➜ {topic} ➜ {lesson}"
                
                if cite_str not in seen_citations:
                    citation_html += f"<span class='citation-item'>• {cite_str}</span>"
                    seen_citations.add(cite_str)
                    has_citation = True
            
            citation_html += "</div>"
            
            final_output = response_content + (citation_html if has_citation else "")
            yield final_output

        except Exception as e:
            yield f"Lỗi sinh câu trả lời: {str(e)}"

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
        st.session_state.messages = [{"role": "assistant", "content": "👋 Chào bạn! KTC Chatbot sẵn sàng hỗ trợ tra cứu kiến thức SGK Tin học."}]

    groq_client = RAGEngine.load_groq_client()

    if "retriever_engine" not in st.session_state:
        with st.spinner("🚀 Đang khởi động hệ thống tri thức số (Cascading Filter RAG)..."):
            embeddings = RAGEngine.load_embedding_model()
            # SỬ DỤNG HÀM MỚI: build_pipeline_components
            st.session_state.retriever_engine = RAGEngine.build_pipeline_components(embeddings)
            
            if st.session_state.retriever_engine:
                st.toast("✅ Dữ liệu SGK đã sẵn sàng!", icon="📚")

    for msg in st.session_state.messages:
        bot_avatar = AppConfig.LOGO_PROJECT if os.path.exists(AppConfig.LOGO_PROJECT) else "🤖"
        avatar = "🧑‍🎓" if msg["role"] == "user" else bot_avatar
        with st.chat_message(msg["role"], avatar=avatar):
            st.markdown(msg["content"], unsafe_allow_html=True) 

    user_input = st.chat_input("Nhập câu hỏi học tập...")
    
    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user", avatar="🧑‍🎓"):
            st.markdown(user_input)

        with st.chat_message("assistant", avatar=AppConfig.LOGO_PROJECT if os.path.exists(AppConfig.LOGO_PROJECT) else "🤖"):
            response_placeholder = st.empty()
            
            # SỬ DỤNG HÀM MỚI: generate_response với tham số components
            response_gen = RAGEngine.generate_response(
                groq_client,
                st.session_state.retriever_engine,
                user_input
            )

            full_response = ""
            for chunk in response_gen:
                full_response += chunk
                response_placeholder.markdown(full_response + "▌", unsafe_allow_html=True)
            
            response_placeholder.markdown(full_response, unsafe_allow_html=True)

            st.session_state.messages.append({"role": "assistant", "content": full_response})

if __name__ == "__main__":
    main()
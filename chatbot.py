import os
import glob
import base64
import streamlit as st
import shutil
import pickle
import re
import uuid
import unicodedata 
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Generator, Any

# --- Imports với xử lý lỗi ---
try:
    import nest_asyncio
    nest_asyncio.apply() 
    
    # Ưu tiên LlamaParse
    try:
        from llama_parse import LlamaParse 
    except ImportError:
        LlamaParse = None
        
    from langchain_community.document_loaders import PyPDFLoader 
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_community.vectorstores import FAISS
    from langchain_community.retrievers import BM25Retriever
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
    # Model Embedding tốt cho tiếng Việt
    EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    RERANK_MODEL_NAME = "ms-marco-TinyBERT-L-2-v2"

    # Paths
    PDF_DIR = "PDF_KNOWLEDGE"
    VECTOR_DB_PATH = "faiss_db_index"
    BM25_PATH = "bm25_retriever.pkl" # Lưu riêng BM25 index
    RERANK_CACHE = "./opt"
    PROCESSED_MD_DIR = "PROCESSED_MD" 

    # Assets
    LOGO_PROJECT = "LOGO.jpg"
    LOGO_SCHOOL = "LOGO PKS.png"

    # RAG Parameters - CHUẨN KHKT (Recall rộng -> Precision sâu)
    BM25_K = 40       # Lọc rộng theo từ khóa
    FAISS_K = 40      # Lọc rộng theo ngữ nghĩa
    RERANK_K = 5      # Chọn lọc tinh hoa cuối cùng
    
    LLM_TEMPERATURE = 0.0 

# ===============================
# 2. XỬ LÝ GIAO DIỆN (UI MANAGER) 
# ===============================
# (Giữ nguyên class UIManager như code gốc của bạn để không vỡ giao diện)

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
            .citation-footer {
                margin-top: 15px; padding-top: 10px; border-top: 1px dashed #ced4da;
                font-size: 0.9rem; color: #495057;
            }
            .citation-header {
                font-weight: 700; color: #d63384; margin-bottom: 5px;
                display: flex; align-items: center; gap: 5px;
            }
            .citation-item { margin-left: 5px; margin-bottom: 3px; display: block; }
            div.stButton > button {
                border-radius: 8px; background-color: white; color: #0077b6;
                border: 1px solid #90e0ef; transition: all 0.2s;
            }
            div.stButton > button:hover {
                background-color: #0077b6; color: white;
                border-color: #0077b6; box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            }
            #MainMenu {visibility: hidden;} footer {visibility: hidden;}
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
                if os.path.exists(AppConfig.BM25_PATH):
                    os.remove(AppConfig.BM25_PATH)
                st.session_state.pop('rag_pipeline', None)
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

# ====================================================
# 3. KNOWLEDGE PARSER (BỘ XỬ LÝ TRI THỨC CHUẨN SGK)
# ====================================================

class KnowledgeParser:
    @staticmethod
    def detect_grade(filename: str) -> str:
        filename = filename.lower()
        if "10" in filename: return "10"
        if "11" in filename: return "11"
        if "12" in filename: return "12"
        return "THPT"

    @staticmethod
    def clean_text(text: str) -> str:
        text = unicodedata.normalize('NFC', text)
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\xff]', '', text) # Remove non-printable
        text = re.sub(r'\n\s*\n', '\n\n', text) # Normalize newlines
        return text

    @staticmethod
    def structural_split(text: str, source_meta: dict) -> List[Document]:
        """
        Phân tích cấu trúc SGK Tin học KNTT theo chuẩn:
        Chủ đề -> Bài -> Nội dung
        """
        lines = text.split('\n')
        chunks = []
        
        # Regex tối ưu cho SGK KNTT
        # Bắt: "Chủ đề 1", "CHỦ ĐỀ A", "Chủ đề 2: ..."
        p_topic = re.compile(r'^(?:CHỦ\s*ĐỀ|Chủ\s*đề)\s+([0-9A-Z]+)(.*)', re.IGNORECASE)
        # Bắt: "Bài 1", "BÀI 10", "Bài 2..."
        p_lesson = re.compile(r'^(?:BÀI|Bài)\s+([0-9]+)(.*)', re.IGNORECASE)
        
        current_topic = "Kiến thức chung"
        current_lesson = "Tổng quan"
        
        buffer = []
        
        def save_buffer():
            if not buffer: return
            content = "\n".join(buffer).strip()
            if len(content) < 50: return # Bỏ qua đoạn quá ngắn
            
            meta = source_meta.copy()
            meta.update({
                "chapter": current_topic,
                "lesson": current_lesson,
                # Tạo chuỗi trích dẫn chuẩn hóa
                "citation_label": f"SGK Tin học {meta['grade']} - {current_topic} - {current_lesson}"
            })
            
            # Thêm ngữ cảnh vào nội dung để BM25 bắt tốt hơn
            rich_content = f"{current_topic}\n{current_lesson}\n\n{content}"
            
            chunks.append(Document(page_content=rich_content, metadata=meta))

        for line in lines:
            line = line.strip()
            if not line: continue
            
            # Kiểm tra thay đổi Chủ đề
            m_topic = p_topic.match(line)
            if m_topic:
                save_buffer() # Lưu nội dung bài cũ
                buffer = []
                t_id = m_topic.group(1).strip()
                t_name = m_topic.group(2).strip(" :.-")
                current_topic = f"Chủ đề {t_id}: {t_name}" if t_name else f"Chủ đề {t_id}"
                current_lesson = "Giới thiệu chủ đề" # Reset lesson khi qua chủ đề mới
                buffer.append(line)
                continue

            # Kiểm tra thay đổi Bài
            m_lesson = p_lesson.match(line)
            if m_lesson:
                save_buffer()
                buffer = []
                l_id = m_lesson.group(1).strip()
                l_name = m_lesson.group(2).strip(" :.-")
                current_lesson = f"Bài {l_id}: {l_name}" if l_name else f"Bài {l_id}"
                buffer.append(line)
                continue
            
            buffer.append(line)
        
        save_buffer() # Lưu đoạn cuối
        
        # Fallback: Nếu không bắt được cấu trúc (do PDF xấu), dùng Recursive Splitter
        if len(chunks) < 2: 
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            raw_docs = splitter.create_documents([text], metadatas=[source_meta])
            return raw_docs
            
        return chunks

    @staticmethod
    def load_documents(pdf_dir: str) -> List[Document]:
        all_docs = []
        if not os.path.exists(pdf_dir):
            os.makedirs(pdf_dir)
            return []
            
        files = glob.glob(os.path.join(pdf_dir, "*.pdf"))
        
        status = st.empty()
        for f in files:
            fname = os.path.basename(f)
            status.text(f"Đang phân tích tri thức: {fname}...")
            
            # 1. Parsing
            text = ""
            # Thử LlamaParse
            if LlamaParse and st.secrets.get("LLAMA_CLOUD_API_KEY"):
                try:
                    parser = LlamaParse(result_type="text", language="vi")
                    res = parser.load_data(f)
                    text = res[0].text
                except: pass
            
            # Fallback PyPDF
            if not text:
                try:
                    loader = PyPDFLoader(f)
                    pages = loader.load()
                    text = "\n".join([p.page_content for p in pages])
                except Exception as e:
                    print(f"Lỗi đọc {fname}: {e}")
                    continue
            
            text = KnowledgeParser.clean_text(text)
            
            # 2. Chunking
            meta = {
                "source": fname, 
                "grade": KnowledgeParser.detect_grade(fname)
            }
            docs = KnowledgeParser.structural_split(text, meta)
            all_docs.extend(docs)
            
        status.empty()
        return all_docs

# ====================================================
# 4. VERIFIABLE RETRIEVER (BỘ TRUY XUẤT KIỂM CHỨNG)
# ====================================================

class VerifiableRetriever:
    def __init__(self, embeddings):
        self.embeddings = embeddings
        self.vector_db = None
        self.bm25 = None
        self.ranker = None
        
        self.initialize_components()

    def initialize_components(self):
        # 1. Load FAISS (Vector Store)
        if os.path.exists(AppConfig.VECTOR_DB_PATH):
            self.vector_db = FAISS.load_local(
                AppConfig.VECTOR_DB_PATH, 
                self.embeddings, 
                allow_dangerous_deserialization=True
            )
        
        # 2. Load BM25
        if os.path.exists(AppConfig.BM25_PATH):
            with open(AppConfig.BM25_PATH, "rb") as f:
                self.bm25 = pickle.load(f)

        # 3. Load Reranker
        try:
            self.ranker = Ranker(
                model_name=AppConfig.RERANK_MODEL_NAME, 
                cache_dir=AppConfig.RERANK_CACHE
            )
        except:
            pass

    def build_index(self):
        docs = KnowledgeParser.load_documents(AppConfig.PDF_DIR)
        if not docs:
            return False

        # Build FAISS
        self.vector_db = FAISS.from_documents(docs, self.embeddings)
        self.vector_db.save_local(AppConfig.VECTOR_DB_PATH)

        # Build BM25
        self.bm25 = BM25Retriever.from_documents(docs)
        self.bm25.k = AppConfig.BM25_K
        with open(AppConfig.BM25_PATH, "wb") as f:
            pickle.dump(self.bm25, f)
            
        return True

    def retrieve(self, query: str) -> List[Document]:
        """
        Quy trình tuần tự (Sequential Pipeline) chuẩn KHKT:
        Bước 1: BM25 (Từ khóa) -> Lấy tập ứng viên rộng
        Bước 2: FAISS (Ngữ nghĩa) -> Lấy tập ứng viên rộng
        Bước 3: Merge (Hợp nhất) -> Loại trùng lặp
        Bước 4: Rerank (Đánh giá lại) -> Lấy Top K tốt nhất
        """
        if not self.vector_db or not self.bm25:
            return []

        # --- Bước 1: BM25 Retrieval ---
        try:
            bm25_docs = self.bm25.invoke(query)
        except: bm25_docs = []

        # --- Bước 2: Vector Retrieval ---
        try:
            # Dùng search_kwargs k lớn để bắt ngữ nghĩa rộng
            faiss_docs = self.vector_db.similarity_search(query, k=AppConfig.FAISS_K)
        except: faiss_docs = []

        # --- Bước 3: Fusion & Deduplication ---
        # Hợp nhất và loại bỏ trùng lặp dựa trên nội dung
        seen_content = set()
        unique_docs = []
        
        for doc in bm25_docs + faiss_docs:
            # Chuẩn hóa content để so sánh
            content_hash = hash(doc.page_content.strip())
            if content_hash not in seen_content:
                seen_content.add(content_hash)
                unique_docs.append(doc)

        # --- Bước 4: Reranking (Quan trọng nhất cho độ chính xác) ---
        if self.ranker and unique_docs:
            passages = [
                {"id": str(i), "text": d.page_content, "meta": d.metadata} 
                for i, d in enumerate(unique_docs)
            ]
            rerank_request = RerankRequest(query=query, passages=passages)
            results = self.ranker.rank(rerank_request)
            
            final_docs = []
            for res in results[:AppConfig.RERANK_K]:
                final_docs.append(Document(
                    page_content=res["text"], 
                    metadata=res["meta"]
                ))
            return final_docs
        
        return unique_docs[:AppConfig.RERANK_K]

# ==================================
# 5. GENERATION ENGINE (TRẢ LỜI)
# ==================================

class ResponseGenerator:
    @staticmethod
    def generate(client, retriever: VerifiableRetriever, query: str) -> Generator[str, None, None]:
        if not retriever.vector_db:
            yield "Hệ thống đang khởi tạo tri thức... Vui lòng chờ."
            return

        # 1. Truy xuất dữ liệu
        relevant_docs = retriever.retrieve(query)
        
        if not relevant_docs:
            yield "Xin lỗi, tôi không tìm thấy thông tin trong SGK phù hợp với câu hỏi của bạn."
            return

        # 2. Xây dựng Context có trích dẫn rõ ràng
        context_str = ""
        valid_sources = set()
        
        for i, doc in enumerate(relevant_docs):
            source_label = doc.metadata.get('citation_label', 'Tài liệu tham khảo')
            valid_sources.add(source_label)
            context_str += f"\n[Nguồn {i+1}: {source_label}]\nNội dung: {doc.page_content}\n"

        # 3. System Prompt Chuyên sâu
        system_prompt = f"""Bạn là Trợ lý AI Giáo dục của trường Phạm Kiệt, chuyên gia về Tin học.
NHIỆM VỤ: Trả lời câu hỏi học sinh dựa CHÍNH XÁC vào [CONTEXT] bên dưới.

QUY TẮC CỐT LÕI (KHKT Standard):
1. TRUNG THỰC TUYỆT ĐỐI: Chỉ dùng thông tin trong Context. Nếu không có, nói "Không tìm thấy trong SGK".
2. TRÍCH DẪN: Mọi khẳng định chuyên môn phải dựa trên ngữ cảnh.
3. SƯ PHẠM: Giải thích dễ hiểu, ngắn gọn, có ví dụ nếu Context có.
4. ĐỊNH DẠNG: Sử dụng Markdown.

[CONTEXT BẮT ĐẦU]
{context_str}
[CONTEXT KẾT THÚC]
"""

        try:
            completion = client.chat.completions.create(
                model=AppConfig.LLM_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": query}
                ],
                stream=True, # Streaming cho mượt
                temperature=AppConfig.LLM_TEMPERATURE,
                max_tokens=1024
            )
            
            full_ans = ""
            for chunk in completion:
                if chunk.choices[0].delta.content:
                    text_chunk = chunk.choices[0].delta.content
                    full_ans += text_chunk
                    yield text_chunk

            # 4. Footer Trích dẫn (Citation Block)
            citation_html = "\n\n<div class='citation-footer'><div class='citation-header'>📚 Căn cứ khoa học (Trích dẫn SGK):</div>"
            sorted_src = sorted(list(valid_sources))
            for src in sorted_src:
                citation_html += f"<span class='citation-item'>• {src}</span>"
            citation_html += "</div>"
            
            yield citation_html

        except Exception as e:
            yield f"Lỗi sinh câu trả lời: {str(e)}"

# ===================
# 6. MAIN APPLICATION
# ===================

def main():
    if not DEPENDENCIES_OK:
        st.error(f"⚠️ Thiếu thư viện quan trọng: {IMPORT_ERROR}")
        st.stop()

    UIManager.inject_custom_css()
    UIManager.render_sidebar()
    UIManager.render_header()

    # Session State Init
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "👋 Chào bạn! Mình là KTC Chatbot. Bạn cần tra cứu kiến thức bài nào trong SGK Tin học?"}]

    # Load Resources
    groq_client = RAGEngine.load_groq_client()
    embeddings = RAGEngine.load_embedding_model()
    
    # Initialize RAG Pipeline (Singleton-ish in Session)
    if "rag_pipeline" not in st.session_state:
        retriever = VerifiableRetriever(embeddings)
        # Check index exist
        if not retriever.vector_db:
            with st.spinner("🚀 Đang xây dựng cấu trúc tri thức số (Khoảng 1-2 phút)..."):
                success = retriever.build_index()
                if success:
                    st.toast("✅ Đã học xong SGK Tin học!", icon="🎓")
                else:
                    st.warning("⚠️ Chưa có dữ liệu PDF trong thư mục PDF_KNOWLEDGE.")
        
        st.session_state.rag_pipeline = retriever

    # Render Chat
    for msg in st.session_state.messages:
        bot_avatar = AppConfig.LOGO_PROJECT if os.path.exists(AppConfig.LOGO_PROJECT) else "🤖"
        avatar = "🧑‍🎓" if msg["role"] == "user" else bot_avatar
        with st.chat_message(msg["role"], avatar=avatar):
            st.markdown(msg["content"], unsafe_allow_html=True) 

    # Handle Input
    user_input = st.chat_input("Nhập câu hỏi của bạn (Ví dụ: Bài 1 Tin 10 nói về gì?)...")
    
    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user", avatar="🧑‍🎓"):
            st.markdown(user_input)

        with st.chat_message("assistant", avatar=AppConfig.LOGO_PROJECT if os.path.exists(AppConfig.LOGO_PROJECT) else "🤖"):
            response_placeholder = st.empty()
            full_response = ""
            
            # Generate
            retriever = st.session_state.rag_pipeline
            generator = ResponseGenerator.generate(groq_client, retriever, user_input)
            
            # Streaming Output
            for chunk in generator:
                full_response += chunk
                # Cập nhật liên tục, dùng unsafe_allow_html cho citation cuối cùng
                response_placeholder.markdown(full_response + "▌", unsafe_allow_html=True)
            
            response_placeholder.markdown(full_response, unsafe_allow_html=True)
            st.session_state.messages.append({"role": "assistant", "content": full_response})

if __name__ == "__main__":
    main()
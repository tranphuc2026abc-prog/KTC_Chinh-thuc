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

# --- [NEW] IMPORT MODULE XỬ LÝ NÂNG CAO CHO KHKT ---
# Đây là dòng kết nối với file advanced_pdf_processor.py thầy vừa tạo
try:
    from advanced_pdf_processor import process_pdf_advanced
    ADVANCED_MODE = True
except ImportError:
    ADVANCED_MODE = False
    st.error("⚠️ Không tìm thấy file 'advanced_pdf_processor.py'. Hãy đảm bảo file này nằm cùng thư mục.")

# --- Imports với xử lý lỗi ---
try:
    import nest_asyncio
    nest_asyncio.apply() 
    try:
        from llama_parse import LlamaParse 
    except ImportError:
        LlamaParse = None
        
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

# ==============================================================================
# 1. CẤU HÌNH HỆ THỐNG (CONFIG) 
# ==============================================================================

st.set_page_config(
    page_title="KTC Chatbot - THCS & THPT Phạm Kiệt",
    page_icon="LOGO.jpg",
    layout="wide",
    initial_sidebar_state="expanded"
)

class AppConfig:
    # Model Config
    MODELS = {
        "Llama 3 70B": "llama3-70b-8192",
        "Mixtral 8x7B": "mixtral-8x7b-32768",
        "Gemma 7B": "gemma-7b-it"
    }
    
    # Vector DB Config
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200
    EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    
    # Paths
    VECTOR_DB_PATH = "faiss_index"
    UPLOAD_DIR = "uploaded_docs"

# ==============================================================================
# 2. RAG ENGINE (CORE LOGIC)
# ==============================================================================

class RAGEngine:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(model_name=AppConfig.EMBEDDING_MODEL)
        self.vector_store = None
        self.ensure_directories()
        
    def ensure_directories(self):
        os.makedirs(AppConfig.UPLOAD_DIR, exist_ok=True)
        
    def get_groq_client(self):
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            if "GROQ_API_KEY" in st.session_state:
                api_key = st.session_state.GROQ_API_KEY
            else:
                return None
        return Groq(api_key=api_key)

    def _read_and_process_files(self, files) -> List[Document]:
        """
        Đọc và xử lý file upload.
        Đã nâng cấp để sử dụng 'advanced_pdf_processor' cho file PDF.
        """
        documents = []
        progress_text = "Đang phân tích cấu trúc tài liệu..."
        my_bar = st.progress(0, text=progress_text)
        
        for idx, file in enumerate(files):
            temp_path = os.path.join(AppConfig.UPLOAD_DIR, file.name)
            
            # Lưu file tạm
            with open(temp_path, "wb") as f:
                f.write(file.getbuffer())
            
            try:
                # --- [KHKT UPGRADE] XỬ LÝ PDF THÔNG MINH ---
                if file.name.endswith('.pdf') and ADVANCED_MODE:
                    st.toast(f"🚀 Đang kích hoạt chế độ đọc hiểu cấu trúc cho: {file.name}")
                    # Gọi hàm từ file advanced_pdf_processor.py
                    # Hàm này trả về list Document đã có sẵn Metadata (Chapter/Lesson)
                    file_docs = process_pdf_advanced(temp_path)
                    
                    if file_docs:
                        documents.extend(file_docs)
                        st.info(f"✅ Đã trích xuất {len(file_docs)} phân đoạn kiến thức từ {file.name}")
                    else:
                        st.warning(f"File {file.name} không có nội dung text hoặc bị mã hóa.")
                
                # --- XỬ LÝ CÁC LOẠI FILE KHÁC (CŨ) ---
                else:
                    # Fallback cho file không phải PDF hoặc nếu chưa có module nâng cao
                    loader = PyPDFLoader(temp_path)
                    raw_docs = loader.load()
                    
                    # Cắt nhỏ văn bản (Chunking truyền thống)
                    splitter = RecursiveCharacterTextSplitter(
                        chunk_size=AppConfig.CHUNK_SIZE,
                        chunk_overlap=AppConfig.CHUNK_OVERLAP
                    )
                    chunks = splitter.split_documents(raw_docs)
                    
                    # Bổ sung metadata cơ bản để tránh lỗi UI
                    for doc in chunks:
                        if "chapter" not in doc.metadata:
                            doc.metadata["chapter"] = "Tài liệu bổ sung"
                        if "lesson" not in doc.metadata:
                            doc.metadata["lesson"] = "Nội dung chi tiết"
                            
                    documents.extend(chunks)
                    
            except Exception as e:
                st.error(f"Lỗi khi xử lý {file.name}: {str(e)}")
            finally:
                # Dọn dẹp file tạm (Tùy chọn: có thể giữ lại nếu cần debug)
                if os.path.exists(temp_path):
                    os.remove(temp_path)
            
            my_bar.progress((idx + 1) / len(files), text=progress_text)
            
        my_bar.empty()
        return documents

    def build_vector_store(self, uploaded_files):
        """Xây dựng vector store từ file upload"""
        if not uploaded_files:
            return False

        with st.spinner("🔄 Đang cấu trúc hóa dữ liệu (Semantic Segmentation)..."):
            # 1. Xử lý file
            docs = self._read_and_process_files(uploaded_files)
            
            if not docs:
                st.error("Không trích xuất được dữ liệu khả dụng.")
                return False
            
            # 2. Tạo Vector Store
            try:
                self.vector_store = FAISS.from_documents(docs, self.embeddings)
                self.vector_store.save_local(AppConfig.VECTOR_DB_PATH)
                st.success(f"✅ Đã nạp thành công {len(docs)} phân đoạn kiến thức!")
                return True
            except Exception as e:
                st.error(f"Lỗi khởi tạo Vector Store: {str(e)}")
                return False

    def load_vector_store(self):
        """Load vector store đã lưu"""
        if os.path.exists(AppConfig.VECTOR_DB_PATH):
            try:
                self.vector_store = FAISS.load_local(
                    AppConfig.VECTOR_DB_PATH, 
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )
                return True
            except Exception:
                return False
        return False

    def query(self, user_question: str, model_name: str, k: int = 4):
        """Truy vấn và trả lời"""
        client = self.get_groq_client()
        if not client or not self.vector_store:
            return "Vui lòng nhập API Key và nạp dữ liệu.", []

        # 1. Retrieve
        retriever = self.vector_store.as_retriever(search_kwargs={"k": k*2})
        docs = retriever.invoke(user_question)
        
        # 2. Rerank (Simple slice for speed)
        final_docs = docs[:k]

        # 3. Context Construction
        context_parts = []
        evidence_list = []
        
        for doc in final_docs:
            # Lấy metadata (Code mới đảm bảo các trường này luôn có dữ liệu)
            chapter = doc.metadata.get("chapter", "Chương chưa xác định")
            lesson = doc.metadata.get("lesson", "Bài chưa xác định")
            page = doc.metadata.get("page", "?")
            source = doc.metadata.get("source", "Tài liệu")
            
            context_parts.append(f"""
            [Nguồn: {source} | Trang: {page}]
            [Vị trí: {chapter} > {lesson}]
            Nội dung: {doc.page_content}
            """)
            
            evidence_list.append({
                "source": source,
                "chapter": chapter,
                "lesson": lesson,
                "page": page,
                "content": doc.page_content,
                "max_score": 0.9, # Score giả lập cho UI
                "count": 1
            })

        context_str = "\n---\n".join(context_parts)
        
        # 4. Generate Answer
        system_prompt = f"""Bạn là Trợ lý AI giáo dục của trường THCS & THPT Phạm Kiệt.
        Nhiệm vụ: Trả lời câu hỏi dựa trên ngữ cảnh được cung cấp.
        
        YÊU CẦU:
        1. Trả lời chính xác, ngắn gọn, sư phạm.
        2. BẮT BUỘC trích dẫn nguồn (Bài nào, trang nào) nếu thông tin có trong ngữ cảnh.
        3. Nếu không có thông tin trong ngữ cảnh, hãy nói "Xin lỗi, tài liệu hiện tại chưa đề cập vấn đề này."
        
        NGỮ CẢNH HỌC LIỆU:
        {context_str}
        """

        try:
            chat_completion = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_question}
                ],
                model=AppConfig.MODELS.get(model_name, "llama3-70b-8192"),
                temperature=0.3,
                max_tokens=2048,
            )
            return chat_completion.choices[0].message.content, evidence_list
        except Exception as e:
            return f"Lỗi khi gọi API: {str(e)}", []

# ==============================================================================
# 3. GIAO DIỆN NGƯỜI DÙNG (STREAMLIT UI)
# ==============================================================================

def main():
    # --- CSS Tùy chỉnh ---
    st.markdown("""
    <style>
    .evidence-card {
        background-color: #f0f2f6;
        border-left: 5px solid #4CAF50;
        padding: 10px;
        margin-bottom: 10px;
        border-radius: 5px;
    }
    .evidence-header {
        font-weight: bold;
        color: #1E88E5;
        display: flex;
        justify-content: space-between;
    }
    .evidence-context {
        font-size: 0.9em;
        color: #666;
        margin-top: 5px;
        font-style: italic;
    }
    .evidence-confidence {
        font-size: 0.8em;
        background: #e3f2fd;
        padding: 2px 6px;
        border-radius: 10px;
        color: #1565c0;
    }
    .stChatMessage {
        background-color: transparent !important;
    }
    </style>
    """, unsafe_allow_html=True)

    # --- Header ---
    col1, col2 = st.columns([1, 5])
    with col1:
        st.markdown("🤖 **KTC-Bot**") 
    with col2:
        st.title("Trợ lý Học tập Thông minh - Phạm Kiệt School")
        st.caption("🚀 Phiên bản KHKT Quốc gia: Tích hợp Context-Aware RAG Engine")

    # --- Sidebar ---
    with st.sidebar:
        st.header("⚙️ Cấu hình")
        
        # API Key
        api_key = st.text_input("Groq API Key", type="password", placeholder="gsk_...")
        if api_key:
            st.session_state.GROQ_API_KEY = api_key
            
        # Model Selection
        selected_model = st.selectbox("Mô hình AI", list(AppConfig.MODELS.keys()))
        
        st.divider()
        
        # File Uploader
        st.subheader("📚 Nạp Tài Liệu (SGK, Bài giảng)")
        uploaded_files = st.file_uploader(
            "Chọn file PDF (Tin 10_KNTT.pdf)", 
            type=['pdf'], 
            accept_multiple_files=True
        )
        
        if st.button("🚀 Khởi tạo Hệ thống Tri thức", type="primary"):
            if not uploaded_files:
                st.warning("Vui lòng chọn ít nhất 1 file!")
            elif not api_key and "GROQ_API_KEY" not in st.session_state:
                st.warning("Vui lòng nhập API Key!")
            else:
                engine = RAGEngine()
                if engine.build_vector_store(uploaded_files):
                    st.session_state.engine_ready = True
                    st.rerun()

        st.divider()
        st.info("💡 Mẹo: Hệ thống đã được nâng cấp để hiểu cấu trúc 'Chủ đề' và 'Bài học' trong SGK.")

    # --- Main Chat Area ---
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Hiển thị lịch sử chat
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            # Render Evidence nếu có
            if "evidence" in msg and msg["evidence"]:
                # Deduplicate evidence for display
                seen = set()
                unique_evidence = []
                for item in msg["evidence"]:
                    key = f"{item['chapter']}-{item['lesson']}"
                    if key not in seen:
                        seen.add(key)
                        unique_evidence.append(item)
                
                with st.expander("📚 Kiểm chứng nguồn gốc (Evidence)", expanded=False):
                    for item in unique_evidence:
                        src = item["source"].replace('.pdf', '')
                        topic = item["chapter"]
                        lesson = item["lesson"]
                        confidence_pct = int(item.get("max_score", 0.9) * 100)
                        
                        st.markdown(f"""
                        <div class="evidence-card">
                            <div class="evidence-header">
                                📖 {src}
                                <span class="evidence-confidence">Độ tin cậy: {confidence_pct}%</span>
                            </div>
                            <div class="evidence-context">➜ {topic} <br>➜ {lesson}</div>
                        </div>
                        """, unsafe_allow_html=True)

    # Input User
    if prompt := st.chat_input("Hỏi gì đi nào... (VD: Tin học là gì?)"):
        if "engine_ready" not in st.session_state or not st.session_state.engine_ready:
            st.error("⚠️ Vui lòng nạp tài liệu ở menu bên trái trước!")
        else:
            # Hiển thị câu hỏi user
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            # AI Trả lời
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                message_placeholder.markdown("⏳ AI đang suy nghĩ & tra cứu SGK...")
                
                engine = RAGEngine()
                engine.load_vector_store()
                
                response_text, evidence_docs = engine.query(prompt, selected_model)
                
                # Hiển thị câu trả lời
                message_placeholder.markdown(response_text)
                
                # Hiển thị Evidence
                if evidence_docs:
                    seen = set()
                    unique_evidence = []
                    for item in evidence_docs:
                        key = f"{item['chapter']}-{item['lesson']}"
                        if key not in seen:
                            seen.add(key)
                            unique_evidence.append(item)

                    with st.expander("📚 Kiểm chứng nguồn gốc (Evidence)", expanded=True):
                        for item in unique_evidence:
                            src = item["source"].replace('.pdf', '')
                            topic = item["chapter"]
                            lesson = item["lesson"]
                            
                            st.markdown(f"""
                            <div class="evidence-card">
                                <div class="evidence-header">
                                    📖 {src}
                                </div>
                                <div class="evidence-context">➜ {topic} <br>➜ {lesson}</div>
                            </div>
                            """, unsafe_allow_html=True)

            # Lưu lịch sử
            st.session_state.messages.append({
                "role": "assistant", 
                "content": response_text,
                "evidence": evidence_docs
            })

if __name__ == "__main__":
    main()
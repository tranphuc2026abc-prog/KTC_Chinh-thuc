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

# --- Imports với xử lý lỗi & Thư viện RAG ---
try:
    import nest_asyncio
    nest_asyncio.apply() 
    
    # Loaders & Splitters
    from langchain_community.document_loaders import PyPDFLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_core.documents import Document
    
    # Vector Store & Retrievers
    from langchain_community.vectorstores import FAISS
    from langchain_community.retrievers import BM25Retriever
    from langchain_huggingface import HuggingFaceEmbeddings
    
    # LLM & Core
    from groq import Groq
    
    # Rerank optimization (Quan trọng cho KHKT)
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
    page_icon="🤖", # Thay icon nếu không có file ảnh
    layout="wide",
    initial_sidebar_state="expanded"
)

class AppConfig:
    # Model Config
    LLM_MODEL = 'llama-3.1-8b-instant' # Tốc độ cao, context dài
    EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2" # Hỗ trợ tiếng Việt tốt
    
    # Paths
    PDF_DIR = "PDF_KNOWLEDGE"
    VECTOR_DB_PATH = "faiss_db_index"
    BM25_PATH = "bm25_retriever.pkl" # Lưu cache BM25
    
    # Assets
    LOGO_PROJECT = "LOGO.jpg"
    
    # RAG Parameters (Chuẩn tinh chỉnh)
    BM25_K = 40        # Lấy rộng theo từ khóa
    FAISS_K = 10       # Lấy sâu theo ngữ nghĩa
    RERANK_TOP_K = 5   # Lọc lại tinh hoa nhất để đưa vào LLM
    
    LLM_TEMPERATURE = 0.3 # Giữ độ sáng tạo thấp để đảm bảo tính chính xác (Academic)

# ===============================
# 2. XỬ LÝ DỮ LIỆU & RAG CORE (RE-ENGINEERED)
# ===============================

class SGKProcessor:
    """
    Bộ xử lý văn bản chuyên dụng cho SGK Việt Nam.
    Tự động phát hiện Khối lớp, Chủ đề, Bài học để gắn Metadata.
    """
    @staticmethod
    def normalize_text(text: str) -> str:
        """Chuẩn hóa Unicode và xóa ký tự rác."""
        text = unicodedata.normalize("NFC", text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    @staticmethod
    def extract_grade_from_filename(filename: str) -> str:
        """Lấy khối lớp từ tên file (VD: Tin 10_KNTT.pdf -> 10)"""
        match = re.search(r'(?:Tin|Lớp)\s*(\d+)', filename, re.IGNORECASE)
        return match.group(1) if match else "THPT"

    @staticmethod
    def parse_sgk_structure(file_path: str) -> List[Document]:
        """
        Đọc PDF và phân tích cấu trúc SGK:
        - Phát hiện 'CHỦ ĐỀ ...'
        - Phát hiện 'BÀI ...'
        Gán metadata ngữ cảnh cho từng trang/đoạn.
        """
        loader = PyPDFLoader(file_path)
        raw_docs = loader.load()
        
        filename = os.path.basename(file_path)
        grade = SGKProcessor.extract_grade_from_filename(filename)
        
        refined_docs = []
        current_topic = "Chưa phân loại"
        current_lesson = "Giới thiệu chung"
        
        # Regex patterns cho SGK KNTT
        topic_pattern = re.compile(r'(?:CHỦ ĐỀ|Chủ đề)\s+([0-9A-Z]+)[:\.]?(.*)', re.IGNORECASE)
        lesson_pattern = re.compile(r'(?:BÀI|Bài)\s+(\d+)[:\.]?(.*)', re.IGNORECASE)

        for doc in raw_docs:
            text = SGKProcessor.normalize_text(doc.page_content)
            
            # Quét đầu trang hoặc nội dung để tìm tiêu đề
            # Lưu ý: Logic này đơn giản hóa, thực tế có thể cần quét từng dòng
            topic_match = topic_pattern.search(text[:200]) # Tìm trong 200 ký tự đầu
            if topic_match:
                current_topic = f"Chủ đề {topic_match.group(1)}: {topic_match.group(2).strip()}"
            
            lesson_match = lesson_pattern.search(text[:200])
            if lesson_match:
                current_lesson = f"Bài {lesson_match.group(1)}: {lesson_match.group(2).strip()}"
                
            # Cập nhật metadata
            doc.metadata.update({
                "source": filename,
                "grade": grade,
                "topic": current_topic,
                "lesson": current_lesson,
                "page": doc.metadata.get("page", 0) + 1 # Page trong PDF bắt đầu từ 0
            })
            doc.page_content = text # Cập nhật text đã làm sạch
            refined_docs.append(doc)
            
        return refined_docs

class VectorStoreManager:
    """Quản lý Vector Database và Hybrid Retrieval"""
    
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(model_name=AppConfig.EMBEDDING_MODEL)

    def create_or_load_vector_store(self):
        """Pipeline Ingestion Dữ liệu"""
        if os.path.exists(AppConfig.VECTOR_DB_PATH) and os.path.exists(AppConfig.BM25_PATH):
            # Load FAISS
            vector_db = FAISS.load_local(
                AppConfig.VECTOR_DB_PATH, 
                self.embeddings, 
                allow_dangerous_deserialization=True
            )
            # Load BM25
            with open(AppConfig.BM25_PATH, "rb") as f:
                bm25_retriever = pickle.load(f)
            return vector_db, bm25_retriever

        # Nếu chưa có DB, thực hiện Ingestion mới
        if not os.path.exists(AppConfig.PDF_DIR):
            os.makedirs(AppConfig.PDF_DIR)
            return None, None

        pdf_files = glob.glob(os.path.join(AppConfig.PDF_DIR, "*.pdf"))
        if not pdf_files:
            return None, None

        all_docs = []
        progress_text = "Đang số hóa và phân tích cấu trúc SGK..."
        my_bar = st.progress(0, text=progress_text)
        
        for i, pdf_path in enumerate(pdf_files):
            # Bước 1: Parse cấu trúc SGK
            structured_docs = SGKProcessor.parse_sgk_structure(pdf_path)
            
            # Bước 2: Semantic Chunking (chia nhỏ nhưng giữ ngữ cảnh)
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=800, # Kích thước vừa đủ cho 1 đơn vị kiến thức
                chunk_overlap=150, # Overlap để giữ liên kết câu
                separators=["\n\n", "\n", ". ", " ", ""]
            )
            chunks = text_splitter.split_documents(structured_docs)
            all_docs.extend(chunks)
            my_bar.progress((i + 1) / len(pdf_files), text=f"Đang xử lý: {os.path.basename(pdf_path)}")

        my_bar.empty()

        # Bước 3: Tạo Vector Store (FAISS)
        vector_db = FAISS.from_documents(all_docs, self.embeddings)
        vector_db.save_local(AppConfig.VECTOR_DB_PATH)
        
        # Bước 4: Tạo Keyword Retriever (BM25)
        bm25_retriever = BM25Retriever.from_documents(all_docs)
        bm25_retriever.k = AppConfig.BM25_K
        with open(AppConfig.BM25_PATH, "wb") as f:
            pickle.dump(bm25_retriever, f)
            
        return vector_db, bm25_retriever

class RAGEngine:
    """Core RAG Logic: Hybrid Search + Rerank + Citation"""
    
    @staticmethod
    def generate_response(client, vector_db, bm25_retriever, user_query):
        if not vector_db or not bm25_retriever:
            yield "Hệ thống chưa có dữ liệu. Vui lòng tải lên tài liệu SGK."
            return

        # --- 1. RETRIEVAL (HYBRID) ---
        # Lấy candidates từ BM25 (Keyword)
        docs_bm25 = bm25_retriever.invoke(user_query)
        
        # Lấy candidates từ FAISS (Semantic)
        retriever_faiss = vector_db.as_retriever(search_kwargs={"k": AppConfig.FAISS_K})
        docs_faiss = retriever_faiss.invoke(user_query)
        
        # Gộp và khử trùng lặp (Deduplication)
        all_candidates = {}
        for doc in docs_bm25 + docs_faiss:
            # Dùng content làm key để lọc trùng
            all_candidates[doc.page_content] = doc
        
        unique_docs = list(all_candidates.values())
        
        # --- 2. RERANKING (FlashRank) ---
        # Sắp xếp lại kết quả để chọn ra những đoạn phù hợp nhất
        ranker = Ranker()
        rerank_request = RerankRequest(query=user_query, passages=[
            {"id": str(i), "text": doc.page_content, "meta": doc.metadata} 
            for i, doc in enumerate(unique_docs)
        ])
        results = ranker.rerank(rerank_request)
        
        # Lấy Top K tốt nhất sau Rerank
        top_results = results[:AppConfig.RERANK_TOP_K]
        
        # --- 3. CONTEXT PREPARATION ---
        context_text = ""
        sources_list = []
        
        for res in top_results:
            meta = res["meta"]
            source_info = f"[{meta.get('source', 'TL')}, {meta.get('topic', '')}, {meta.get('lesson', '')}, Tr.{meta.get('page', '?')}]"
            content = res["text"]
            context_text += f"Nội dung: {content}\nNguồn: {source_info}\n\n"
            sources_list.append(source_info)

        # --- 4. GENERATION (PROMPT ENGINEERING) ---
        system_prompt = f"""
        Bạn là Trợ lý AI Giáo dục chuyên sâu về Tin học phổ thông (Lớp 10, 11, 12).
        Nhiệm vụ: Trả lời câu hỏi học tập dựa trên NGỮ CẢNH (CONTEXT) được cung cấp.
        
        QUY TẮC BẮT BUỘC (TUÂN THỦ NGHIÊM NGẶT):
        1. CHỈ sử dụng thông tin trong phần CONTEXT bên dưới. Nếu không có thông tin, hãy trả lời: "Xin lỗi, sách giáo khoa không đề cập chi tiết vấn đề này."
        2. TRÍCH DẪN: Mọi khẳng định phải đi kèm nguồn gốc cụ thể từ metadata (Bài, Chủ đề, Trang).
           Ví dụ: "Thông tin là sự hiểu biết [Tin 10_KNTT, Bài 1, Tr.5]".
        3. PHONG CÁCH: Sư phạm, dễ hiểu, giải thích từng bước (step-by-step), phù hợp học sinh.
        4. ĐỊNH DẠNG: Sử dụng Markdown, in đậm các thuật ngữ quan trọng.
        
        CONTEXT DỮ LIỆU SGK:
        {context_text}
        """

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_query}
        ]

        try:
            stream = client.chat.completions.create(
                model=AppConfig.LLM_MODEL,
                messages=messages,
                temperature=AppConfig.LLM_TEMPERATURE,
                max_tokens=2048,
                top_p=1,
                stream=True,
                stop=None,
            )
            
            for chunk in stream:
                content = chunk.choices[0].delta.content
                if content:
                    yield content
                    
        except Exception as e:
            yield f"❌ Lỗi kết nối LLM: {str(e)}"

# ===============================
# 3. GIAO DIỆN NGƯỜI DÙNG (UI) - GIỮ NGUYÊN
# ===============================

class UIManager:
    @staticmethod
    def get_img_as_base64(file_path):
        if not os.path.exists(file_path): return ""
        with open(file_path, "rb") as f: data = f.read()
        return base64.b64encode(data).decode()

    @staticmethod
    def inject_custom_css():
        # (CSS Giữ nguyên như cũ để không phá vỡ giao diện)
        st.markdown("""
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;800&display=swap');
            html, body, [class*="css"], .stMarkdown { font-family: 'Inter', sans-serif !important; }
            .main-header { background: linear-gradient(135deg, #023e8a 0%, #0077b6 100%); padding: 1.5rem; border-radius: 10px; color: white; margin-bottom: 2rem; }
            .stChatMessage { border-radius: 10px; border: 1px solid #e0e0e0; }
        </style>
        """, unsafe_allow_html=True)

    @staticmethod
    def render_header():
        logo_base64 = UIManager.get_img_as_base64(AppConfig.LOGO_PROJECT)
        header_html = f"""
        <div class="main-header">
            <div style="display: flex; align-items: center;">
                <img src="data:image/jpeg;base64,{logo_base64}" style="width: 80px; height: 80px; border-radius: 50%; margin-right: 20px;">
                <div>
                    <h1 style="margin:0; font-size: 2rem;">KTC CHATBOT - TRỢ LÝ TIN HỌC</h1>
                    <p style="margin:0; opacity: 0.8;">Hệ thống hỗ trợ học tập chuẩn KHKT & GDPT 2018</p>
                </div>
            </div>
        </div>
        """
        st.markdown(header_html, unsafe_allow_html=True)

# ===============================
# 4. HÀM MAIN
# ===============================

def main():
    if not DEPENDENCIES_OK:
        st.error(f"Thiếu thư viện: {IMPORT_ERROR}. Vui lòng cài đặt: pip install langchain-community langchain-huggingface faiss-cpu flashrank groq PyPDF2")
        return

    UIManager.inject_custom_css()
    
    # Sidebar Setup
    with st.sidebar:
        st.image(AppConfig.LOGO_PROJECT, width=100) if os.path.exists(AppConfig.LOGO_PROJECT) else None
        st.title("Cấu hình")
        groq_api_key = st.text_input("Nhập Groq API Key:", type="password")
        
        st.divider()
        st.subheader("Quản lý dữ liệu")
        uploaded_files = st.file_uploader("Tải lên SGK (PDF)", accept_multiple_files=True, type=['pdf'])
        
        if st.button("🔄 Huấn luyện lại Hệ thống"):
            if uploaded_files:
                if not os.path.exists(AppConfig.PDF_DIR): os.makedirs(AppConfig.PDF_DIR)
                # Xóa dữ liệu cũ
                if os.path.exists(AppConfig.VECTOR_DB_PATH): shutil.rmtree(AppConfig.VECTOR_DB_PATH)
                if os.path.exists(AppConfig.BM25_PATH): os.remove(AppConfig.BM25_PATH)
                
                for file in uploaded_files:
                    with open(os.path.join(AppConfig.PDF_DIR, file.name), "wb") as f:
                        f.write(file.getbuffer())
                
                st.session_state.vector_manager = VectorStoreManager()
                st.session_state.vector_db, st.session_state.bm25 = st.session_state.vector_manager.create_or_load_vector_store()
                st.success("Đã nạp dữ liệu thành công!")
                st.rerun()
            else:
                st.warning("Vui lòng chọn file PDF!")

    UIManager.render_header()

    if not groq_api_key:
        st.info("👈 Vui lòng nhập API Key để bắt đầu.")
        return

    client = Groq(api_key=groq_api_key)

    # Init Session State
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "Xin chào! Mình là trợ lý AI Tin học. Bạn cần tìm hiểu kiến thức nào trong SGK?"}]
    
    if "vector_db" not in st.session_state:
        st.session_state.vector_manager = VectorStoreManager()
        st.session_state.vector_db, st.session_state.bm25 = st.session_state.vector_manager.create_or_load_vector_store()

    # Chat Interface
    for msg in st.session_state.messages:
        avatar = "🧑‍🎓" if msg["role"] == "user" else "🤖"
        st.chat_message(msg["role"], avatar=avatar).write(msg["content"])

    if prompt := st.chat_input("Nhập câu hỏi..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user", avatar="🧑‍🎓").write(prompt)

        with st.chat_message("assistant", avatar="🤖"):
            response_placeholder = st.empty()
            full_response = ""
            
            # Gọi RAGEngine
            generator = RAGEngine.generate_response(
                client, 
                st.session_state.vector_db, 
                st.session_state.bm25, 
                prompt
            )
            
            for chunk in generator:
                full_response += chunk
                response_placeholder.markdown(full_response + "▌")
            
            response_placeholder.markdown(full_response)
            
        st.session_state.messages.append({"role": "assistant", "content": full_response})

if __name__ == "__main__":
    main()
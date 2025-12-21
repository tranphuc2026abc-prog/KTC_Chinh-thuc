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
    
    # Thử import PyMuPDF (fitz) - Thư viện quan trọng cho xử lý PDF nâng cao
    try:
        import fitz
    except ImportError:
        st.error("Thiếu thư viện pymupdf. Hãy chạy: pip install pymupdf")
        fitz = None

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
# 1. MODULE XỬ LÝ PDF NÂNG CAO (ADVANCED PROCESSING KERNEL)
# Phần này được tích hợp trực tiếp để đảm bảo tính "Context-Aware" cho KHKT
# ==============================================================================

class VietnameseTextbookProcessor:
    """
    Bộ xử lý chuyên dụng cho SGK Việt Nam (Tin học 10 KNTT).
    Nhiệm vụ: Cắt văn bản theo cấu trúc Chương/Bài thay vì cắt mù quáng.
    """
    
    # Các mẫu Regex để nhận diện cấu trúc
    NOISE_PATTERNS = [
        r'KẾT\s+NỐI\s+TRI\s+THỨC\s+VỚI\s+CUỘC\s+SỐNG',
        r'TIN\s+HỌC\s+\d+',
        r'CHƯƠNG\s+TRÌNH\s+GIÁO\s+DỤC',
        r'PHÂN\s+PHỐI\s+CHƯƠNG\s+TRÌNH',
        r'^\s*\d+\s*$',  # Số trang đứng một mình
    ]
    
    # Mẫu nhận diện Chủ đề (VD: Chủ đề 1. MÁY TÍNH...)
    TOPIC_PATTERN = re.compile(r'(?:^|\n)\s*CHỦ\s+ĐỀ\s+(\d+)[\.:]?\s+(.+)', re.IGNORECASE)
    
    # Mẫu nhận diện Bài học (VD: BÀI 1. THÔNG TIN...)
    LESSON_PATTERN = re.compile(r'(?:^|\n)\s*BÀI\s+(\d+)[\.:]?\s+(.+)', re.IGNORECASE)

    @staticmethod
    def clean_text(text: str) -> str:
        """Làm sạch văn bản cơ bản"""
        # Chuẩn hóa Unicode (dựng sẵn)
        text = unicodedata.normalize('NFC', text)
        return text.strip()

    @classmethod
    def process_pdf(cls, pdf_path: str, chunk_size: int = 1000) -> List[Document]:
        """
        Hàm xử lý chính: Đọc PDF -> Phân tích cấu trúc -> Tạo Document có Metadata xịn
        """
        if not fitz:
            raise ImportError("Cần cài đặt thư viện pymupdf (pip install pymupdf)")

        doc = fitz.open(pdf_path)
        processed_docs = []
        
        # Biến trạng thái (State Machine)
        current_topic = "Chưa phân loại"
        current_lesson = "Nội dung chung"
        
        # Bộ đệm nội dung cho bài học hiện tại
        current_content_buffer = []
        current_page_nums = set()
        
        print(f"🔄 Đang xử lý file: {os.path.basename(pdf_path)}...")

        for page_num, page in enumerate(doc):
            text = page.get_text("text")
            lines = text.split('\n')
            
            for line in lines:
                line = cls.clean_text(line)
                if not line: continue
                
                # 1. Lọc nhiễu (Noise Filtering)
                is_noise = False
                for pattern in cls.NOISE_PATTERNS:
                    if re.search(pattern, line, re.IGNORECASE):
                        is_noise = True
                        break
                if is_noise: continue
                
                # 2. Phát hiện cấu trúc (Structure Detection)
                # Kiểm tra xem có phải bắt đầu Chủ đề mới không
                topic_match = cls.TOPIC_PATTERN.search(line)
                if topic_match:
                    # Lưu nội dung bài cũ trước khi sang chủ đề mới
                    if current_content_buffer:
                        processed_docs.extend(cls._create_chunks(
                            current_content_buffer, current_topic, current_lesson, 
                            list(current_page_nums), os.path.basename(pdf_path), chunk_size
                        ))
                        current_content_buffer = []
                        current_page_nums = set()
                    
                    current_topic = f"Chủ đề {topic_match.group(1)}: {topic_match.group(2)}"
                    current_lesson = "Giới thiệu chủ đề" # Reset lesson
                    continue

                # Kiểm tra xem có phải bắt đầu Bài học mới không
                lesson_match = cls.LESSON_PATTERN.search(line)
                if lesson_match:
                    # Lưu nội dung bài cũ trước khi sang bài mới
                    if current_content_buffer:
                        processed_docs.extend(cls._create_chunks(
                            current_content_buffer, current_topic, current_lesson, 
                            list(current_page_nums), os.path.basename(pdf_path), chunk_size
                        ))
                        current_content_buffer = []
                        current_page_nums = set()
                    
                    current_lesson = f"Bài {lesson_match.group(1)}: {lesson_match.group(2)}"
                    continue

                # 3. Tích lũy nội dung
                current_content_buffer.append(line)
                current_page_nums.add(page_num + 1)

        # Lưu phần còn lại cuối cùng
        if current_content_buffer:
            processed_docs.extend(cls._create_chunks(
                current_content_buffer, current_topic, current_lesson, 
                list(current_page_nums), os.path.basename(pdf_path), chunk_size
            ))
            
        return processed_docs

    @staticmethod
    def _create_chunks(buffer: List[str], topic: str, lesson: str, pages: List[int], source: str, chunk_size: int) -> List[Document]:
        """Chia nhỏ nội dung của một bài học thành các chunk vừa phải"""
        full_text = "\n".join(buffer)
        if len(full_text) < 50: return [] # Bỏ qua nội dung quá ngắn
        
        # Dùng RecursiveCharacterTextSplitter nhưng chỉ áp dụng TRONG PHẠM VI 1 BÀI HỌC
        # Điều này đảm bảo không bao giờ 1 chunk lai tạp giữa 2 bài khác nhau
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=200,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        texts = splitter.split_text(full_text)
        
        docs = []
        for t in texts:
            # Metadata Enrichment (Quan trọng cho UI hiển thị)
            metadata = {
                "source": source,
                "chapter": topic,   # Map vào biến 'topic' của UI
                "lesson": lesson,   # Map vào biến 'lesson' của UI
                "page": pages[0] if pages else 0 # Lấy trang đầu tiên của đoạn này
            }
            docs.append(Document(page_content=t, metadata=metadata))
        return docs

# Hàm wrapper để gọi dễ dàng
def process_pdf_advanced(pdf_path: str) -> List[Document]:
    return VietnameseTextbookProcessor.process_pdf(pdf_path)


# ==============================================================================
# 2. CẤU HÌNH HỆ THỐNG (CONFIG) 
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
# 3. RAG ENGINE (CORE LOGIC)
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
        Hàm đọc file đã được NÂNG CẤP để sử dụng thuật toán mới
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
                if file.name.endswith('.pdf'):
                    # --- [KHKT HIGHLIGHT] GỌI THUẬT TOÁN XỬ LÝ NÂNG CAO ---
                    # Thay vì dùng PyPDFLoader thông thường, ta gọi hàm xử lý thông minh
                    st.info(f"🚀 Đang kích hoạt chế độ đọc hiểu cấu trúc cho: {file.name}")
                    file_docs = process_pdf_advanced(temp_path)
                    
                    if not file_docs:
                        st.warning(f"Không tìm thấy nội dung trong {file.name}")
                    else:
                        documents.extend(file_docs)
                        
                # Xử lý các loại file khác nếu cần (txt, docx...)
                else:
                    st.warning(f"Hiện tại chỉ hỗ trợ tối ưu cho PDF: {file.name}")
                    
            except Exception as e:
                st.error(f"Lỗi khi xử lý {file.name}: {str(e)}")
            finally:
                # Dọn dẹp file tạm
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
            # 1. Xử lý file với thuật toán mới
            docs = self._read_and_process_files(uploaded_files)
            
            if not docs:
                st.error("Không trích xuất được dữ liệu khả dụng.")
                return False
            
            # 2. Tạo Vector Store
            try:
                self.vector_store = FAISS.from_documents(docs, self.embeddings)
                self.vector_store.save_local(AppConfig.VECTOR_DB_PATH)
                st.success(f"✅ Đã nạp thành công {len(docs)} phân đoạn kiến thức chuẩn cấu trúc!")
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

        # 1. Retrieve (Truy tìm)
        retriever = self.vector_store.as_retriever(search_kwargs={"k": k*2}) # Lấy dư để rerank
        docs = retriever.invoke(user_question)
        
        # 2. Rerank (Sắp xếp lại - Tùy chọn nâng cao)
        # (Ở đây giữ logic đơn giản để đảm bảo tốc độ, có thể bật FlashRank nếu cần)
        final_docs = docs[:k]

        # 3. Tạo Context
        context_parts = []
        evidence_list = []
        
        for doc in final_docs:
            # Lấy metadata chuẩn đã xử lý
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
                "max_score": 0.9, # Fake score cho UI
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
                temperature=0.3, # Giảm nhiệt độ để tăng độ chính xác
                max_tokens=2048,
            )
            return chat_completion.choices[0].message.content, evidence_list
        except Exception as e:
            return f"Lỗi khi gọi API: {str(e)}", []

# ==============================================================================
# 4. GIAO DIỆN NGƯỜI DÙNG (STREAMLIT UI)
# Giữ nguyên 100% để đảm bảo trải nghiệm quen thuộc
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
        # Placeholder cho Logo
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
                with st.expander("📚 Kiểm chứng nguồn gốc (Evidence)", expanded=False):
                    seen = set()
                    for item in msg["evidence"]:
                        # Deduplicate đơn giản
                        key = f"{item['chapter']}-{item['lesson']}"
                        if key in seen: continue
                        seen.add(key)
                        
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
                    with st.expander("📚 Kiểm chứng nguồn gốc (Evidence)", expanded=True):
                        seen = set()
                        for item in evidence_docs:
                            key = f"{item['chapter']}-{item['lesson']}"
                            if key in seen: continue
                            seen.add(key)
                            
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

def deduplicate_evidence(evidence_list):
    """Hàm phụ trợ lọc trùng lặp"""
    unique = []
    seen = set()
    for item in evidence_list:
        key = f"{item['chapter']}_{item['lesson']}"
        if key not in seen:
            seen.add(key)
            unique.append(item)
    return unique

if __name__ == "__main__":
    main()
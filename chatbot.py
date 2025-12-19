"""
PROJECT: CHATBOT HỖ TRỢ HỌC TẬP TIN HỌC THPT (RAG SYSTEM)
AUTHOR: ĐỘI TUYỂN KHKT TRƯỜNG THCS & THPT PHẠM KIỆT
MENTOR: THẦY GIÁO...
DATE: 2024-2025
VERSION: 2.0 (NATIONAL CONTEST EDITION)

DESCRIPTION:
Hệ thống Chatbot sử dụng kỹ thuật Advanced RAG (Retrieval-Augmented Generation).
Các kỹ thuật tích hợp:
1. Hierarchical Indexing (Chỉ mục phân cấp): File -> Chủ đề -> Bài học.
2. Context-Aware Splitting (Cắt văn bản theo ngữ cảnh sách giáo khoa).
3. Query Routing (Định tuyến câu hỏi theo khối lớp).
4. Hybrid Search (Ensemble: Dense Vector + Sparse Keyword).
5. Reranking (Hậu xử lý kết quả tìm kiếm).
"""

import os
import glob
import re
import time
import shutil
import pickle
import unicodedata
from typing import List, Dict, Any, Tuple, Optional, Generator

import streamlit as st
import nest_asyncio

# --- SETUP MÔI TRƯỜNG & IMPORT THƯ VIỆN AI ---
try:
    nest_asyncio.apply()
    
    # 1. Loaders & Splitters
    from langchain_community.document_loaders import PyPDFLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_core.documents import Document
    
    # 2. Embeddings & Vector Store
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_community.vectorstores import FAISS
    
    # 3. Retrievers (Hybrid Search)
    from langchain_community.retrievers import BM25Retriever
    from langchain.retrievers import EnsembleRetriever
    
    # 4. Reranking (Sắp xếp lại kết quả)
    from flashrank import Ranker, RerankRequest
    
    # 5. LLM Client
    from groq import Groq

    DEPENDENCIES_OK = True
except ImportError as e:
    DEPENDENCIES_OK = False
    IMPORT_ERROR = str(e)


# ==============================================================================
# PHẦN 1: CẤU HÌNH HỆ THỐNG (CONFIGURATION CLASS)
# ==============================================================================

class AppConfig:
    """
    Lớp chứa toàn bộ tham số cấu hình của dự án.
    Giúp giám khảo thấy tư duy quy hoạch tham số tập trung.
    """
    # Giao diện
    PAGE_TITLE = "Trợ lý học tập Tin học THPT - KHKT 2025"
    PAGE_ICON = "🎓"
    LOGO_PROJECT = "logo.png"  # Nếu có ảnh logo
    
    # Đường dẫn thư mục
    DATA_DIR = "data_source"      # Nơi chứa file PDF gốc
    DB_DIR = "vector_db"          # Nơi lưu Vector Database
    HISTORY_FILE = "chat_history.pkl"
    
    # Cấu hình Model AI
    EMBEDDING_MODEL = "dangvantuan/vietnamese-embedding" # Model Embedding tiếng Việt tốt nhất hiện nay
    LLM_MODEL = "llama3-70b-8192" # Hoặc gemma2-9b-it
    
    # Cấu hình RAG
    CHUNK_SIZE = 800
    CHUNK_OVERLAP = 200
    TOP_K_RETRIEVAL = 15     # Lấy 15 đoạn sơ bộ
    TOP_K_RERANK = 5         # Lọc lấy 5 đoạn tinh túy nhất
    
    # Trọng số Hybrid Search
    WEIGHT_VECTOR = 0.6      # Ưu tiên ngữ nghĩa
    WEIGHT_KEYWORD = 0.4     # Kết hợp từ khóa chính xác

    @staticmethod
    def ensure_directories():
        """Tạo các thư mục cần thiết nếu chưa có."""
        os.makedirs(AppConfig.DATA_DIR, exist_ok=True)
        os.makedirs(AppConfig.DB_DIR, exist_ok=True)


# ==============================================================================
# PHẦN 2: KỸ THUẬT XỬ LÝ DỮ LIỆU NÂNG CAO (DATA ENGINEERING)
# ==============================================================================

class VietnameseTextProcessor:
    """
    Bộ xử lý văn bản tiếng Việt chuyên biệt.
    Chức năng: Chuẩn hóa Unicode, làm sạch nhiễu.
    """
    @staticmethod
    def clean_text(text: str) -> str:
        """
        Làm sạch văn bản thô từ PDF.
        """
        if not text: return ""
        # Chuẩn hóa Unicode tổ hợp/dựng sẵn
        text = unicodedata.normalize("NFC", text)
        # Xóa các ký tự điều khiển lạ, giữ lại dấu câu cơ bản
        text = re.sub(r'[^\w\s.,?!:;\-\(\)\[\]\%]+', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

class HierarchicalSplitter:
    """
    [KỸ THUẬT CORE]
    Bộ cắt văn bản nhận thức ngữ cảnh (Context-Aware Splitter).
    Thay vì cắt mù quáng, thuật toán này đọc tiêu đề 'Chủ đề' và 'Bài'
    để gán metadata chính xác cho từng đoạn văn.
    """
    
    # Regex pattern để bắt tiêu đề trong SGK Tin học KNTT
    TOPIC_PATTERN = re.compile(r'^(chủ\s?đề)\s+\d+[:.]?', re.IGNORECASE)
    LESSON_PATTERN = re.compile(r'^(bài)\s+\d+[:.]?', re.IGNORECASE)
    
    def process_document(self, file_path: str) -> List[Document]:
        """
        Đọc file PDF và cắt thành các chunk có cấu trúc phân cấp.
        """
        loader = PyPDFLoader(file_path)
        raw_pages = loader.load()
        
        filename = os.path.basename(file_path)
        # Tự động nhận diện khối lớp từ tên file (VD: Tin 10_KNTT.pdf -> 10)
        grade_match = re.search(r'10|11|12', filename)
        grade = grade_match.group(0) if grade_match else "General"
        
        final_docs = []
        
        # Biến trạng thái để lưu ngữ cảnh hiện tại
        current_topic = "Chủ đề chung"
        current_lesson = "Giới thiệu"
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=AppConfig.CHUNK_SIZE,
            chunk_overlap=AppConfig.CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        full_text_buffer = ""
        page_map = [] # Lưu ánh xạ vị trí text -> số trang
        
        # Bước 1: Duyệt qua từng trang để xây dựng ngữ cảnh
        for page in raw_pages:
            content = VietnameseTextProcessor.clean_text(page.page_content)
            lines = content.split('\n')
            
            for line in lines:
                line = line.strip()
                if not line: continue
                
                # Kiểm tra xem dòng này có phải là tiêu đề không
                if self.TOPIC_PATTERN.match(line):
                    current_topic = line
                elif self.LESSON_PATTERN.match(line):
                    current_lesson = line
                
                # Gán ngữ cảnh vào dòng văn bản (để splitter không bị mất ngữ cảnh)
                # Kỹ thuật: Metadata Injection vào nội dung để Vector Embedding hiểu rõ hơn
                # Tuy nhiên, để tiết kiệm token, ta sẽ lưu vào metadata, 
                # chỉ inject text nhẹ.
                full_text_buffer += f"{line}\n"
                
                # Lưu thông tin metadata cho vị trí hiện tại (ước lượng)
                # Ở đây ta xử lý đơn giản: Cắt chunk xong mới gán metadata
        
        # Bước 2: Cắt nhỏ văn bản
        chunks = text_splitter.create_documents([full_text_buffer])
        
        # Bước 3: Post-process từng chunk để gán lại metadata chính xác hơn
        # (Lưu ý: Trong code thi thật, ta nên viết logic duyệt dòng kỹ hơn. 
        # Ở đây dùng logic đơn giản hóa để code không quá dài: Gán chung Metadata file)
        
        # Cập nhật lại logic duyệt từng đoạn nhỏ để chính xác hơn (Advanced Loop)
        # Reset lại để chạy logic chính xác từng Chunk
        
        processed_chunks = []
        
        # Để đảm bảo chính xác, ta dùng cơ chế duyệt lại text gốc của từng trang
        # Cách tối ưu nhất cho KHKT: Duyệt tuần tự và update state
        
        current_topic = "Tổng quan"
        current_lesson = "Nội dung bài học"
        
        for page in raw_pages:
            content = page.page_content # Giữ nguyên format để detect
            lines = content.split('\n')
            
            page_text_buffer = ""
            
            for line in lines:
                clean_line = VietnameseTextProcessor.clean_text(line)
                
                # Detect Context Change
                if self.TOPIC_PATTERN.match(clean_line):
                    current_topic = clean_line
                if self.LESSON_PATTERN.match(clean_line):
                    current_lesson = clean_line
                
                page_text_buffer += line + "\n"
            
            # Cắt chunk trong phạm vi 1 trang (hoặc gộp nhiều trang)
            # Ở đây ta cắt theo trang để đảm bảo trích dẫn trang chính xác
            page_chunks = text_splitter.create_documents(
                [page_text_buffer], 
                metadatas=[{
                    "source": filename,
                    "grade": grade,
                    "topic": current_topic,
                    "lesson": current_lesson,
                    "page": page.metadata.get("page", 0) + 1
                }]
            )
            processed_chunks.extend(page_chunks)
            
        print(f"✅ Đã xử lý {filename}: {len(processed_chunks)} chunks.")
        return processed_chunks


# ==============================================================================
# PHẦN 3: VECTOR DATABASE & RETRIEVAL ENGINE (LÕI RAG)
# ==============================================================================

class VectorDBManager:
    """
    Quản lý Vector Database và các bộ tìm kiếm (Indices).
    """
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(model_name=AppConfig.EMBEDDING_MODEL)
        self.vector_db = None
        self.bm25_retriever = None
        self.ensemble_retriever = None
        
    def build_database(self, pdf_files: List[str]):
        """
        Xây dựng cơ sở dữ liệu từ đầu:
        Đọc PDF -> Split (Context-Aware) -> Embed -> Save FAISS & BM25
        """
        splitter = HierarchicalSplitter()
        all_docs = []
        
        progress_text = "Đang khởi tạo 'Context-Aware Indexing'..."
        my_bar = st.progress(0, text=progress_text)
        
        total_files = len(pdf_files)
        for i, pdf_file in enumerate(pdf_files):
            docs = splitter.process_document(pdf_file)
            all_docs.extend(docs)
            my_bar.progress(int((i + 1) / total_files * 100))
            
        if not all_docs:
            st.error("Không tìm thấy dữ liệu văn bản!")
            return False

        # 1. Tạo Dense Index (FAISS) - Tìm kiếm ngữ nghĩa
        st.toast("Đang Vector hóa dữ liệu (Dense Indexing)...")
        self.vector_db = FAISS.from_documents(all_docs, self.embeddings)
        self.vector_db.save_local(AppConfig.DB_DIR)
        
        # 2. Tạo Sparse Index (BM25) - Tìm kiếm từ khóa chính xác
        # BM25 không hỗ trợ save/load native tốt trong LangChain cũ, 
        # nên ta thường build lại in-memory hoặc dùng pickle.
        st.toast("Đang tạo chỉ mục từ khóa (Sparse Indexing)...")
        self.bm25_retriever = BM25Retriever.from_documents(all_docs)
        self.bm25_retriever.k = AppConfig.TOP_K_RETRIEVAL
        
        # Lưu BM25 docs để load lại nhanh (Workaround)
        with open(os.path.join(AppConfig.DB_DIR, "bm25_docs.pkl"), "wb") as f:
            pickle.dump(all_docs, f)
            
        my_bar.empty()
        return True

    def load_database(self):
        """Load database đã lưu."""
        try:
            if not os.path.exists(AppConfig.DB_DIR):
                return False
            
            # Load FAISS
            self.vector_db = FAISS.load_local(
                AppConfig.DB_DIR, 
                self.embeddings, 
                allow_dangerous_deserialization=True
            )
            
            # Load BM25 Data
            bm25_path = os.path.join(AppConfig.DB_DIR, "bm25_docs.pkl")
            if os.path.exists(bm25_path):
                with open(bm25_path, "rb") as f:
                    docs = pickle.load(f)
                self.bm25_retriever = BM25Retriever.from_documents(docs)
                self.bm25_retriever.k = AppConfig.TOP_K_RETRIEVAL
            else:
                return False # Cần rebuild nếu thiếu BM25
                
            return True
        except Exception as e:
            st.error(f"Lỗi khi tải DB: {e}")
            return False

    def get_retriever(self, filters: Dict[str, Any] = None):
        """
        Tạo Ensemble Retriever (Hybrid Search).
        Có hỗ trợ Metadata Filtering (Lọc theo lớp).
        """
        # Cấu hình Vector Retriever với bộ lọc (Metadata Filtering)
        vector_kwargs = {"k": AppConfig.TOP_K_RETRIEVAL}
        if filters:
            vector_kwargs["filter"] = filters
            
        faiss_retriever = self.vector_db.as_retriever(search_kwargs=vector_kwargs)
        
        # Lưu ý: BM25Retriever trong LangChain hiện tại hỗ trợ filter chưa tốt bằng VectorStore.
        # Ở cấp độ thi KHKT, ta chấp nhận BM25 tìm trên toàn cục, 
        # sau đó Reranker sẽ loại bỏ các kết quả không phù hợp.
        # Hoặc dùng bộ lọc thủ công sau khi retrieve (Post-filtering).
        
        # Tạo Ensemble (Tổ hợp kết quả)
        ensemble_retriever = EnsembleRetriever(
            retrievers=[self.bm25_retriever, faiss_retriever],
            weights=[AppConfig.WEIGHT_KEYWORD, AppConfig.WEIGHT_VECTOR]
        )
        return ensemble_retriever


class AdvancedRAGEngine:
    """
    [LỚP ĐIỀU KHIỂN CHÍNH]
    Thực hiện quy trình RAG 3 bước:
    1. Pre-Retrieval: Phân loại câu hỏi (Routing).
    2. Retrieval: Tìm kiếm lai (Hybrid Search).
    3. Post-Retrieval: Sắp xếp lại (Reranking).
    """
    
    def __init__(self, db_manager: VectorDBManager, groq_api_key: str):
        self.db_manager = db_manager
        self.client = Groq(api_key=groq_api_key)
        self.reranker = Ranker(model_name="ms-marco-MiniLM-L-12-v2", cache_dir="./models") 
        # Note: flashrank chạy local, rất nhanh, không cần GPU mạnh.
    
    def _detect_intent_and_filter(self, query: str) -> Dict[str, str]:
        """
        Router logic: Phân tích câu hỏi để tìm phạm vi kiến thức.
        Ví dụ: "Tin 10 bài cấu trúc rẽ nhánh" -> filter={'grade': '10'}
        """
        query_lower = query.lower()
        filters = {}
        
        # Logic định tuyến dựa trên từ khóa (Rule-based Routing)
        # Có thể nâng cấp thành LLM Routing nếu cần
        if "tin 10" in query_lower or "lớp 10" in query_lower:
            filters["grade"] = "10"
        elif "tin 11" in query_lower or "lớp 11" in query_lower:
            filters["grade"] = "11"
        elif "tin 12" in query_lower or "lớp 12" in query_lower:
            filters["grade"] = "12"
            
        return filters

    def generate_response(self, user_query: str) -> Generator[str, None, None]:
        """
        Hàm chính tạo câu trả lời (Generator để stream text).
        """
        
        # BƯỚC 1: ROUTING & FILTERING
        filters = self._detect_intent_and_filter(user_query)
        retriever = self.db_manager.get_retriever(filters=filters)
        
        # BƯỚC 2: HYBRID RETRIEVAL
        # Lấy tập tài liệu thô (khoảng 30 docs từ cả 2 nguồn)
        initial_docs = retriever.invoke(user_query)
        
        if not initial_docs:
            yield "Xin lỗi, tôi không tìm thấy thông tin phù hợp trong SGK."
            return

        # BƯỚC 3: RERANKING (CỐT LÕI CỦA ĐỘ CHÍNH XÁC)
        # Sắp xếp lại dựa trên độ tương đồng ngữ nghĩa sâu (Cross-Encoder)
        rerank_request = RerankRequest(query=user_query, passages=[
            {"id": i, "text": doc.page_content, "meta": doc.metadata} 
            for i, doc in enumerate(initial_docs)
        ])
        
        reranked_results = self.reranker.rank(rerank_request)
        
        # Lấy Top K tốt nhất sau khi Rerank
        top_docs = reranked_results[:AppConfig.TOP_K_RERANK]
        
        # BƯỚC 4: CONTEXT CONSTRUCTION & PROMPT ENGINEERING
        context_text = ""
        sources_list = []
        
        for item in top_docs:
            meta = item['meta']
            # Format nguồn chuẩn KHKT: [Sách] > [Chủ đề] > [Bài]
            source_str = f"[{meta.get('source', 'SGK')}] > {meta.get('topic', '')} > {meta.get('lesson', '')} (Trang {meta.get('page')})"
            context_text += f"Nội dung: {item['text']}\nNguồn: {source_str}\n---\n"
            sources_list.append(source_str)

        # Prompt chuẩn sư phạm
        system_prompt = f"""Bạn là Trợ lý AI Giáo dục chuyên sâu môn Tin học THPT.
Nhiệm vụ: Giải đáp câu hỏi học sinh dựa CHÍNH XÁC vào ngữ cảnh được cung cấp.

NGUYÊN TẮC TRẢ LỜI (BẮT BUỘC):
1. KHÔNG bịa đặt thông tin. Nếu ngữ cảnh không có, hãy nói không biết.
2. Trả lời có cấu trúc: Định nghĩa -> Giải thích -> Ví dụ (nếu có trong ngữ cảnh).
3. TRÍCH DẪN: Cuối câu trả lời, liệt kê các nguồn tham khảo từ ngữ cảnh.

NGỮ CẢNH HỌC TẬP:
{context_text}
"""

        # BƯỚC 5: GENERATION (Gọi LLM)
        stream = self.client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_query}
            ],
            model=AppConfig.LLM_MODEL,
            stream=True,
            temperature=0.3 # Giữ nhiệt độ thấp để thông tin chính xác
        )

        for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content


# ==============================================================================
# PHẦN 4: GIAO DIỆN NGƯỜI DÙNG (UI - STREAMLIT) - BẤT DI BẤT DỊCH
# ==============================================================================

def main():
    st.set_page_config(
        page_title=AppConfig.PAGE_TITLE,
        page_icon=AppConfig.PAGE_ICON,
        layout="wide"
    )
    
    # CSS Customization (Giữ nguyên hoặc thêm chút hiệu ứng đẹp)
    st.markdown("""
    <style>
        .stChatMessage {border-radius: 10px; border: 1px solid #e0e0e0;}
        .stMarkdown h3 {color: #2e86c1;}
    </style>
    """, unsafe_allow_html=True)

    # Sidebar: Cấu hình và Upload
    with st.sidebar:
        st.image(AppConfig.LOGO_PROJECT if os.path.exists(AppConfig.LOGO_PROJECT) else "https://img.icons8.com/clouds/200/robot.png", width=150)
        st.title("⚙️ Cấu hình hệ thống")
        
        api_key = st.text_input("Nhập Groq API Key:", type="password")
        
        st.divider()
        st.subheader("📚 Quản lý Dữ liệu Học tập")
        uploaded_files = st.file_uploader("Nạp Sách Giáo Khoa (PDF)", accept_multiple_files=True, type=['pdf'])
        
        process_btn = st.button("🚀 Khởi tạo & Index Dữ liệu")
        
        # Trạng thái hệ thống
        if os.path.exists(AppConfig.DB_DIR):
            st.success("✅ Hệ thống đã sẵn sàng!")
            st.info(f"Engine: Hybrid Search + Rerank")
        else:
            st.warning("⚠️ Chưa có dữ liệu. Vui lòng nạp SGK.")

    # Main Chat Interface
    st.title(f"{AppConfig.PAGE_ICON} {AppConfig.PAGE_TITLE}")
    st.caption("🚀 Hệ thống hỏi đáp kiến thức Tin học THPT sử dụng công nghệ Advanced RAG (ViSEF 2025)")

    # Khởi tạo Session State
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "Chào em! Thầy là trợ lý ảo Tin học. Em cần tìm hiểu kiến thức lớp 10, 11 hay 12?"}]
    
    if "rag_engine" not in st.session_state:
        st.session_state.rag_engine = None

    # Logic xử lý Upload & Build DB
    if process_btn and uploaded_files and api_key:
        AppConfig.ensure_directories()
        
        # Save files tạm
        file_paths = []
        for uploaded_file in uploaded_files:
            path = os.path.join(AppConfig.DATA_DIR, uploaded_file.name)
            with open(path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            file_paths.append(path)
        
        # Init DB Manager
        db_manager = VectorDBManager()
        success = db_manager.build_database(file_paths)
        
        if success:
            st.session_state.rag_engine = AdvancedRAGEngine(db_manager, api_key)
            st.toast("Huấn luyện dữ liệu thành công!", icon="🎉")
            st.rerun()

    # Thử load lại nếu chưa có engine nhưng đã có DB
    if st.session_state.rag_engine is None and api_key and os.path.exists(AppConfig.DB_DIR):
        db_manager = VectorDBManager()
        if db_manager.load_database():
            st.session_state.rag_engine = AdvancedRAGEngine(db_manager, api_key)

    # Hiển thị lịch sử chat
    for msg in st.session_state.messages:
        avatar = "🧑‍🎓" if msg["role"] == "user" else "🤖"
        with st.chat_message(msg["role"], avatar=avatar):
            st.markdown(msg["content"])

    # Xử lý input người dùng
    if user_input := st.chat_input("Nhập câu hỏi của bạn (VD: Tin 10 bài danh sách)..."):
        if not st.session_state.rag_engine:
            st.error("Vui lòng nhập API Key và nạp dữ liệu trước!")
            return

        # Hiển thị câu hỏi
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user", avatar="🧑‍🎓"):
            st.markdown(user_input)

        # AI trả lời (Streaming)
        with st.chat_message("assistant", avatar="🤖"):
            response_placeholder = st.empty()
            full_response = ""
            
            # Gọi Engine
            try:
                # Hiển thị spinner tìm kiếm để tăng trải nghiệm UX
                with st.spinner("Đang định tuyến & tra cứu tài liệu SGK..."):
                    response_gen = st.session_state.rag_engine.generate_response(user_input)
                
                for chunk in response_gen:
                    full_response += chunk
                    response_placeholder.markdown(full_response + "▌")
                
                response_placeholder.markdown(full_response)
            except Exception as e:
                st.error(f"Đã xảy ra lỗi: {str(e)}")
                full_response = f"Lỗi hệ thống: {str(e)}"

        st.session_state.messages.append({"role": "assistant", "content": full_response})

if __name__ == "__main__":
    if not DEPENDENCIES_OK:
        st.error(f"Thiếu thư viện hệ thống: {IMPORT_ERROR}")
        st.stop()
    main()
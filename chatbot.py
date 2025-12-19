import os
import glob
import streamlit as st
import shutil
import re
import uuid
import unicodedata 
from pathlib import Path
from typing import List, Optional

# --- Imports (Giữ nguyên thư viện như yêu cầu) ---
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
    # Rerank optimization
    from flashrank import Ranker, RerankRequest
    DEPENDENCIES_OK = True
except ImportError as e:
    DEPENDENCIES_OK = False
    IMPORT_ERROR = str(e)

# =============================
# 1. CẤU HÌNH HỆ THỐNG (CONFIG) 
# =============================

st.set_page_config(
    page_title="KTC Chatbot - THCS & THPT Phạm Kiệt",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

class AppConfig:
    # --- Cấu hình API & Model ---
    GROQ_API_KEY = "gsk_..." # ĐIỀN API KEY CỦA THẦY VÀO ĐÂY HOẶC DÙNG ST.SECRETS
    LLM_MODEL = "llama-3.3-70b-versatile" 
    EMBEDDING_MODEL = "dangvantuan/vietnamese-embedding"
    
    # --- Cấu hình thư mục ---
    UPLOAD_DIR = "PDF_KNOWLEDGE"
    VECTOR_DB_DIR = "FAISS_DB"
    LOGO_PROJECT = "LOGO.jpg"
    
    # --- Prompt Engineering (Giữ nguyên) ---
    SYSTEM_PROMPT = """Bạn là Trợ lý học tập môn Tin học, hỗ trợ học sinh dựa trên SGK Kết nối tri thức (KNTT).
    
    QUY TẮC TRẢ LỜI:
    1.  CHỈ sử dụng thông tin từ ngữ cảnh (Context) được cung cấp bên dưới.
    2.  Nếu không tìm thấy thông tin trong Context, hãy nói rõ: "Dựa trên tài liệu SGK hiện có, tôi chưa tìm thấy thông tin này."
    3.  Trích dẫn nguồn chính xác: (Tên sách - Chủ đề - Bài).
    4.  Giọng văn: Sư phạm, khích lệ, dễ hiểu, phù hợp học sinh.
    
    CẤU TRÚC TRẢ LỜI:
    - **Lời giải đáp:** [Nội dung chi tiết]
    - **Nguồn tham khảo:** [Tự động trích xuất từ metadata]
    """

# ========================================================
# 2. XỬ LÝ DỮ LIỆU ĐẶC THÙ CHO SGK KNTT (CORE RAG LOGIC)
# ========================================================

class KNTT_TextProcessor:
    """
    Class chuyên biệt để xử lý cấu trúc: Tên sách -> Chủ đề -> Bài
    Dành cho dự thi KHKT - Tối ưu hóa việc truy xuất nguồn.
    """
    
    @staticmethod
    def normalize_text(text: str) -> str:
        """Làm sạch văn bản cơ bản."""
        if not text: return ""
        text = unicodedata.normalize("NFC", text)
        return re.sub(r'\s+', ' ', text).strip()

    @staticmethod
    def parse_structure(text: str, filename: str) -> List[Document]:
        """
        Phân tích cú pháp văn bản theo cấu trúc: Chủ đề -> Bài.
        Chỉ giữ lại nội dung thuộc (Chủ đề AND Bài).
        """
        lines = text.split('\n')
        structured_docs = []
        
        # Regex bắt "Chủ đề" (Ví dụ: Chủ đề 1, Chủ đề E...)
        # Bắt các biến thể: "Chủ đề 1", "CHỦ ĐỀ 1", "Chủ đề A:"
        topic_pattern = re.compile(r'^(?:Chủ đề|CHỦ ĐỀ)\s+([0-9A-Za-z]+)(?:[:\.]|\s+)(.+)$', re.IGNORECASE)
        
        # Regex bắt "Bài" (Ví dụ: Bài 1, Bài 5...)
        lesson_pattern = re.compile(r'^(?:Bài|BÀI)\s+([0-9]+)(?:[:\.]|\s+)(.+)$', re.IGNORECASE)

        current_topic_id = None
        current_topic_title = None
        current_lesson_id = None
        current_lesson_title = None
        
        current_buffer = []
        
        # Tên sách chuẩn hóa (Bỏ đuôi .pdf)
        source_name = os.path.splitext(filename)[0]

        def commit_chunk():
            """Lưu đoạn văn bản hiện tại nếu đủ điều kiện (Có Topic AND Lesson)."""
            nonlocal current_buffer
            content = "\n".join(current_buffer).strip()
            
            # ĐIỀU KIỆN SỐNG CÒN: Phải có cả Chủ đề và Bài mới lưu
            if content and current_topic_id and current_lesson_id:
                # Tạo Topic đầy đủ: "Chủ đề 1. Máy tính..."
                full_topic = f"Chủ đề {current_topic_id}: {current_topic_title}"
                # Tạo Lesson đầy đủ: "Bài 5. Dữ liệu..."
                full_lesson = f"Bài {current_lesson_id}: {current_lesson_title}"
                
                doc = Document(
                    page_content=content,
                    metadata={
                        "source": source_name,
                        "topic": full_topic,
                        "lesson": full_lesson,
                        "chunk_uid": str(uuid.uuid4())
                    }
                )
                structured_docs.append(doc)
            
            # Reset buffer sau khi commit (hoặc bỏ qua)
            current_buffer = []

        for line in lines:
            line_clean = KNTT_TextProcessor.normalize_text(line)
            if not line_clean:
                continue

            # 1. Kiểm tra xem dòng này có phải là CHỦ ĐỀ mới không?
            topic_match = topic_pattern.match(line_clean)
            if topic_match:
                commit_chunk() # Lưu nội dung bài cũ trước khi sang chủ đề mới
                current_topic_id = topic_match.group(1).strip()
                current_topic_title = topic_match.group(2).strip()
                current_lesson_id = None # Sang chủ đề mới thì reset bài
                current_lesson_title = None
                continue # Dòng tiêu đề không đưa vào nội dung body

            # 2. Kiểm tra xem dòng này có phải là BÀI mới không?
            lesson_match = lesson_pattern.match(line_clean)
            if lesson_match:
                commit_chunk() # Lưu nội dung phần trước
                current_lesson_id = lesson_match.group(1).strip()
                current_lesson_title = lesson_match.group(2).strip()
                continue

            # 3. Nội dung thường
            # Chỉ thu thập nếu ĐÃ xác định được đang ở trong Chủ đề nào và Bài nào
            if current_topic_id and current_lesson_id:
                current_buffer.append(line_clean)

        # Commit phần cuối cùng
        commit_chunk()
        
        return structured_docs

class VectorStoreManager:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(model_name=AppConfig.EMBEDDING_MODEL)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

    def build_db(self, uploaded_files):
        """Quy trình: LlamaParse -> Cấu trúc hóa KNTT -> Split -> Vector DB"""
        
        if not os.path.exists(AppConfig.UPLOAD_DIR):
            os.makedirs(AppConfig.UPLOAD_DIR)

        all_processed_docs = []
        status_text = st.empty()

        for uploaded_file in uploaded_files:
            file_path = os.path.join(AppConfig.UPLOAD_DIR, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            status_text.text(f"⏳ Đang xử lý cấu trúc SGK: {uploaded_file.name}...")
            
            # 1. Parse PDF sang Markdown/Text bằng LlamaParse
            # (Giả định API Key LlamaParse đã được set trong môi trường hoặc code)
            parser = LlamaParse(result_type="text", language="vi")
            parsed_result = parser.load_data(file_path)
            
            if not parsed_result:
                continue
                
            raw_text = parsed_result[0].text
            
            # 2. KỸ THUẬT QUAN TRỌNG: Cấu trúc hóa dữ liệu theo chuẩn KNTT
            # Bước này lọc bỏ rác và gắn thẻ Topic/Lesson
            structured_docs = KNTT_TextProcessor.parse_structure(raw_text, uploaded_file.name)
            
            # 3. Chia nhỏ chunk (nhưng vẫn giữ metadata đã gắn)
            # Dùng split_documents để bảo toàn metadata source/topic/lesson
            chunks = self.text_splitter.split_documents(structured_docs)
            all_processed_docs.extend(chunks)

        if not all_processed_docs:
            st.error("❌ Không tìm thấy nội dung hợp lệ (Chủ đề -> Bài) trong tài liệu!")
            return None

        # 4. Tạo Vector DB
        status_text.text(f"🧠 Đang mã hóa {len(all_processed_docs)} đoạn tri thức...")
        vector_db = FAISS.from_documents(all_processed_docs, self.embeddings)
        vector_db.save_local(AppConfig.VECTOR_DB_DIR)
        
        # Lưu cache BM25 (cho Hybrid Search)
        with open(f"{AppConfig.VECTOR_DB_DIR}/bm25_docs.pkl", "wb") as f:
            import pickle
            pickle.dump(all_processed_docs, f)
            
        status_text.empty()
        return vector_db

    def load_db(self):
        if os.path.exists(AppConfig.VECTOR_DB_DIR):
            return FAISS.load_local(
                AppConfig.VECTOR_DB_DIR, 
                self.embeddings, 
                allow_dangerous_deserialization=True
            )
        return None

# ==================================
# 3. RAG ENGINE (Hybrid + Rerank)
# ==================================

class RAGEngine:
    @staticmethod
    def get_retriever(vector_db):
        # 1. Vector Retriever
        faiss_retriever = vector_db.as_retriever(search_kwargs={"k": 5})
        
        # 2. BM25 Retriever (Keyword)
        try:
            with open(f"{AppConfig.VECTOR_DB_DIR}/bm25_docs.pkl", "rb") as f:
                import pickle
                docs = pickle.load(f)
            bm25_retriever = BM25Retriever.from_documents(docs)
            bm25_retriever.k = 5
            
            # 3. Hybrid Ensemble
            ensemble_retriever = EnsembleRetriever(
                retrievers=[bm25_retriever, faiss_retriever],
                weights=[0.4, 0.6]
            )
            return ensemble_retriever
        except:
            return faiss_retriever

    @staticmethod
    def generate_response(client, retriever, query):
        # A. Retrieve
        docs = retriever.invoke(query)
        
        # B. Rerank (Tối ưu hóa thứ hạng)
        # Nếu thầy Khanh chưa cài flashrank hoặc muốn tắt thì comment đoạn này
        try:
            ranker = Ranker(model_name="ms-marco-MiniLM-L-12-v2", cache_dir="./opt")
            rerank_request = RerankRequest(query=query, passages=[
                {"id": d.metadata.get("chunk_uid", "0"), "text": d.page_content, "meta": d.metadata} 
                for d in docs
            ])
            results = ranker.rank(rerank_request)
            # Lấy top 3 sau rerank
            top_docs = results[:3]
            context_text = ""
            sources_set = set()
            
            for r in top_docs:
                meta = r['meta']
                # Tạo chuỗi nguồn chuẩn: Tin 10 - Chủ đề 1 - Bài 5
                src_str = f"{meta.get('source')} -> {meta.get('topic')} -> {meta.get('lesson')}"
                sources_set.add(src_str)
                context_text += f"\n---\nNội dung: {r['text']}\nNguồn: {src_str}\n"
                
        except Exception as e:
            # Fallback nếu Rerank lỗi
            top_docs = docs[:3]
            context_text = ""
            sources_set = set()
            for d in top_docs:
                meta = d.metadata
                src_str = f"{meta.get('source', 'Unknown')} -> {meta.get('topic', 'Unknown')} -> {meta.get('lesson', 'Unknown')}"
                sources_set.add(src_str)
                context_text += f"\n---\nNội dung: {d.page_content}\nNguồn: {src_str}\n"

        # C. Generate
        full_prompt = f"""{AppConfig.SYSTEM_PROMPT}
        
        CÂU HỎI CỦA HỌC SINH: {query}
        
        DỮ LIỆU SGK THAM KHẢO (ĐÃ ĐƯỢC LỌC):
        {context_text}
        
        HÃY TRẢ LỜI:"""

        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": full_prompt}],
            model=AppConfig.LLM_MODEL,
            stream=True,
        )

        for chunk in chat_completion:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
        
        # Hiển thị nguồn cuối câu trả lời (Optional - hoặc để LLM tự nói)
        yield "\n\n**📚 Nguồn SGK:**\n" + "\n".join([f"- {s}" for s in sources_set])

# =======================
# 4. GIAO DIỆN STREAMLIT
# =======================

def main():
    if not DEPENDENCIES_OK:
        st.error(f"❌ Thiếu thư viện: {IMPORT_ERROR}. Vui lòng chạy: pip install -r requirements.txt")
        return

    # Sidebar quản lý dữ liệu
    with st.sidebar:
        st.image(AppConfig.LOGO_PROJECT if os.path.exists(AppConfig.LOGO_PROJECT) else "https://via.placeholder.com/150", width=100)
        st.title("🗂️ KHO TRI THỨC KNTT")
        
        uploaded_files = st.file_uploader(
            "Nạp SGK (PDF)", 
            type=["pdf"], 
            accept_multiple_files=True
        )
        
        if st.button("🔄 Cập nhật Tri thức (Build RAG)"):
            if uploaded_files:
                manager = VectorStoreManager()
                with st.spinner("Đang xây dựng lại não bộ AI..."):
                    db = manager.build_db(uploaded_files)
                    if db:
                        st.success("✅ Đã học xong SGK mới!")
                        st.session_state.vector_db = db
                        st.rerun()
            else:
                st.warning("⚠️ Vui lòng chọn file PDF SGK!")

        st.markdown("---")
        st.markdown("**Hướng dẫn:**\nUpload file PDF SGK có tên chuẩn (VD: Tin 10_KNTT.pdf). Hệ thống tự động lọc theo Chủ đề/Bài.")

    # Main Chat Interface
    st.title("🤖 TRỢ LÝ HỌC TẬP TIN HỌC (RAG SYSTEM)")
    
    # Khởi tạo session
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "retriever_engine" not in st.session_state:
        manager = VectorStoreManager()
        db = manager.load_db()
        if db:
            st.session_state.retriever_engine = RAGEngine.get_retriever(db)
        else:
            st.info("👋 Xin chào! Hãy nạp tài liệu SGK ở cột bên trái để bắt đầu.")

    # Hiển thị lịch sử chat
    for msg in st.session_state.messages:
        avatar = "🧑‍🎓" if msg["role"] == "user" else "🤖"
        with st.chat_message(msg["role"], avatar=avatar):
            st.markdown(msg["content"])

    # Xử lý input
    if user_input := st.chat_input("Nhập câu hỏi bài học (Ví dụ: Trí tuệ nhân tạo là gì trong bài 17?)"):
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user", avatar="🧑‍🎓"):
            st.markdown(user_input)

        with st.chat_message("assistant", avatar="🤖"):
            if "retriever_engine" not in st.session_state:
                st.warning("⚠️ Chưa có dữ liệu SGK! Vui lòng nạp file bên trái.")
            else:
                try:
                    # Init Groq Client (Thêm API Key vào đây hoặc Secrets)
                    groq_client = Groq(api_key=AppConfig.GROQ_API_KEY)
                    
                    response_placeholder = st.empty()
                    full_response = ""
                    
                    # Gọi hàm Generate
                    response_gen = RAGEngine.generate_response(
                        groq_client, 
                        st.session_state.retriever_engine, 
                        user_input
                    )
                    
                    for chunk in response_gen:
                        full_response += chunk
                        response_placeholder.markdown(full_response + "▌")
                    
                    response_placeholder.markdown(full_response)
                    
                    st.session_state.messages.append({"role": "assistant", "content": full_response})
                    
                except Exception as e:
                    st.error(f"Lỗi hệ thống: {str(e)}")

if __name__ == "__main__":
    main()
# ==============================================================================
#   DỰ ÁN KHKT: TRỢ LÝ ẢO TRA CỨU KIẾN THỨC TIN HỌC (KTC AI)
#   Tác giả: Nhóm KHKT THCS & THPT Phạm Kiệt
#   GVHD: Thầy Khanh
# ==============================================================================

import os
import glob
import time
import streamlit as st
from typing import List, Tuple

# --- AI & Data Processing Libraries ---
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

# --- Translation Libraries ---
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

# --- LLM Client ---
from groq import Groq

# ==============================================================================
# 1. CẤU HÌNH HỆ THỐNG & HẰNG SỐ
# ==============================================================================

st.set_page_config(
    page_title="KTC Assistant - Trợ lý Tin học",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

class AppConfig:
    # Model Settings
    LLM_MODEL = 'llama-3.1-8b-instant'
    EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    TRANSLATION_MODEL = "Helsinki-NLP/opus-mt-vi-en"
    
    # Data Settings
    PDF_DIR = "PDF_KNOWLEDGE"
    VECTOR_DB_PATH = "faiss_db_index"
    
    # RAG Settings (Đã tinh chỉnh để tìm kiếm sâu hơn)
    CHUNK_SIZE = 1000 # Tăng kích thước đoạn đọc để hiểu ngữ cảnh tốt hơn
    CHUNK_OVERLAP = 200
    TOP_K_RETRIEVAL = 6 # Tăng số lượng nguồn tham khảo lên 6 để tránh bỏ sót

# ==============================================================================
# 2. GIAO DIỆN (CSS & STYLING) - ĐÃ TỐI ƯU CHO GỌN
# ==============================================================================

def inject_custom_css():
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
        
        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
        }
        
        /* Thu gọn khoảng cách đầu trang để đỡ tốn diện tích */
        .block-container {
            padding-top: 2rem !important; 
            padding-bottom: 2rem !important;
        }
        
        /* Header Styling */
        .main-header {
            background: linear-gradient(90deg, #0f4c81 0%, #00c6ff 100%);
            padding: 15px; /* Giảm padding */
            border-radius: 10px;
            color: white;
            text-align: center;
            margin-bottom: 15px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        .main-header h1 {
            margin: 0;
            font-size: 2rem; /* Giảm cỡ chữ tiêu đề chút */
            font-weight: 700;
            color: white !important;
        }
        .main-header p {
            font-size: 1rem;
            opacity: 0.9;
            margin-bottom: 0px;
        }

        /* Sidebar Info - Làm gọn tối đa */
        .project-info {
            background-color: #f0f2f6;
            padding: 10px; /* Giảm padding */
            border-radius: 8px;
            font-size: 0.85rem; /* Chữ nhỏ lại xíu cho gọn */
            line-height: 1.4;
            border-left: 4px solid #0f4c81;
            margin-bottom: 10px;
        }
        
        /* Ẩn bớt khoảng trắng thừa mặc định của Streamlit Sidebar */
        section[data-testid="stSidebar"] > div {
            padding-top: 1rem;
        }
    </style>
    """, unsafe_allow_html=True)

# ==============================================================================
# 3. QUẢN LÝ TÀI NGUYÊN (CACHING RESOURCE)
# ==============================================================================

@st.cache_resource(show_spinner=False)
def load_groq_client():
    """Khởi tạo kết nối đến Groq API."""
    try:
        api_key = st.secrets["GROQ_API_KEY"]
        return Groq(api_key=api_key)
    except Exception as e:
        st.error(f"❌ Lỗi cấu hình API Key: {e}")
        return None

@st.cache_resource(show_spinner=False)
def load_embedding_model():
    """Tải model Embedding (Chạy 1 lần duy nhất)."""
    return HuggingFaceEmbeddings(model_name=AppConfig.EMBEDDING_MODEL)

@st.cache_resource(show_spinner=False)
def load_translator():
    """Tải model Dịch thuật (Chạy 1 lần duy nhất)."""
    try:
        tokenizer = AutoTokenizer.from_pretrained(AppConfig.TRANSLATION_MODEL)
        model = AutoModelForSeq2SeqLM.from_pretrained(AppConfig.TRANSLATION_MODEL)
        translator = pipeline(
            "translation", 
            model=model, 
            tokenizer=tokenizer, 
            src_lang="vi", 
            tgt_lang="en"
        )
        return translator
    except Exception as e:
        print(f"Translator Error: {e}")
        return None

# ==============================================================================
# 4. XỬ LÝ DỮ LIỆU & RAG LOGIC
# ==============================================================================

class KnowledgeBaseManager:
    def __init__(self):
        self.embeddings = load_embedding_model()
    
    def load_documents(self) -> List[Document]:
        """Đọc toàn bộ file PDF trong thư mục."""
        if not os.path.exists(AppConfig.PDF_DIR):
            os.makedirs(AppConfig.PDF_DIR)
            return []
        
        pdf_files = glob.glob(os.path.join(AppConfig.PDF_DIR, "*.pdf"))
        docs = []
        
        for pdf_path in pdf_files:
            try:
                reader = PdfReader(pdf_path)
                filename = os.path.basename(pdf_path)
                for i, page in enumerate(reader.pages):
                    text = page.extract_text()
                    if text and len(text.strip()) > 50: # Chỉ lấy trang có nội dung đáng kể
                        docs.append(Document(
                            page_content=text,
                            metadata={"source": filename, "page": i + 1}
                        ))
            except Exception as e:
                st.warning(f"⚠️ Không đọc được file {pdf_path}. Bỏ qua.")
        
        return docs

    def get_vector_store(self, force_rebuild=False):
        """Tải hoặc xây dựng lại Vector Database."""
        # Nếu thư mục DB tồn tại và không bị ép build lại, thì load lên
        if os.path.exists(AppConfig.VECTOR_DB_PATH) and not force_rebuild:
            try:
                return FAISS.load_local(
                    AppConfig.VECTOR_DB_PATH, 
                    self.embeddings, 
                    allow_dangerous_deserialization=True
                )
            except Exception:
                pass # Nếu lỗi load thì build lại từ đầu

        # Build mới (Chạy khi xóa folder hoặc lần đầu tiên)
        docs = self.load_documents()
        if not docs:
            return None
            
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=AppConfig.CHUNK_SIZE,
            chunk_overlap=AppConfig.CHUNK_OVERLAP
        )
        splits = splitter.split_documents(docs)
        
        vector_db = FAISS.from_documents(splits, self.embeddings)
        vector_db.save_local(AppConfig.VECTOR_DB_PATH)
        return vector_db

# ==============================================================================
# 5. CÁC HÀM HỖ TRỢ (UTILITIES)
# ==============================================================================

def translate_query(text: str, translator) -> str:
    """Dịch câu hỏi từ Việt sang Anh để RAG hoạt động tốt hơn với tài liệu tiếng Anh."""
    if not translator:
        return text
    try:
        # Giới hạn độ dài để tránh lỗi model
        result = translator(text[:512]) 
        return result[0]['translation_text']
    except Exception:
        return text

def retrieve_info(vector_db, query: str) -> Tuple[str, List[str]]:
    """Tìm kiếm thông tin liên quan trong Vector DB."""
    if not vector_db:
        return "", []
    
    try:
        results = vector_db.similarity_search(query, k=AppConfig.TOP_K_RETRIEVAL)
        context_str = ""
        sources_list = []
        
        for doc in results:
            context_str += f"---\nNội dung: {doc.page_content}\n"
            sources_list.append(f"📄 {doc.metadata['source']} (Trang {doc.metadata['page']})")
            
        return context_str, list(set(sources_list)) # Remove duplicates
    except Exception as e:
        return "", []

def generate_response_stream(client, context, question):
    """Tạo câu trả lời từ LLM (Streaming) - Đã tối ưu cho câu hỏi tổng quan."""
    
    system_prompt = f"""
    Bạn là KTC Assistant, một trợ lý giáo dục chuyên nghiệp.
    
    [QUY TRÌNH TƯ DUY VÀ TRẢ LỜI]:
    1. Phân tích câu hỏi:
       - Nếu là câu hỏi chi tiết (Ví dụ: "Hàm IF là gì?"): Chỉ dùng thông tin trong [DỮ LIỆU TRA CỨU] để trả lời chính xác.
       - Nếu là câu hỏi TỔNG QUÁT/TÓM TẮT (Ví dụ: "Tóm tắt Tin 12", "Tin 10 học gì?"):
         + Nếu [DỮ LIỆU TRA CỨU] quá ít hoặc chỉ nói về một phần nhỏ (ví dụ chỉ nói về đạo đức), bạn ĐƯỢC PHÉP dùng kiến thức chuẩn của mình về chương trình GDPT 2018 để khái quát nội dung chính (Liệt kê các chủ đề/Module chính).
         + Sau đó dùng [DỮ LIỆU TRA CỨU] để trích dẫn chi tiết minh họa.
    
    2. Phong cách:
       - Sư phạm, dễ hiểu, trình bày mạch lạc (dùng Markdown, gạch đầu dòng).
       - Luôn trích dẫn nguồn nếu lấy từ tài liệu.

    [DỮ LIỆU TRA CỨU TỪ SÁCH GIÁO KHOA]:
    {context}
    """
    
    try:
        stream = client.chat.completions.create(
            model=AppConfig.LLM_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question}
            ],
            stream=True,
            temperature=0.5 # Tăng nhẹ độ sáng tạo để AI tổng hợp tốt hơn
        )
        return stream
    except Exception as e:
        return f"Lỗi kết nối AI: {str(e)}"
# ==============================================================================
# 6. MAIN APP LOOP (LOGO NHỎ & CÂN ĐỐI)
# ==============================================================================

def main():
    inject_custom_css()
    
    # --- Cấu hình Sidebar ---
    with st.sidebar:
        # 1. Xử lý Logo: Chia cột để logo nhỏ lại và nằm giữa (Tỷ lệ 1-3-1)
        if os.path.exists("LOGO.jpg"):
            c1, c2, c3 = st.columns([1, 3, 1]) 
            with c2: 
                st.image("LOGO.jpg", use_container_width=True)
        else:
            st.title("🤖")
        
        st.write("") # Tạo khoảng cách nhỏ xíu dưới logo
        
        # 2. Thông tin dự án (Gọn gàng)
        st.markdown("""
        <div class="project-info">
            <div style="text-align: center; font-weight: bold; margin-bottom: 5px; color: #0f4c81;">
                🏆 SẢN PHẨM DỰ THI<br>KHKT CẤP TRƯỜNG
            </div>
            <b>THCS & THPT Phạm Kiệt</b><br>
            Tác giả: Bùi Tá Tùng & Cao Sỹ Bảo Chung<br>
            GVHD: Thầy Khanh
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Nút xóa lịch sử
        if st.button("🗑️ Xóa lịch sử chat", use_container_width=True, key="btn_clear"):
            st.session_state.messages = []
            st.rerun()

    # --- Giao diện chính (Bên phải) ---
    st.markdown("""
    <div class="main-header">
        <h1>🎓 TRỢ LÝ ẢO KTC AI</h1>
        <p>Hỗ trợ tra cứu kiến thức Tin học & Nghiên cứu khoa học</p>
    </div>
    """, unsafe_allow_html=True)

    # Khởi tạo Session State
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Chào bạn! Mình là **KTC AI**. Mình có thể giúp gì cho bài học hôm nay? 🧑‍💻"}
        ]
    
    # Load Resources
    groq_client = load_groq_client()
    translator = load_translator()
    
    if "vector_db" not in st.session_state:
        with st.spinner("🚀 Đang khởi động hệ thống tri thức..."):
            # Logic: Nếu không tìm thấy vector_db trên đĩa thì tự động build lại
            kb = KnowledgeBaseManager()
            st.session_state.vector_db = kb.get_vector_store()

    if not groq_client:
        st.error("⚠️ Lỗi API Key: Vui lòng kiểm tra file cấu hình secrets.")
        st.stop()

    # Hiển thị lịch sử chat
    for msg in st.session_state.messages:
        avatar = "🧑‍🎓" if msg["role"] == "user" else "🤖"
        with st.chat_message(msg["role"], avatar=avatar):
            st.markdown(msg["content"])

    # Xử lý Chat Input
    if prompt := st.chat_input("Nhập câu hỏi của bạn tại đây..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user", avatar="🧑‍🎓"):
            st.markdown(prompt)

        with st.chat_message("assistant", avatar="🤖"):
            message_placeholder = st.empty()
            
            search_query = prompt
            if translator:
                translated = translate_query(prompt, translator)
                if translated and translated != prompt:
                    search_query = translated

            context_text, sources = retrieve_info(st.session_state.vector_db, search_query)
            
            if not context_text:
                context_text = "Không tìm thấy thông tin cụ thể trong tài liệu. Trả lời dựa trên kiến thức chung."

            stream = generate_response_stream(groq_client, context_text, prompt)
            
            full_response = ""
            if isinstance(stream, str):
                full_response = stream
                message_placeholder.markdown(full_response)
            else:
                for chunk in stream:
                    if chunk.choices[0].delta.content:
                        full_response += chunk.choices[0].delta.content
                        message_placeholder.markdown(full_response + "▌")
                message_placeholder.markdown(full_response)

            if sources:
                with st.expander("📚 Tài liệu tham khảo & Minh chứng"):
                    for src in sources:
                        st.markdown(f"- {src}")
            
            st.session_state.messages.append({"role": "assistant", "content": full_response})

if __name__ == "__main__":
    main()
import os
import glob
import time
import base64
import streamlit as st
from pathlib import Path

# --- Imports với xử lý lỗi ---
try:
    from pypdf import PdfReader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_community.vectorstores import FAISS
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_core.documents import Document
    from groq import Groq
    DEPENDENCIES_OK = True
except ImportError as e:
    DEPENDENCIES_OK = False
    IMPORT_ERROR = str(e)

# ==============================================================================
# 1. CẤU HÌNH HỆ THỐNG
# ==============================================================================

st.set_page_config(
    page_title="KTC Chatbot - THCS & THPT Phạm Kiệt",
    page_icon="LOGO.jpg",
    layout="wide",
    initial_sidebar_state="expanded"
)

class AppConfig:
    LLM_MODEL = 'llama-3.1-8b-instant'
    EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    PDF_DIR = "PDF_KNOWLEDGE"
    VECTOR_DB_PATH = "faiss_db_index"
    
    # Tên file ảnh (Đảm bảo file nằm cùng thư mục code)
    LOGO_PROJECT = "LOGO.jpg"     # Logo Nhóm KTC
    LOGO_SCHOOL = "LOGO PKS.png"  # Logo Trường Phạm Kiệt
    
    CHUNK_SIZE = 1000 
    CHUNK_OVERLAP = 200
    TOP_K_RETRIEVAL = 4

# ==============================================================================
# 2. HÀM HỖ TRỢ XỬ LÝ ẢNH (CHO HEADER)
# ==============================================================================

def get_img_as_base64(file_path):
    """Chuyển đổi ảnh sang base64 để nhúng vào HTML Header"""
    if not os.path.exists(file_path):
        return ""
    with open(file_path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

# ==============================================================================
# 3. UI/UX: GIAO DIỆN HI-TECH (CSS NÂNG CAO)
# ==============================================================================

def inject_custom_css():
    st.markdown("""
    <style>
        /* Import Font hiện đại 'Inter' */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;800&display=swap');
        
        /* 1. GLOBAL FONT SETTINGS */
        html, body, [class*="css"], .stMarkdown, .stButton, .stTextInput, .stChatInput {
            font-family: 'Inter', sans-serif !important;
        }
        
        /* 2. SIDEBAR STYLING */
        section[data-testid="stSidebar"] {
            background-color: #f8f9fa;
            border-right: 1px solid #e9ecef;
        }
        
        /* Card thông tin Sidebar */
        .project-card {
            background: white;
            padding: 15px;
            border-radius: 12px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.05);
            margin-bottom: 20px;
            border: 1px solid #dee2e6;
        }
        
        .project-title {
            color: #0077b6;
            font-weight: 800;
            font-size: 1.1rem;
            margin-bottom: 5px;
            text-align: center;
            text-transform: uppercase;
        }
        
        .project-sub {
            font-size: 0.8rem;
            color: #6c757d;
            text-align: center;
            margin-bottom: 15px;
            font-style: italic;
        }

        /* Table định dạng thông tin trong Sidebar */
        .info-table {
            width: 100%;
            border-collapse: collapse;
            font-size: 0.85rem;
        }
        .info-table td {
            padding: 6px 0;
            vertical-align: top;
        }
        .info-label {
            font-weight: 700;
            color: #495057;
            width: 65px; /* Cố định chiều rộng nhãn để thẳng hàng */
        }
        .info-value {
            color: #212529;
            text-align: right;
        }

        /* 3. MAIN HEADER - 2 CỘT */
        .main-header {
            background: linear-gradient(135deg, #023e8a 0%, #0077b6 100%);
            padding: 1.5rem 2rem;
            border-radius: 15px;
            color: white;
            margin-bottom: 2rem;
            box-shadow: 0 8px 20px rgba(0, 119, 182, 0.3);
            display: flex;
            align-items: center;
            justify-content: space-between; /* Đẩy 2 phần sang 2 bên */
        }
        
        .header-left h1 {
            color: #caf0f8 !important;
            font-weight: 900;
            margin: 0;
            font-size: 2.2rem;
            letter-spacing: -0.5px;
        }
        
        .header-left p {
            color: #e0fbfc;
            margin: 5px 0 0 0;
            font-size: 1rem;
            opacity: 0.9;
        }
        
        .header-right img {
            border-radius: 50%; /* Bo tròn logo nhóm */
            border: 3px solid rgba(255,255,255,0.3);
            box-shadow: 0 4px 10px rgba(0,0,0,0.2);
            width: 100px; /* Kích thước logo */
            height: 100px;
            object-fit: cover;
        }

        /* 4. CHAT BUBBLES */
        [data-testid="stChatMessageContent"] {
            border-radius: 15px !important;
            padding: 1rem !important;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        }
        [data-testid="stChatMessageContent"]:has(+ [data-testid="stChatMessageAvatar"]) {
            background: #e3f2fd;
            color: #0d47a1;
        }
        [data-testid="stChatMessageContent"]:not(:has(+ [data-testid="stChatMessageAvatar"])) {
            background: white;
            border: 1px solid #e9ecef;
            border-left: 5px solid #00b4d8;
        }

        /* 5. BUTTONS */
        div.stButton > button {
            border-radius: 8px;
            background-color: white;
            color: #0077b6;
            border: 1px solid #90e0ef;
            transition: all 0.2s;
        }
        div.stButton > button:hover {
            background-color: #0077b6;
            color: white;
            border-color: #0077b6;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }

        /* Ẩn footer mặc định */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)

# ==============================================================================
# 4. LOGIC BACKEND
# ==============================================================================

@st.cache_resource(show_spinner=False)
def load_groq_client():
    try:
        api_key = st.secrets.get("GROQ_API_KEY") or os.environ.get("GROQ_API_KEY")
        if not api_key: return None
        return Groq(api_key=api_key)
    except: return None

@st.cache_resource(show_spinner=False)
def load_embedding_model():
    try:
        return HuggingFaceEmbeddings(
            model_name=AppConfig.EMBEDDING_MODEL,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
    except: return None

def load_vector_db(embeddings):
    if not embeddings: return None
    if os.path.exists(AppConfig.VECTOR_DB_PATH):
        try:
            return FAISS.load_local(AppConfig.VECTOR_DB_PATH, embeddings, allow_dangerous_deserialization=True)
        except: pass
    if not os.path.exists(AppConfig.PDF_DIR): return None
    pdf_files = glob.glob(os.path.join(AppConfig.PDF_DIR, "*.pdf"))
    if not pdf_files: return None
    docs = []
    for pdf_path in pdf_files:
        try:
            reader = PdfReader(pdf_path)
            for page_num, page in enumerate(reader.pages):
                text = page.extract_text()
                if text and len(text.strip()) > 50:
                    docs.append(Document(page_content=text, metadata={"source": os.path.basename(pdf_path), "page": page_num + 1}))
        except: continue
    if docs:
        splitter = RecursiveCharacterTextSplitter(chunk_size=AppConfig.CHUNK_SIZE, chunk_overlap=AppConfig.CHUNK_OVERLAP)
        splits = splitter.split_documents(docs)
        vector_db = FAISS.from_documents(splits, embeddings)
        return vector_db
    return None

def get_rag_response(client, vector_db, query):
    context_text = ""
    sources = []
    if vector_db:
        results = vector_db.similarity_search_with_score(query, k=AppConfig.TOP_K_RETRIEVAL)
        for doc, score in results:
            src = doc.metadata.get('source', 'Tài liệu')
            page = doc.metadata.get('page', '1')
            content = doc.page_content.replace("\n", " ").strip()
            context_text += f"Content: {content}\nSource: {src} (Page {page})\n\n"
            sources.append(f"{src} - Trang {page}")

    system_prompt = f"""Bạn là KTC Chatbot - Trợ lý ảo hỗ trợ học tập môn Tin học (THPT).
    
    NHIỆM VỤ:
    - Trả lời câu hỏi dựa trên thông tin được cung cấp trong [CONTEXT].
    - Hỗ trợ giải bài tập lập trình Python, CSDL và kiến thức Tin học đại cương.
    - Luôn trả lời bằng tiếng Việt, giọng văn sư phạm, dễ hiểu.
    
    [CONTEXT]:
    {context_text}
    """

    try:
        stream = client.chat.completions.create(
            model=AppConfig.LLM_MODEL,
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": query}],
            stream=True,
            temperature=0.3,
            max_tokens=2000
        )
        return stream, list(set(sources))
    except Exception as e:
        return f"Error: {str(e)}", []

# ==============================================================================
# 5. MAIN APP
# ==============================================================================

def main():
    if not DEPENDENCIES_OK:
        st.error(f"⚠️ Lỗi thư viện: {IMPORT_ERROR}")
        st.stop()
        
    inject_custom_css()
    
    # --- SIDEBAR (LOGO TRƯỜNG LÊN TOP) ---
    with st.sidebar:
        # 1. Logo Trường Phạm Kiệt (Trên cùng)
        if os.path.exists(AppConfig.LOGO_SCHOOL):
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                st.image(AppConfig.LOGO_SCHOOL, use_container_width=True)
            st.markdown("<div style='text-align:center; font-weight:700; color:#023e8a; margin-bottom:20px;'>THCS & THPT PHẠM KIỆT</div>", unsafe_allow_html=True)
        
        # 2. Thông tin Dự án (Layout Bảng chống rớt dòng)
        st.markdown("""
        <div class="project-card">
            <div class="project-title">KTC CHATBOT</div>
            <div class="project-sub">Sản phẩm dự thi KHKT cấp trường</div>
            <hr style="margin: 10px 0; border-top: 1px dashed #dee2e6;">
            
            <table class="info-table">
                <tr>
                    <td class="info-label">Tác giả:</td>
                    <td class="info-value">
                        <b>Bùi Tá Tùng</b><br>
                        <b>Cao Sỹ Bảo Chung</b>
                    </td>
                </tr>
                <tr>
                    <td class="info-label">GVHD:</td>
                    <td class="info-value">Thầy <b>Nguyễn Thế Khanh</b></td>
                </tr>
                <tr>
                    <td class="info-label">Năm học:</td>
                    <td class="info-value"><b>2025 - 2026</b></td>
                </tr>
            </table>
        </div>
        """, unsafe_allow_html=True)
        
        # 3. Tiện ích
        st.markdown("### ⚙️ Tiện ích")
        if st.button("🗑️ Xóa lịch sử chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    # --- MAIN CONTENT ---
    
    # Banner Header (2 Cột: Text trái - Logo Nhóm phải)
    logo_nhom_b64 = get_img_as_base64(AppConfig.LOGO_PROJECT)
    img_html = f'<img src="data:image/jpeg;base64,{logo_nhom_b64}" alt="Logo">' if logo_nhom_b64 else ""

    st.markdown(f"""
    <div class="main-header">
        <div class="header-left">
            <h1>KTC CHATBOT</h1>
            <p>Hỗ trợ học tập Tin học & Khoa học máy tính</p>
        </div>
        <div class="header-right">
            {img_html}
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Khởi tạo Chat
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "👋 Chào bạn! Mình là KTC Chatbot. Bạn cần hỗ trợ bài tập Tin học phần nào?"}]
    
    if "vector_db" not in st.session_state:
        with st.spinner("🚀 Đang khởi động hệ thống..."):
            embeddings = load_embedding_model()
            st.session_state.vector_db = load_vector_db(embeddings)

    groq_client = load_groq_client()

    # Hiển thị tin nhắn
    for msg in st.session_state.messages:
        # Avatar: Nếu là bot thì dùng Logo Nhóm (nếu có), không thì dùng icon
        bot_avatar = AppConfig.LOGO_PROJECT if os.path.exists(AppConfig.LOGO_PROJECT) else "🤖"
        avatar = "🧑‍🎓" if msg["role"] == "user" else bot_avatar
        
        with st.chat_message(msg["role"], avatar=avatar):
            st.markdown(msg["content"])

    # Gợi ý câu hỏi (Tin học THPT)
    if len(st.session_state.messages) < 2:
        st.markdown("##### 💡 Gợi ý ôn tập:")
        cols = st.columns(3)
        prompt_btn = None
        
        if cols[0].button("🐍 Python: Số nguyên tố"):
            prompt_btn = "Viết chương trình Python nhập vào một số nguyên n và kiểm tra xem n có phải là số nguyên tố hay không."
        if cols[1].button("🗃️ CSDL: Khóa chính"):
            prompt_btn = "Giải thích khái niệm Khóa chính (Primary Key) trong CSDL quan hệ và cho ví dụ."
        if cols[2].button("⚖️ Luật An ninh mạng"):
            prompt_btn = "Nêu các hành vi bị nghiêm cấm theo Luật An ninh mạng Việt Nam."
        
        if prompt_btn:
            st.session_state.temp_input = prompt_btn
            st.rerun()

    # Input và Xử lý
    if "temp_input" in st.session_state and st.session_state.temp_input:
        user_input = st.session_state.temp_input
        del st.session_state.temp_input
    else:
        user_input = st.chat_input("Nhập câu hỏi của bạn...")

    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user", avatar="🧑‍🎓"):
            st.markdown(user_input)

        with st.chat_message("assistant", avatar=AppConfig.LOGO_PROJECT if os.path.exists(AppConfig.LOGO_PROJECT) else "🤖"):
            response_placeholder = st.empty()
            with st.spinner("Đang suy nghĩ..."):
                if not groq_client:
                    st.error("❌ Chưa kết nối API.")
                    st.stop()
                stream, sources = get_rag_response(groq_client, st.session_state.vector_db, user_input)
            
            full_response = ""
            if isinstance(stream, str):
                response_placeholder.error(stream)
            else:
                for chunk in stream:
                    if chunk.choices[0].delta.content:
                        full_response += chunk.choices[0].delta.content
                        response_placeholder.markdown(full_response + "▌")
                response_placeholder.markdown(full_response)
            
            if sources:
                with st.expander("📚 Tài liệu tham khảo"):
                    for src in sources: st.caption(f"• {src}")
            
            st.session_state.messages.append({"role": "assistant", "content": full_response})

if __name__ == "__main__":
    main()
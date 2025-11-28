import os
import glob
import time
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
    LOGO_PROJECT = "LOGO.jpg"
    LOGO_SCHOOL = "LOGO PKS.png"
    CHUNK_SIZE = 1000 
    CHUNK_OVERLAP = 200
    TOP_K_RETRIEVAL = 4

# ==============================================================================
# 2. UI/UX: GIAO DIỆN HI-TECH (CSS NÂNG CAO)
# ==============================================================================

def inject_custom_css():
    st.markdown("""
    <style>
        /* Import Font hiện đại 'Inter' */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;800&display=swap');
        
        /* 1. GLOBAL FONT SETTINGS - ÉP SANS-SERIF TOÀN BỘ */
        html, body, [class*="css"], .stMarkdown, .stButton, .stTextInput, .stChatInput {
            font-family: 'Inter', sans-serif !important;
        }
        
        /* 2. SIDEBAR STYLING */
        section[data-testid="stSidebar"] {
            background-color: #f8f9fa; /* Màu nền xám nhẹ sạch sẽ */
            border-right: 1px solid #e9ecef;
        }
        
        /* Căn chỉnh khoảng cách nội dung Sidebar */
        div[data-testid="stSidebarUserContent"] {
            padding: 20px 15px;
        }

        /* Card thông tin tác giả */
        .project-card {
            background: white;
            padding: 20px;
            border-radius: 15px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.05);
            margin-bottom: 25px;
            text-align: center;
            border: 1px solid #f1f3f5;
        }
        
        .project-title {
            color: #0077b6;
            font-weight: 800;
            font-size: 1.2rem;
            margin-bottom: 5px;
            letter-spacing: 1px;
        }
        
        .project-desc {
            font-size: 0.85rem;
            color: #6c757d;
            font-style: italic;
            margin-bottom: 15px;
        }
        
        .info-row {
            display: flex;
            justify-content: space-between;
            font-size: 0.9rem;
            margin-bottom: 8px;
            border-bottom: 1px dashed #eee;
            padding-bottom: 4px;
        }
        .info-label { font-weight: 600; color: #495057; }
        .info-val { color: #212529; text-align: right; }

        /* 3. MAIN HEADER - HIỆU ỨNG GLOW */
        .main-header {
            background: linear-gradient(135deg, #000428 0%, #004e92 100%);
            padding: 2rem;
            border-radius: 20px;
            color: white;
            margin-bottom: 2rem;
            /* Đổ bóng 3D */
            box-shadow: 0 10px 25px -5px rgba(0, 78, 146, 0.4); 
            display: flex;
            flex-direction: column;
            justify-content: center;
        }
        
        .header-title h1 {
            color: #00d2ff !important;
            font-weight: 900;
            margin: 0;
            font-size: 2.5rem;
            letter-spacing: -1px;
            text-transform: uppercase;
        }
        
        .header-subtitle {
            font-size: 1.1rem;
            color: #caf0f8;
            margin-top: 5px;
            font-weight: 300;
        }

        /* 4. CHAT BUBBLES */
        [data-testid="stChatMessageContent"] {
            border-radius: 15px !important;
            padding: 1rem !important;
            font-size: 1rem !important;
            line-height: 1.6 !important;
        }
        /* User */
        [data-testid="stChatMessageContent"]:has(+ [data-testid="stChatMessageAvatar"]) {
            background: #e3f2fd;
            color: #0d47a1;
        }
        /* AI */
        [data-testid="stChatMessageContent"]:not(:has(+ [data-testid="stChatMessageAvatar"])) {
            background: white;
            border: 1px solid #e9ecef;
            box-shadow: 0 2px 5px rgba(0,0,0,0.02);
            border-left: 5px solid #00d2ff;
        }

        /* 5. BUTTONS STYLING - ĐỒNG BỘ */
        div.stButton > button {
            width: 100%;
            border-radius: 10px;
            font-weight: 600;
            border: none;
            padding: 0.5rem 1rem;
            transition: all 0.3s ease;
        }
        
        /* Nút phụ (Gợi ý, Xóa lịch sử) */
        div.stButton > button {
            background-color: white;
            color: #0077b6;
            border: 1px solid #bde0fe;
        }
        div.stButton > button:hover {
            background-color: #0077b6;
            color: white;
            border-color: #0077b6;
            transform: translateY(-2px);
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }

        /* Ẩn footer mặc định */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)

# ==============================================================================
# 3. LOGIC BACKEND
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
    # Load nếu đã có file index, nếu chưa có thì thôi (Học sinh sẽ copy file index vào)
    if os.path.exists(AppConfig.VECTOR_DB_PATH):
        try:
            return FAISS.load_local(AppConfig.VECTOR_DB_PATH, embeddings, allow_dangerous_deserialization=True)
        except: pass
        
    # Cơ chế fallback: Nếu chưa có DB thì build từ PDF (chạy lần đầu)
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
                    docs.append(Document(
                        page_content=text,
                        metadata={"source": os.path.basename(pdf_path), "page": page_num + 1}
                    ))
        except: continue
    
    if docs:
        splitter = RecursiveCharacterTextSplitter(chunk_size=AppConfig.CHUNK_SIZE, chunk_overlap=AppConfig.CHUNK_OVERLAP)
        splits = splitter.split_documents(docs)
        vector_db = FAISS.from_documents(splits, embeddings)
        # vector_db.save_local(AppConfig.VECTOR_DB_PATH) # Có thể mở lại nếu cần lưu
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

    system_prompt = f"""Bạn là KTC Assistant - Trợ lý ảo hỗ trợ học tập trường THCS & THPT Phạm Kiệt.
    
    NHIỆM VỤ:
    - Trả lời câu hỏi dựa trên thông tin được cung cấp trong [CONTEXT].
    - Nếu thông tin không có trong [CONTEXT], hãy dùng kiến thức chung.
    - Luôn trả lời bằng tiếng Việt, giọng văn thân thiện, khuyến khích học sinh.
    
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
# 4. MAIN APP
# ==============================================================================

def main():
    if not DEPENDENCIES_OK:
        st.error(f"⚠️ Lỗi thư viện: {IMPORT_ERROR}")
        st.stop()
        
    inject_custom_css()
    
    # --- SIDEBAR (ĐÃ CHỈNH SỬA) ---
    with st.sidebar:
        # 1. Logo Dự án (Căn giữa)
        if os.path.exists(AppConfig.LOGO_PROJECT):
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                st.image(AppConfig.LOGO_PROJECT, use_container_width=True)
        
        # 2. Thông tin Dự án (Layout mới)
        st.markdown("""
        <div class="project-card">
            <div class="project-title">KTC CHATBOT</div>
            <div class="project-desc">Sản phẩm dự thi KHKT cấp trường</div>
            <div class="info-row">
                <span class="info-label">Tác giả:</span>
                <span class="info-val">Tá Tùng & Bảo Chung</span>
            </div>
            <div class="info-row">
                <span class="info-label">GVHD:</span>
                <span class="info-val">Thầy Khanh</span>
            </div>
            <div class="info-row" style="border:none;">
                <span class="info-label">Năm học:</span>
                <span class="info-val">2024 - 2025</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # 3. Công cụ (Đã xóa nút Cập nhật)
        st.markdown("### ⚙️ Tiện ích")
        if st.button("🗑️ Xóa lịch sử chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

        # 4. Logo Trường (Footer)
        st.markdown("---")
        if os.path.exists(AppConfig.LOGO_SCHOOL):
            col1, col2, col3 = st.columns([1, 3, 1])
            with col2:
                st.image(AppConfig.LOGO_SCHOOL, use_container_width=True)
            st.markdown("""
            <div style="text-align: center; color: #6c757d; font-weight: 600; margin-top: 5px;">
                THCS & THPT Phạm Kiệt
            </div>
            """, unsafe_allow_html=True)

    # --- MAIN CONTENT ---
    
    # Banner Header (Đổ bóng & Bo góc)
    st.markdown(f"""
    <div class="main-header">
        <div class="header-title">
            <h1>KTC ASSISTANT</h1>
        </div>
        <div class="header-subtitle">
            Knowledge in Technology & Computer Science
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Init State
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "👋 Chào bạn! Mình là KTC Assistant. Mình có thể giúp gì cho bạn hôm nay?"}]
    
    if "vector_db" not in st.session_state:
        with st.spinner("🚀 Đang khởi động hệ thống..."):
            embeddings = load_embedding_model()
            st.session_state.vector_db = load_vector_db(embeddings)

    groq_client = load_groq_client()

    # Chat History
    for msg in st.session_state.messages:
        avatar = "🧑‍🎓" if msg["role"] == "user" else (AppConfig.LOGO_PROJECT if os.path.exists(AppConfig.LOGO_PROJECT) else "🤖")
        with st.chat_message(msg["role"], avatar=avatar):
            st.markdown(msg["content"])

    # Gợi ý
    if len(st.session_state.messages) < 2:
        st.markdown("##### 💡 Gợi ý câu hỏi:")
        cols = st.columns(3)
        prompt_btn = None
        if cols[0].button("📝 Cấu trúc báo cáo KHKT?"): prompt_btn = "Hãy cho tôi dàn ý chi tiết bài báo cáo dự án KHKT."
        if cols[1].button("🐍 Viết Code Python?"): prompt_btn = "Viết code Python tính tổng danh sách."
        if cols[2].button("🏫 Giới thiệu trường?"): prompt_btn = "Giới thiệu về trường THCS & THPT Phạm Kiệt."
        
        if prompt_btn:
            st.session_state.temp_input = prompt_btn
            st.rerun()

    # Input handling
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
                with st.expander("📚 Nguồn minh chứng"):
                    for src in sources: st.caption(f"• {src}")
            
            st.session_state.messages.append({"role": "assistant", "content": full_response})

if __name__ == "__main__":
    main()
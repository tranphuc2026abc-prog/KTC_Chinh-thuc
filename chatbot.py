import os
import glob
import time
import streamlit as st
from pathlib import Path

# --- Imports với xử lý lỗi thông minh ---
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
# 1. CẤU HÌNH HỆ THỐNG (CONFIG)
# ==============================================================================

st.set_page_config(
    page_title="KTC Chatbot - THCS & THPT Phạm Kiệt",
    page_icon="LOGO.jpg", # Dùng logo dự án làm icon tab
    layout="wide",
    initial_sidebar_state="expanded"
)

class AppConfig:
    """Cấu hình trung tâm"""
    # Thay đổi model ở đây nếu cần
    LLM_MODEL = 'llama-3.1-8b-instant'
    EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    
    # Đường dẫn files
    PDF_DIR = "PDF_KNOWLEDGE"
    VECTOR_DB_PATH = "faiss_db_index"
    LOGO_PROJECT = "LOGO.jpg"     # Logo KTC
    LOGO_SCHOOL = "LOGO PKS.png"  # Logo Trường Phạm Kiệt
    
    # Tham số RAG
    CHUNK_SIZE = 1000 
    CHUNK_OVERLAP = 200
    TOP_K_RETRIEVAL = 4
    MAX_CONTEXT_LENGTH = 3500

# ==============================================================================
# 2. UI/UX: GIAO DIỆN HI-TECH (CUSTOM CSS)
# ==============================================================================

def inject_custom_css():
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;700&display=swap');
        
        /* --- TỔNG THỂ --- */
        html, body, [class*="css"] {
            font-family: 'Roboto', sans-serif;
        }
        
        /* Màu chủ đạo theo Logo KTC: Xanh Cyan (#00E5FF) và Xanh đậm */
        
        /* --- HEADER --- */
        .main-header {
            background: linear-gradient(135deg, #0f2027 0%, #203a43 50%, #2c5364 100%);
            padding: 1.5rem;
            border-radius: 15px;
            color: white;
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 2rem;
            box-shadow: 0 4px 15px rgba(0, 229, 255, 0.2); /* Đổ bóng xanh neon */
            border: 1px solid rgba(255,255,255,0.1);
        }
        
        .header-title h1 {
            color: #00E5FF !important; /* Màu Cyan của logo KTC */
            font-weight: 800;
            margin: 0;
            font-size: 2.2rem;
            text-shadow: 0 0 10px rgba(0, 229, 255, 0.5);
        }
        
        .header-title p {
            margin: 5px 0 0 0;
            font-size: 1rem;
            color: #e0e0e0;
        }

        /* --- SIDEBAR --- */
        [data-testid="stSidebar"] {
            background-color: #f8f9fa;
            border-right: 1px solid #ddd;
        }
        
        .project-card {
            background: white;
            padding: 15px;
            border-radius: 12px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.05);
            margin-bottom: 20px;
            text-align: center;
            border: 1px solid #eee;
        }
        
        .author-info {
            font-size: 0.9rem;
            color: #333;
            margin-top: 10px;
            text-align: left;
        }
        
        .school-logo-container {
            margin-top: 20px;
            text-align: center;
            opacity: 0.9;
        }

        /* --- CHAT AREA --- */
        .stChatMessage {
            background-color: transparent;
        }
        
        /* User message: Màu xanh nhạt dễ chịu */
        [data-testid="stChatMessageContent"]:has(+ [data-testid="stChatMessageAvatar"]) {
            background: linear-gradient(to right, #e3f2fd, #bbdefb);
            border-radius: 20px 20px 5px 20px;
            color: #0d47a1;
            border: none;
            box-shadow: 0 1px 2px rgba(0,0,0,0.1);
        }
        
        /* AI message: Màu trắng sạch sẽ, viền xanh neon nhẹ */
        [data-testid="stChatMessageContent"]:not(:has(+ [data-testid="stChatMessageAvatar"])) {
            background-color: white;
            border: 1px solid #e1f5fe;
            border-left: 4px solid #00E5FF; /* Điểm nhấn KTC */
            border-radius: 5px 20px 20px 20px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.05);
        }

        /* --- SUGGESTION BUTTONS --- */
        div.stButton > button {
            border-radius: 20px;
            border: 1px solid #b3e5fc;
            background-color: white;
            color: #0277bd;
            font-size: 0.9rem;
            transition: all 0.3s;
        }
        div.stButton > button:hover {
            border-color: #00E5FF;
            color: #00E5FF;
            background-color: #e0f7fa;
            transform: translateY(-2px);
        }

        /* Ẩn bớt footer mặc định */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        
    </style>
    """, unsafe_allow_html=True)

# ==============================================================================
# 3. LOGIC BACKEND (ĐÃ TỐI ƯU CACHING)
# ==============================================================================

@st.cache_resource(show_spinner=False)
def load_groq_client():
    try:
        # Ưu tiên lấy từ secrets, nếu không có thì thử biến môi trường
        api_key = st.secrets.get("GROQ_API_KEY") or os.environ.get("GROQ_API_KEY")
        if not api_key: return None
        return Groq(api_key=api_key)
    except Exception: return None

@st.cache_resource(show_spinner=False)
def load_embedding_model():
    """Load model 1 lần duy nhất khi khởi động app"""
    try:
        return HuggingFaceEmbeddings(
            model_name=AppConfig.EMBEDDING_MODEL,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
    except: return None

# Tối ưu: Chỉ quét lại PDF khi file index chưa tồn tại hoặc user yêu cầu
def load_vector_db(embeddings, force_rebuild=False):
    if not embeddings: return None
    
    # Nếu đã có DB và không bắt buộc rebuild -> Load ngay (Nhanh)
    if os.path.exists(AppConfig.VECTOR_DB_PATH) and not force_rebuild:
        try:
            return FAISS.load_local(AppConfig.VECTOR_DB_PATH, embeddings, allow_dangerous_deserialization=True)
        except: pass # Nếu lỗi file cũ thì rebuild

    # Rebuild (Chậm hơn chút)
    if not os.path.exists(AppConfig.PDF_DIR):
        os.makedirs(AppConfig.PDF_DIR, exist_ok=True)
        return None

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
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=AppConfig.CHUNK_SIZE,
            chunk_overlap=AppConfig.CHUNK_OVERLAP
        )
        splits = splitter.split_documents(docs)
        vector_db = FAISS.from_documents(splits, embeddings)
        vector_db.save_local(AppConfig.VECTOR_DB_PATH)
        return vector_db
    return None

def get_rag_response(client, vector_db, query):
    """Xử lý logic RAG: Tìm kiếm -> Tạo prompt -> Stream trả lời"""
    
    # 1. Tìm kiếm ngữ cảnh
    context_text = ""
    sources = []
    
    if vector_db:
        results = vector_db.similarity_search_with_score(query, k=AppConfig.TOP_K_RETRIEVAL)
        context_parts = []
        for doc, score in results:
            # Lọc bớt kết quả không liên quan (score càng nhỏ càng tốt với L2 distance, nhưng FAISS mặc định similarity score khác)
            # Với FAISS mặc định cosine similarity hay L2 cần check kỹ. Ở đây ta lấy top K thôi.
            src = doc.metadata.get('source', 'Tài liệu')
            page = doc.metadata.get('page', '1')
            content = doc.page_content.replace("\n", " ").strip()
            
            context_parts.append(f"Content: {content}\nSource: {src} (Page {page})")
            sources.append(f"{src} - Trang {page}")
        
        context_text = "\n\n".join(context_parts)

    # 2. System Prompt (Guardrails)
    system_prompt = f"""Bạn là KTC Assistant - Trợ lý ảo hỗ trợ học tập trường THCS & THPT Phạm Kiệt.
    
    NHIỆM VỤ:
    - Trả lời câu hỏi dựa trên thông tin được cung cấp trong [CONTEXT].
    - Nếu thông tin không có trong [CONTEXT], hãy dùng kiến thức chung nhưng nói rõ là "Theo kiến thức của tôi...".
    - Luôn trả lời bằng tiếng Việt, giọng văn thân thiện, khuyến khích học sinh.
    - Trình bày Markdown rõ ràng (dùng gạch đầu dòng, bôi đậm ý chính).
    
    [CONTEXT]:
    {context_text}
    """

    # 3. Gọi API
    try:
        stream = client.chat.completions.create(
            model=AppConfig.LLM_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query}
            ],
            stream=True,
            temperature=0.3,
            max_tokens=2000
        )
        return stream, list(set(sources))
    except Exception as e:
        return f"Error: {str(e)}", []

# ==============================================================================
# 4. CHƯƠNG TRÌNH CHÍNH
# ==============================================================================

def main():
    if not DEPENDENCIES_OK:
        st.error(f"⚠️ Lỗi thư viện: {IMPORT_ERROR}. Vui lòng chạy `pip install -r requirements.txt`")
        st.stop()
        
    inject_custom_css()
    
    # --- SIDEBAR: Nơi thể hiện thương hiệu ---
    with st.sidebar:
        # 1. Logo KTC (Dự án)
        if os.path.exists(AppConfig.LOGO_PROJECT):
            st.image(AppConfig.LOGO_PROJECT, use_container_width=True)
        
        # 2. Thông tin dự án
        st.markdown("""
        <div class="project-card">
            <h3 style="margin:0; color:#0277bd;">KTC CHATBOT</h3>
            <p style="font-size:0.8rem; color:gray;">Trợ lý ảo thông minh</p>
            <hr style="margin:10px 0;">
            <div class="author-info">
                <b>👨‍💻 Tác giả:</b> Tá Tùng & Bảo Chung<br>
                <b>🧑‍🏫 GVHD:</b> Thầy Khanh<br>
                <b>🏆 Dự án:</b> KHKT 2024-2025
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # 3. Công cụ
        with st.expander("🛠️ Cài đặt & Dữ liệu"):
            if st.button("🔄 Nạp lại dữ liệu gốc", use_container_width=True):
                with st.spinner("Đang xử lý PDF..."):
                    embeddings = load_embedding_model()
                    st.session_state.vector_db = load_vector_db(embeddings, force_rebuild=True)
                st.success("Dữ liệu đã cập nhật!")
                time.sleep(1)
                st.rerun()
                
            if st.button("🗑️ Xóa lịch sử chat", use_container_width=True):
                st.session_state.messages = []
                st.rerun()

        # 4. Logo Trường (Footer Sidebar)
        st.markdown("---")
        if os.path.exists(AppConfig.LOGO_SCHOOL):
            st.markdown('<div class="school-logo-container">', unsafe_allow_html=True)
            st.image(AppConfig.LOGO_SCHOOL, width=120, caption="THCS & THPT Phạm Kiệt")
            st.markdown('</div>', unsafe_allow_html=True)

    # --- MAIN UI ---
    
    # Header ấn tượng
    st.markdown(f"""
    <div class="main-header">
        <div class="header-title">
            <h1>KTC ASSISTANT</h1>
            <p>Knowledge in Technology & Computer Science</p>
        </div>
        </div>
    """, unsafe_allow_html=True)

    # Init Session State
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "👋 Chào bạn! Mình là KTC Assistant. Mình có thể giúp gì cho bài nghiên cứu hoặc bài tập Tin học của bạn hôm nay?"}
        ]
    
    if "vector_db" not in st.session_state:
        with st.spinner("🚀 Đang khởi động hệ thống AI..."):
            embeddings = load_embedding_model()
            st.session_state.vector_db = load_vector_db(embeddings)

    groq_client = load_groq_client()

    # Hiển thị lịch sử chat
    for msg in st.session_state.messages:
        # Avatar tùy chỉnh: Bot dùng logo KTC (nếu có) hoặc icon robot
        avatar = "🧑‍🎓" if msg["role"] == "user" else (AppConfig.LOGO_PROJECT if os.path.exists(AppConfig.LOGO_PROJECT) else "🤖")
        with st.chat_message(msg["role"], avatar=avatar):
            st.markdown(msg["content"])

    # Gợi ý câu hỏi (Xử lý thông minh không reload xấu)
    if len(st.session_state.messages) < 2:
        st.markdown("#### 💡 Gợi ý câu hỏi:")
        cols = st.columns(3)
        prompt_from_button = None
        
        if cols[0].button("📝 Cấu trúc bài báo cáo?"):
            prompt_from_button = "Hãy cho tôi dàn ý chi tiết bài báo cáo dự án KHKT."
        if cols[1].button("🐍 Code Python cơ bản?"):
            prompt_from_button = "Viết cho tôi một đoạn code Python tính tổng danh sách."
        if cols[2].button("🏫 Giới thiệu về trường?"):
            prompt_from_button = "Giới thiệu đôi nét về trường THCS & THPT Phạm Kiệt."
            
        if prompt_from_button:
            # Gán vào input giả lập
            st.session_state.temp_input = prompt_from_button
            st.rerun()

    # Xử lý input (từ chat box hoặc từ button gợi ý)
    if "temp_input" in st.session_state and st.session_state.temp_input:
        user_input = st.session_state.temp_input
        del st.session_state.temp_input # Xóa ngay sau khi lấy
    else:
        user_input = st.chat_input("Nhập câu hỏi của bạn tại đây...")

    if user_input:
        # 1. Hiển thị câu hỏi User
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user", avatar="🧑‍🎓"):
            st.markdown(user_input)

        # 2. Xử lý trả lời
        with st.chat_message("assistant", avatar=AppConfig.LOGO_PROJECT if os.path.exists(AppConfig.LOGO_PROJECT) else "🤖"):
            response_container = st.empty()
            
            # Hiệu ứng Spinner đẹp
            with st.spinner("🧠 KTC đang suy nghĩ..."):
                if not groq_client:
                    st.error("❌ Chưa kết nối API Groq.")
                    st.stop()
                    
                stream, sources = get_rag_response(groq_client, st.session_state.vector_db, user_input)
            
            # Streaming text
            full_response = ""
            if isinstance(stream, str): # Trường hợp lỗi
                response_container.error(stream)
                full_response = stream
            else:
                for chunk in stream:
                    if chunk.choices[0].delta.content:
                        content = chunk.choices[0].delta.content
                        full_response += content
                        response_container.markdown(full_response + "▌")
                response_container.markdown(full_response)
            
            # Hiển thị nguồn (Minh chứng KHKT)
            if sources:
                with st.expander("📚 Nguồn tham khảo (Minh chứng)"):
                    for src in sources:
                        st.caption(f"• {src}")
            
            # Lưu lịch sử
            st.session_state.messages.append({"role": "assistant", "content": full_response})

if __name__ == "__main__":
    main()
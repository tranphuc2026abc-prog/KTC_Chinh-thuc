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
    page_title="KTC Assistant - Trợ lý KHKT",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

class AppConfig:
    """Cấu hình trung tâm cho ứng dụng"""
    # Model AI
    LLM_MODEL = 'llama-3.1-8b-instant'
    EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    
    # Đường dẫn files
    PDF_DIR = "PDF_KNOWLEDGE"
    VECTOR_DB_PATH = "faiss_db_index"
    LOGO_PATH = "LOGO.jpg"
    
    # Tham số RAG
    CHUNK_SIZE = 1000 
    CHUNK_OVERLAP = 200
    TOP_K_RETRIEVAL = 4
    
    # Tối ưu performance
    MAX_CONTEXT_LENGTH = 3000

# ==============================================================================
# 2. UI/UX: GIAO DIỆN HIỆN ĐẠI (ĐÃ TỐI ƯU)
# ==============================================================================

def inject_custom_css():
    """CSS tối ưu cho giao diện - Clean & Compact"""
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
        
        /* Font chữ toàn hệ thống */
        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
        }

        /* 1. Header nhỏ gọn, hiện đại hơn */
        .main-header {
            background: linear-gradient(90deg, #4b6cb7 0%, #182848 100%);
            padding: 1rem 1.5rem; /* Thu nhỏ padding */
            border-radius: 12px;
            color: white;
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 1.5rem;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        
        .main-header h1 {
            color: white !important;
            font-weight: 700;
            margin: 0;
            font-size: 1.5rem; /* Font nhỏ lại cho cân đối */
        }
        
        .main-header p {
            margin: 0;
            opacity: 0.8;
            font-size: 0.9rem;
            font-style: italic;
        }

        /* 2. Sidebar tinh tế (Flat Design) */
        [data-testid="stSidebar"] {
            background-color: #f8f9fa;
        }
        
        .sidebar-card {
            background: transparent; /* Bỏ nền trắng nổi */
            padding: 10px 0;
            border-bottom: 1px solid #e9ecef;
            margin-bottom: 15px;
        }
        
        .sidebar-card h4 {
            color: #182848;
            font-size: 1rem;
            font-weight: 700;
            margin-bottom: 10px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        
        .sidebar-text {
            font-size: 0.9rem;
            color: #495057;
            margin-bottom: 5px;
            line-height: 1.4;
        }

        /* 3. Chat Messages */
        .stChatMessage {
            background-color: transparent;
        }
        
        /* User message styling */
        [data-testid="stChatMessageContent"]:has(+ [data-testid="stChatMessageAvatar"]) {
            background-color: #e7f1ff;
            border-radius: 15px 15px 0 15px;
            color: #0f172a;
        }
        
        /* AI message styling */
        [data-testid="stChatMessageContent"]:not(:has(+ [data-testid="stChatMessageAvatar"])) {
            background-color: white;
            border: 1px solid #e2e8f0;
            border-radius: 15px 15px 15px 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.02);
        }

        /* 4. Suggestion Chips (Nút gợi ý) */
        .suggestion-btn {
            border: 1px solid #e2e8f0;
            background: white;
            border-radius: 20px;
            padding: 5px 15px;
            margin: 0 5px;
            cursor: pointer;
            font-size: 0.85rem;
            transition: all 0.2s;
            color: #64748b;
        }
        .suggestion-btn:hover {
            border-color: #4b6cb7;
            color: #4b6cb7;
            background: #f8fafc;
        }

        /* Ẩn bớt decoration của Streamlit */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        
    </style>
    """, unsafe_allow_html=True)

# ==============================================================================
# 3. LOGIC XỬ LÝ (BACKEND)
# ==============================================================================

@st.cache_resource(show_spinner=False)
def load_groq_client():
    try:
        api_key = st.secrets.get("GROQ_API_KEY")
        if not api_key:
            return None
        return Groq(api_key=api_key)
    except Exception:
        return None

@st.cache_resource(show_spinner=False)
def load_embedding_model():
    try:
        with st.spinner("🔄 Khởi động hệ thống AI..."):
            embeddings = HuggingFaceEmbeddings(
                model_name=AppConfig.EMBEDDING_MODEL,
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
        return embeddings
    except Exception:
        return None

@st.cache_data(show_spinner=False, ttl=3600)
def load_and_process_pdfs(pdf_dir):
    docs = []
    if not os.path.exists(pdf_dir):
        os.makedirs(pdf_dir, exist_ok=True)
        return docs
    
    pdf_files = glob.glob(os.path.join(pdf_dir, "*.pdf"))
    if not pdf_files: return docs
    
    # Gom gọn xử lý PDF để giao diện không bị giật
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
        return splitter.split_documents(docs)
    return []

class KnowledgeBase:
    def __init__(self):
        self.embeddings = load_embedding_model()
        self.db_path = AppConfig.VECTOR_DB_PATH

    def get_vector_store(self, force_rebuild=False):
        if not self.embeddings: return None
        if os.path.exists(self.db_path) and not force_rebuild:
            try:
                return FAISS.load_local(self.db_path, self.embeddings, allow_dangerous_deserialization=True)
            except: pass
        
        splits = load_and_process_pdfs(AppConfig.PDF_DIR)
        if splits:
            vector_db = FAISS.from_documents(splits, self.embeddings)
            vector_db.save_local(self.db_path)
            return vector_db
        return None
    
    def rebuild_database(self):
        if os.path.exists(self.db_path):
            import shutil
            shutil.rmtree(self.db_path)
        load_and_process_pdfs.clear()
        return self.get_vector_store(force_rebuild=True)

def get_context(vector_db, query):
    if not vector_db: return "", []
    try:
        results = vector_db.similarity_search_with_score(query, k=AppConfig.TOP_K_RETRIEVAL)
        context_parts, sources = [], []
        total_len = 0
        for doc, score in results:
            if score > 1.6: continue
            src = doc.metadata.get('source', 'Tài liệu')
            page = doc.metadata.get('page', '1')
            content = doc.page_content.replace("\n", " ").strip()
            if total_len + len(content) > AppConfig.MAX_CONTEXT_LENGTH: break
            context_parts.append(f"[{src} - Tr.{page}]: {content}")
            sources.append(f"{src} (Trang {page})")
            total_len += len(content)
        return "\n\n".join(context_parts), list(set(sources))
    except: return "", []

def generate_stream(client, context, question):
    system_prompt = f"""Bạn là KTC Assistant - Trợ lý AI chuyên về Tin học & KHKT.
    
    YÊU CẦU:
    1. Dựa CỐT LÕI vào [CONTEXT] bên dưới.
    2. Trả lời ngắn gọn, súc tích, đi thẳng vào vấn đề.
    3. Định dạng Markdown đẹp mắt (Bold từ khóa chính).
    4. Giọng văn: Thân thiện, khích lệ (như một người thầy/người bạn).
    
    [CONTEXT]:
    {context if context else "Không có thông tin trong tài liệu."}
    """
    try:
        completion = client.chat.completions.create(
            model=AppConfig.LLM_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question}
            ],
            stream=True,
            temperature=0.3,
            max_tokens=1500
        )
        for chunk in completion:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
    except Exception as e:
        yield f"⚠️ Lỗi: {str(e)}"

# ==============================================================================
# 4. HÀM CHÍNH (MAIN APP)
# ==============================================================================

def main():
    if not DEPENDENCIES_OK:
        st.error(f"❌ Thiếu thư viện: {IMPORT_ERROR}")
        st.stop()
    
    inject_custom_css()
    
    # --- SIDEBAR ---
    with st.sidebar:
        if os.path.exists(AppConfig.LOGO_PATH):
            st.image(AppConfig.LOGO_PATH, use_container_width=True)
        else:
            st.markdown("### 🤖 KTC Assistant")

        st.markdown("---")
        
        # Thẻ thông tin gọn gàng hơn
        st.markdown("""
        <div class="sidebar-card">
            <h4>DỰ ÁN KHKT 2024-2025</h4>
            <div class="sidebar-text"><b>🏫 Đơn vị:</b> THCS & THPT Phạm Kiệt</div>
            <div class="sidebar-text"><b>👨‍💻 Tác giả:</b> Tá Tùng & Bảo Chung</div>
            <div class="sidebar-text"><b>🧑‍🏫 GVHD:</b> Thầy Khanh</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Cài đặt đã đổi tên thân thiện
        with st.expander("⚙️ Cấu hình hệ thống"):
            top_k = st.slider("Độ rộng tìm kiếm (Chunks)", 1, 8, AppConfig.TOP_K_RETRIEVAL)
            AppConfig.TOP_K_RETRIEVAL = top_k
            
            if st.button("🔄 Cập nhật dữ liệu mới", use_container_width=True):
                with st.spinner("Đang nạp lại dữ liệu..."):
                    kb = KnowledgeBase()
                    st.session_state.vector_db = kb.rebuild_database()
                st.success("Đã xong!")
                time.sleep(1)
                st.rerun()

        if st.button("🗑️ Xóa hội thoại", use_container_width=True):
            st.session_state.messages = [{"role": "assistant", "content": "Chào bạn! Mình là trợ lý ảo KTC. Bạn cần hỗ trợ gì về dự án hoặc bài học hôm nay?"}]
            st.rerun()

    # --- MAIN CONTENT ---
    
    # Header nhỏ gọn
    st.markdown("""
    <div class="main-header">
        <div>
            <h1>TRỢ LÝ ẢO KTC</h1>
            <p>Hệ thống AI hỗ trợ Nghiên cứu Khoa học & Tin học</p>
        </div>
        <div style="font-size: 2rem;">🎓</div>
    </div>
    """, unsafe_allow_html=True)

    # Init State
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "Chào bạn! Mình là trợ lý ảo KTC. Bạn cần hỗ trợ gì về dự án hoặc bài học hôm nay?"}]
    
    if "vector_db" not in st.session_state:
        kb = KnowledgeBase()
        st.session_state.vector_db = kb.get_vector_store()
    
    groq_client = load_groq_client()

    # Render Chat
    for msg in st.session_state.messages:
        avatar = "🧑‍🎓" if msg["role"] == "user" else "🤖"
        with st.chat_message(msg["role"], avatar=avatar):
            st.markdown(msg["content"])

    # --- GỢI Ý CÂU HỎI (SUGGESTION CHIPS) ---
    # Chỉ hiện khi chưa có nhiều hội thoại
    if len(st.session_state.messages) < 3:
        st.markdown("Running suggestion chips...") # Debug invisible line
        col1, col2, col3 = st.columns(3)
        # Lưu ý: Button trong Streamlit sẽ rerun app. Ta cần xử lý input từ button.
        suggestion = None
        if col1.button("📝 Cấu trúc báo cáo KHKT?", use_container_width=True):
            suggestion = "Hãy cho tôi biết cấu trúc chuẩn của một bài báo cáo KHKT cấp trường."
        if col2.button("🐍 Python cơ bản?", use_container_width=True):
            suggestion = "Tổng hợp các kiến thức cơ bản về Python trong Tin học 11."
        if col3.button("🤖 AI là gì?", use_container_width=True):
            suggestion = "Giải thích khái niệm Trí tuệ nhân tạo đơn giản nhất."
            
        if suggestion:
            # Giả lập việc nhập liệu
            st.session_state.messages.append({"role": "user", "content": suggestion})
            st.rerun()

    # --- CHAT INPUT & XỬ LÝ ---
    # Logic: Ưu tiên lấy từ suggestion nếu có (đã xử lý ở trên qua session state), nếu không thì lấy từ input
    # Nhưng vì button rerun, ta cần check message cuối cùng xem có phải user không để trigger trả lời
    
    prompt = st.chat_input("Nhập câu hỏi của bạn...")
    
    # Biến để trigger AI trả lời
    process_response = False
    
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        process_response = True
    elif len(st.session_state.messages) > 0 and st.session_state.messages[-1]["role"] == "user":
        # Trường hợp vừa click button gợi ý, app rerun, message cuối là user -> cần trả lời
        process_response = True
        prompt = st.session_state.messages[-1]["content"]

    if process_response:
        if not prompt: prompt = st.session_state.messages[-1]["content"]
        
        # Chỉ hiển thị prompt nếu chưa hiển thị (tránh duplicate khi rerun)
        # (Streamlit chat input tự hiển thị, nhưng button thì không -> đã append vào session)
        
        with st.chat_message("assistant", avatar="🤖"):
            response_holder = st.empty()
            
            # Status đẹp hơn
            with st.status("🔍 KTC đang tra cứu dữ liệu...", expanded=True) as status:
                st.write("Đang đọc tài liệu PDF...")
                context, sources = get_context(st.session_state.vector_db, prompt)
                st.write("Đang tổng hợp câu trả lời...")
                status.update(label="✅ Đã tìm thấy thông tin!", state="complete", expanded=False)
            
            # Stream response
            full_res = ""
            if groq_client:
                for chunk in generate_stream(groq_client, context, prompt):
                    full_res += chunk
                    response_holder.markdown(full_res + "▌")
                response_holder.markdown(full_res)
            else:
                st.error("Chưa kết nối được Groq API.")

            # Sources Citation
            if sources:
                with st.expander("📚 Nguồn tài liệu tham khảo (Minh chứng)"):
                    for src in sources:
                        st.markdown(f"- *{src}*")
            
            # Lưu lại câu trả lời AI (nếu chưa lưu)
            if st.session_state.messages[-1]["role"] != "assistant":
                st.session_state.messages.append({"role": "assistant", "content": full_res})

if __name__ == "__main__":
    main()
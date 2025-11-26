import streamlit as st
from groq import Groq
import os
import glob
from pypdf import PdfReader

# --- CÁC THƯ VIỆN RAG (LANGCHAIN) ---
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

# --- 1. CẤU HÌNH TRANG ---
st.set_page_config(
    page_title="Chatbot KTC - Trợ lý Tin học",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CÁC HẰNG SỐ ---
MODEL_NAME = 'llama-3.1-8b-instant'
PDF_DIR = "./PDF_KNOWLEDGE"
LOGO_PATH = "LOGO.jpg" 

# --- 2. CSS TÙY CHỈNH GIAO DIỆN ---
st.markdown("""
<style>
    /* 1. Nền chính */
    .stApp {background-color: #f8f9fa;}
    
    /* 2. Sidebar */
    [data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #e0e0e0;
    }
    
    /* 3. Box Thông tin tác giả */
    .author-box {
        background-color: #f0f8ff;
        border: 1px solid #bae6fd;
        border-radius: 10px;
        padding: 15px;
        font-size: 0.9rem;
        margin-top: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        color: #0f172a;
    }
    .author-header {
        font-weight: bold;
        color: #0284c7;
        margin-bottom: 5px;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-top: 10px;
    }
    .author-header:first-child { margin-top: 0; }
    
    .author-content {
        margin-bottom: 8px;
        color: #334155;
    }
    .author-list {
        margin: 0;
        padding-left: 20px;
        color: #334155;
        font-weight: 500;
    }

    /* 4. Tiêu đề Gradient */
    .gradient-text {
        background: linear-gradient(90deg, #0f4c81, #1cb5e0);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        font-size: 2.5rem;
        padding-bottom: 1rem;
        text-align: center;
        margin-bottom: 0;
    }
    
    /* 5. Bong bóng chat */
    .stChatMessage {
        background-color: transparent; 
        border: none;
        padding: 10px;
    }
    div[data-testid="stChatMessage"]:nth-child(even) { 
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 0px 15px 15px 15px;
        box-shadow: 0 1px 2px rgba(0,0,0,0.05);
    }
    div[data-testid="stChatMessage"]:nth-child(odd) {
        background-color: #e0f2fe;
        border-radius: 15px 0px 15px 15px;
        border: none;
    }

    /* 6. Button */
    .stButton>button {
        border-radius: 8px;
        background-color: #0284c7;
        color: white;
        border: none;
        font-weight: 600;
    }
    
    /* 7. Footer */
    .footer-note {
        text-align: center;
        font-size: 0.75rem;
        color: #94a3b8;
        margin-top: 30px;
        border-top: 1px dashed #cbd5e1;
        padding-top: 10px;
    }
</style>
""", unsafe_allow_html=True)

# --- 3. XỬ LÝ KẾT NỐI ---
try:
    api_key = st.secrets["GROQ_API_KEY"]
except (KeyError, FileNotFoundError):
    st.error("❌ Lỗi: Chưa cấu hình GROQ_API_KEY")
    st.stop()

client = Groq(api_key=api_key)

@st.cache_resource(show_spinner=False)
def initialize_vector_db():
    if not os.path.exists(PDF_DIR):
        os.makedirs(PDF_DIR)
        return None
    
    pdf_files = glob.glob(os.path.join(PDF_DIR, "*.pdf"))
    if not pdf_files:
        return None

    with st.spinner('🔄 Đang khởi tạo bộ não...'):
        documents = []
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

        for pdf_path in pdf_files:
            try:
                reader = PdfReader(pdf_path)
                file_name = os.path.basename(pdf_path)
                for i, page in enumerate(reader.pages):
                    text = page.extract_text()
                    if text:
                        chunks = text_splitter.split_text(text)
                        for chunk in chunks:
                            documents.append(Document(page_content=chunk, metadata={"source": file_name, "page": i + 1}))
            except Exception: pass

        if not documents: return None
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        return FAISS.from_documents(documents, embeddings)

# --- KHỞI TẠO STATE ---
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Chào bạn! Mình là Chatbot KTC 🤖. Mình có thể giúp gì cho bạn về môn Tin học hôm nay?"}]

if "vector_db" not in st.session_state:
    st.session_state.vector_db = initialize_vector_db()

# --- 4. SIDEBAR (ĐÃ SỬA LỖI HTML) ---
with st.sidebar:
    if os.path.exists(LOGO_PATH):
        st.image(LOGO_PATH, use_container_width=True)
    
    st.markdown("""
        <div style='text-align: center; margin-top: 10px;'>
            <h3 style='color: #0f4c81; margin: 0;'>TRỢ LÝ KTC</h3>
            <p style='font-size: 0.8rem; color: #64748b;'>Knowledge & Technology Chatbot</p>
        </div>
        <hr style="margin: 15px 0;">
    """, unsafe_allow_html=True)
    
    # Trạng thái
    if st.session_state.vector_db:
        st.markdown("💾 Dữ liệu SGK: <span style='color:green; font-weight:bold'>● Đã kết nối</span>", unsafe_allow_html=True)
    else:
        st.markdown("💾 Dữ liệu SGK: <span style='color:red; font-weight:bold'>● Chưa nạp</span>", unsafe_allow_html=True)
        
    # --- PHẦN SỬA LỖI HIỂN THỊ HTML ---
    # Tôi đã đưa chuỗi HTML về sát lề trái để tránh lỗi thụt đầu dòng (Indentation Error)
    html_info = """
<div class="author-box">
    <div class="author-header">🏫 Sản phẩm KHKT</div>
    <div class="author-content">Năm học 2025 - 2026</div>
    <div class="author-header">👨‍🏫 GV Hướng Dẫn</div>
    <div class="author-content">Thầy Nguyễn Thế Khanh</div>
    <div class="author-header">🧑‍🎓 Nhóm tác giả</div>
    <ul class="author-list">
        <li>Bùi Tá Tùng</li>
        <li>Cao Sỹ Bảo Chung</li>
    </ul>
</div>
"""
    st.markdown(html_info, unsafe_allow_html=True)
    
    st.markdown("<div style='height: 20px'></div>", unsafe_allow_html=True)
    if st.button("🗑️ Làm mới hội thoại", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# --- 5. GIAO DIỆN CHÍNH ---
col1, col2, col3 = st.columns([1, 8, 1])

with col2:
    st.markdown('<h1 class="gradient-text">CHATBOT HỖ TRỢ HỌC TẬP KTC</h1>', unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #64748b; font-style: italic; margin-bottom: 30px;'>🚀 Ứng dụng AI hỗ trợ tra cứu kiến thức Tin học chương trình GDPT 2018</p>", unsafe_allow_html=True)
    
    for message in st.session_state.messages:
        avatar = "🧑‍🎓" if message["role"] == "user" else "🤖"
        with st.chat_message(message["role"], avatar=avatar):
            st.markdown(message["content"], unsafe_allow_html=True)

    prompt = st.chat_input("Nhập câu hỏi của bạn tại đây...")

    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user", avatar="🧑‍🎓"):
            st.markdown(prompt)

        context_text = ""
        sources_list = []
        if st.session_state.vector_db:
            results = st.session_state.vector_db.similarity_search(prompt, k=3)
            for doc in results:
                context_text += f"\n---\nNội dung: {doc.page_content}\nNguồn: {doc.metadata['source']} (Trang {doc.metadata['page']})"
                sources_list.append(f"{doc.metadata['source']} - Tr. {doc.metadata['page']}")

        SYSTEM_PROMPT = """Bạn là "Chatbot KTC", trợ lý ảo chuyên gia Tin học. Trả lời dựa trên SGK. Luôn trích dẫn nguồn."""
        final_prompt = f"{SYSTEM_PROMPT}\n--- BỐI CẢNH SGK ---\n{context_text}\n--- CÂU HỎI ---\n{prompt}"

        with st.chat_message("assistant", avatar="🤖"):
            placeholder = st.empty()
            full_response = ""
            try:
                chat_completion = client.chat.completions.create(
                    messages=[{"role": "system", "content": final_prompt}, {"role": "user", "content": prompt}],
                    model=MODEL_NAME, stream=True, temperature=0.3
                )

                for chunk in chat_completion:
                    if chunk.choices[0].delta.content:
                        full_response += chunk.choices[0].delta.content
                        placeholder.markdown(full_response + "▌")
                
                if sources_list:
                    unique_sources = list(set(sources_list))
                    citation_html = "<div style='margin-top:10px; font-size: 0.85em; color: #666; border-top: 1px solid #ddd; padding-top: 5px;'>📚 <b>Nguồn tham khảo:</b><br>" + "<br>".join([f"- <i>{s}</i>" for s in unique_sources]) + "</div>"
                    full_response += "\n"
                    placeholder.markdown(full_response + "\n\n" + citation_html, unsafe_allow_html=True)
                else:
                    placeholder.markdown(full_response)

                st.session_state.messages.append({"role": "assistant", "content": full_response})
            except Exception as e:
                st.error(f"Lỗi: {e}")

    st.markdown('<div class="footer-note">⚠️ Lưu ý: AI có thể mắc lỗi. Vui lòng kiểm tra lại thông tin quan trọng với SGK.</div>', unsafe_allow_html=True)
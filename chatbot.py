import streamlit as st
from groq import Groq
import os
import glob
import time
from pypdf import PdfReader

# --- LIBRARY AI & RAG ---
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

# --- 1. CẤU HÌNH HỆ THỐNG & HẰNG SỐ ---
st.set_page_config(
    page_title="KTC Assistant - Trợ lý Tin học 2025",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

CONSTANTS = {
    "MODEL_NAME": 'llama-3.1-8b-instant',
    "PDF_DIR": "./PDF_KNOWLEDGE",
    "VECTOR_STORE_PATH": "./faiss_db_index", # Nơi lưu bộ não vĩnh viễn
    "LOGO_PATH": "LOGO.jpg",
    "EMBEDDING_MODEL": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", # Tốt hơn cho tiếng Việt
    "CHUNK_SIZE": 800, # Giảm chunk size để nội dung cô đọng hơn
    "CHUNK_OVERLAP": 150
}

# --- 2. CLASS XỬ LÝ RAG (OOP STRUCTURE) ---
class KnowledgeBase:
    """Class quản lý việc đọc, xử lý và truy xuất dữ liệu kiến thức."""
    
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(model_name=CONSTANTS["EMBEDDING_MODEL"])

    def load_documents(self):
        """Đọc PDF từ thư mục."""
        if not os.path.exists(CONSTANTS["PDF_DIR"]):
            os.makedirs(CONSTANTS["PDF_DIR"])
            return []
        
        pdf_files = glob.glob(os.path.join(CONSTANTS["PDF_DIR"], "*.pdf"))
        documents = []
        
        for pdf_path in pdf_files:
            try:
                reader = PdfReader(pdf_path)
                file_name = os.path.basename(pdf_path)
                for i, page in enumerate(reader.pages):
                    text = page.extract_text()
                    if text:
                        documents.append(Document(
                            page_content=text, 
                            metadata={"source": file_name, "page": i + 1}
                        ))
            except Exception as e:
                st.warning(f"Không thể đọc file {pdf_path}: {e}")
        return documents

    def build_or_load_vector_db(self, force_rebuild=False):
        """
        Cơ chế thông minh:
        1. Kiểm tra xem đã có Database lưu trên ổ cứng chưa.
        2. Nếu có -> Load lên (mất 1 giây).
        3. Nếu chưa hoặc user ép buộc -> Xây dựng lại (mất nhiều thời gian).
        """
        if os.path.exists(CONSTANTS["VECTOR_STORE_PATH"]) and not force_rebuild:
            try:
                # Load từ ổ cứng
                return FAISS.load_local(
                    CONSTANTS["VECTOR_STORE_PATH"], 
                    self.embeddings, 
                    allow_dangerous_deserialization=True
                )
            except Exception:
                pass # Nếu lỗi file thì build lại

        # Nếu chưa có, bắt đầu build
        documents = self.load_documents()
        if not documents:
            return None

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CONSTANTS["CHUNK_SIZE"], 
            chunk_overlap=CONSTANTS["CHUNK_OVERLAP"]
        )
        splits = text_splitter.split_documents(documents)
        
        if not splits: return None

        # Tạo Vector Store
        vector_db = FAISS.from_documents(splits, self.embeddings)
        # Lưu xuống ổ cứng để lần sau dùng
        vector_db.save_local(CONSTANTS["VECTOR_STORE_PATH"])
        return vector_db

# --- 3. GIAO DIỆN & LOGIC CHÍNH ---

# CSS Tinh chỉnh (Giữ nguyên style đẹp của thầy, tối ưu thêm font)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;700&display=swap');
    html, body, [class*="css"] { font-family: 'Roboto', sans-serif; }
    .stApp {background-color: #f8f9fa;}
    
    /* Sidebar Pro */
    [data-testid="stSidebar"] { background-color: #ffffff; border-right: 1px solid #e0e0e0; }
    
    /* Chat Bubble Pro */
    div[data-testid="stChatMessage"] { padding: 1rem; border-radius: 10px; }
    div[data-testid="stChatMessage"]:nth-child(odd) { background-color: #f0f9ff; border: 1px solid #bae6fd; }
    div[data-testid="stChatMessage"]:nth-child(even) { background-color: #ffffff; border: 1px solid #e2e8f0; box-shadow: 0 2px 5px rgba(0,0,0,0.05); }
    
    .gradient-text {
        background: linear-gradient(90deg, #0052cc, #00c6ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        font-size: 2.2rem;
        text-align: center;
        padding: 10px 0;
    }
    
    .source-box {
        font-size: 0.8rem; color: #555; background: #f1f1f1; 
        padding: 8px; border-radius: 5px; margin-top: 5px; border-left: 3px solid #0284c7;
    }
</style>
""", unsafe_allow_html=True)

# Khởi tạo kết nối Groq
try:
    api_key = st.secrets["GROQ_API_KEY"]
    client = Groq(api_key=api_key)
except Exception:
    st.error("⚠️ Lỗi hệ thống: Chưa cấu hình API Key. Vui lòng kiểm tra secrets.toml")
    st.stop()

# Khởi tạo Session State
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Xin chào! Tôi là **KTC AI**. Hãy hỏi tôi bất cứ điều gì về Tin học trong SGK."}
    ]

if "rag_engine" not in st.session_state:
    st.session_state.rag_engine = KnowledgeBase()

# Load Vector DB (Chỉ load 1 lần đầu, cực nhanh)
if "vector_db" not in st.session_state:
    with st.spinner('🔄 Đang kích hoạt hệ thống tri thức số...'):
        st.session_state.vector_db = st.session_state.rag_engine.build_or_load_vector_db()

# --- SIDEBAR ---
with st.sidebar:
    if os.path.exists(CONSTANTS["LOGO_PATH"]):
        st.image(CONSTANTS["LOGO_PATH"], use_container_width=True)
    
    st.title("⚙️ Control Panel")
    
    # Trạng thái hệ thống
    status_color = "green" if st.session_state.vector_db else "red"
    status_text = "Đã nạp kiến thức" if st.session_state.vector_db else "Chưa có dữ liệu"
    st.markdown(f"**Trạng thái:** <span style='color:{status_color}'>● {status_text}</span>", unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Admin Controls
    if st.button("🔄 Cập nhật lại Dữ liệu (Re-build)", help="Nhấn khi bạn mới bỏ thêm file PDF vào"):
        with st.spinner("Đang đọc lại toàn bộ tài liệu... (Sẽ mất thời gian)"):
            st.session_state.vector_db = st.session_state.rag_engine.build_or_load_vector_db(force_rebuild=True)
        st.success("Đã cập nhật dữ liệu mới!")
        time.sleep(1)
        st.rerun()

    if st.button("🗑️ Xóa lịch sử Chat"):
        st.session_state.messages = []
        st.rerun()

    # Author Info (Giữ nguyên format của thầy)
    st.markdown("""
    <div style="background:#f8f9fa; padding:15px; border-radius:8px; border:1px dashed #ccc; margin-top:20px;">
        <div style="font-weight:bold; color:#0052cc; font-size:0.9rem;">🚀 DỰ ÁN KHKT 2025-2026</div>
        <div style="font-size:0.85rem; margin-top:5px;">GVHD: <b>Thầy Nguyễn Thế Khanh</b></div>
        <div style="font-size:0.85rem;">Học sinh: <b>Bùi Tá Tùng - Cao Sỹ Bảo Chung</b></div>
    </div>
    """, unsafe_allow_html=True)

# --- MAIN CHAT INTERFACE ---
col1, col2, col3 = st.columns([1, 10, 1])

with col2:
    st.markdown('<h1 class="gradient-text">TRỢ LÝ ẢO TIN HỌC KTC</h1>', unsafe_allow_html=True)

    # Hiển thị lịch sử chat
    for message in st.session_state.messages:
        avatar = "🧑‍🎓" if message["role"] == "user" else "🤖"
        with st.chat_message(message["role"], avatar=avatar):
            st.markdown(message["content"], unsafe_allow_html=True)

    # Xử lý khi người dùng nhập liệu
    if prompt := st.chat_input("Bạn muốn tìm hiểu gì về Tin học?"):
        # 1. Hiển thị câu hỏi user
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user", avatar="🧑‍🎓"):
            st.markdown(prompt)

        # 2. Xử lý RAG
        with st.chat_message("assistant", avatar="🤖"):
            message_placeholder = st.empty()
            full_response = ""
            
            # Tìm kiếm context
            context_text = ""
            sources = []
            
            if st.session_state.vector_db:
                # Similarity Search
                results = st.session_state.vector_db.similarity_search(prompt, k=3)
                for doc in results:
                    context_text += f"\n[Nội dung trích xuất]: {doc.page_content}\n[Nguồn]: {doc.metadata.get('source')} - Trang {doc.metadata.get('page')}"
                    sources.append(f"{doc.metadata.get('source')} (Tr. {doc.metadata.get('page')})")
            
            # Prompt Engineering Cao cấp (Instruction Tuning)
            SYSTEM_PROMPT = f"""
            Bạn là trợ lý ảo KTC, chuyên gia về môn Tin học GDPT 2018.
            NHIỆM VỤ: Trả lời câu hỏi dựa trên thông tin được cung cấp dưới đây.
            
            YÊU CẦU:
            1. Giọng văn thân thiện, sư phạm, dễ hiểu cho học sinh.
            2. Chỉ sử dụng thông tin trong phần [THÔNG TIN TÀI LIỆU] để trả lời. Nếu không có thông tin, hãy nói "SGK hiện chưa đề cập vấn đề này".
            3. Trình bày đẹp: Sử dụng Markdown (in đậm, gạch đầu dòng).
            
            [THÔNG TIN TÀI LIỆU]:
            {context_text}
            """
            
            # Streaming Response
            try:
                stream = client.chat.completions.create(
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        # Gửi kèm vài tin nhắn cũ để AI hiểu ngữ cảnh (Context Window)
                        *st.session_state.messages[-4:], 
                    ],
                    model=CONSTANTS["MODEL_NAME"],
                    stream=True,
                    temperature=0.3, # Giữ nhiệt độ thấp để câu trả lời chính xác theo sách
                    max_tokens=1024
                )

                for chunk in stream:
                    if chunk.choices[0].delta.content:
                        content = chunk.choices[0].delta.content
                        full_response += content
                        message_placeholder.markdown(full_response + "▌")
                
                # Hiển thị nguồn tài liệu (Trích dẫn khoa học)
                if sources:
                    unique_sources = list(set(sources))
                    source_html = "<div class='source-box'>📚 <b>Nguồn tham khảo xác thực:</b><br>" + "<br>".join([f"• {s}" for s in unique_sources]) + "</div>"
                    final_content = full_response + "\n\n" + source_html
                    message_placeholder.markdown(final_content, unsafe_allow_html=True)
                    st.session_state.messages.append({"role": "assistant", "content": final_content})
                else:
                    message_placeholder.markdown(full_response)
                    st.session_state.messages.append({"role": "assistant", "content": full_response})

            except Exception as e:
                st.error(f"Đã xảy ra lỗi kết nối AI: {str(e)}")
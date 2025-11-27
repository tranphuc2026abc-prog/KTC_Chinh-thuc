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
# Ngưỡng điểm tương đồng (Thấp hơn là tốt hơn với FAISS L2 distance, nhưng với cosine similarity thì ngược lại). 
# Ở đây ta dùng FAISS mặc định (L2), khoảng cách càng nhỏ càng giống.
SIMILARITY_THRESHOLD = 1.2  

# --- 2. CSS TÙY CHỈNH GIAO DIỆN ---
st.markdown("""
<style>
    .stApp {background-color: #f8f9fa;}
    [data-testid="stSidebar"] {background-color: #ffffff; border-right: 1px solid #e0e0e0;}
    
    /* Box tác giả */
    .author-box {
        background-color: #f0f8ff; border: 1px solid #bae6fd; border-radius: 10px;
        padding: 15px; font-size: 0.9rem; margin-top: 15px; box-shadow: 0 4px 6px rgba(0,0,0,0.05);
    }
    .author-header { font-weight: bold; color: #0284c7; margin-bottom: 5px; font-size: 0.85rem; text-transform: uppercase; margin-top: 10px;}
    .author-header:first-child { margin-top: 0; }
    .author-content { margin-bottom: 8px; color: #334155; }
    .author-list { margin: 0; padding-left: 20px; color: #334155; font-weight: 500; }

    /* Tiêu đề & Chat */
    .gradient-text {
        background: linear-gradient(90deg, #0f4c81, #1cb5e0); -webkit-background-clip: text;
        -webkit-text-fill-color: transparent; font-weight: 800; font-size: 2.5rem;
        text-align: center; margin-bottom: 0;
    }
    div[data-testid="stChatMessage"] { background-color: transparent; border: none; padding: 10px; }
    div[data-testid="stChatMessage"][data-testid="user"] { background-color: #e0f2fe; border-radius: 15px 0px 15px 15px; } 
    div[data-testid="stChatMessage"][data-testid="assistant"] { background-color: #ffffff; border: 1px solid #e2e8f0; border-radius: 0px 15px 15px 15px; box-shadow: 0 1px 2px rgba(0,0,0,0.05); }
    
    .stButton>button { border-radius: 8px; background-color: #0284c7; color: white; border: none; font-weight: 600; }
    .footer-note { text-align: center; font-size: 0.75rem; color: #94a3b8; margin-top: 30px; border-top: 1px dashed #cbd5e1; padding-top: 10px; }
    
    /* Expander cho nguồn */
    .streamlit-expanderHeader {font-size: 0.8rem; color: #666;}
</style>
""", unsafe_allow_html=True)

# --- 3. XỬ LÝ KẾT NỐI ---
try:
    # Ưu tiên lấy từ secrets, nếu không có thì thử biến môi trường (cho debug local)
    api_key = st.secrets.get("GROQ_API_KEY", os.getenv("GROQ_API_KEY"))
    if not api_key:
        raise KeyError("Missing GROQ_API_KEY")
except Exception:
    st.error("❌ Lỗi: Chưa cấu hình GROQ_API_KEY trong .streamlit/secrets.toml")
    st.stop()

client = Groq(api_key=api_key)

# Cache tài nguyên nặng (Embeddings & Vector DB) để không load lại khi reload trang
@st.cache_resource(show_spinner=False)
def initialize_vector_db():
    if not os.path.exists(PDF_DIR):
        os.makedirs(PDF_DIR)
        return None
    
    pdf_files = glob.glob(os.path.join(PDF_DIR, "*.pdf"))
    if not pdf_files:
        return None

    with st.spinner('🔄 Đang nạp dữ liệu tri thức (Mất khoảng 1-2 phút lần đầu)...'):
        documents = []
        # Chunk size quan trọng: 1000 ký tự đủ để chứa 1 đoạn thông tin trọn vẹn
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
        
        # Model Multilingual cực quan trọng cho dự án Anh-Việt
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
        return FAISS.from_documents(documents, embeddings)

# --- KHỞI TẠO STATE ---
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Chào bạn! Mình là Chatbot KTC 🤖. Mình đã học hết tài liệu thầy Khanh giao rồi, bạn hỏi gì đi!"}]

if "vector_db" not in st.session_state:
    st.session_state.vector_db = initialize_vector_db()

# --- 4. SIDEBAR ---
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
    
    if st.session_state.vector_db:
        st.markdown(f"💾 Trạng thái: <span style='color:green; font-weight:bold'>● Sẵn sàng ({st.session_state.vector_db.index.ntotal} vectors)</span>", unsafe_allow_html=True)
    else:
        st.markdown("💾 Trạng thái: <span style='color:red; font-weight:bold'>● Chưa có dữ liệu</span>", unsafe_allow_html=True)
        st.info("💡 Hãy bỏ file PDF vào thư mục `PDF_KNOWLEDGE` và khởi động lại.")
        
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
    if st.button("🗑️ Xóa hội thoại", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# --- 5. GIAO DIỆN CHÍNH ---
col1, col2, col3 = st.columns([1, 8, 1])

with col2:
    st.markdown('<h1 class="gradient-text">CHATBOT HỖ TRỢ HỌC TẬP KTC</h1>', unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #64748b; font-style: italic; margin-bottom: 30px;'>🚀 Hỏi đáp thông minh dựa trên tài liệu Tin học (Anh/Việt)</p>", unsafe_allow_html=True)
    
    # Hiển thị lịch sử chat
    for message in st.session_state.messages:
        avatar = "🧑‍🎓" if message["role"] == "user" else "🤖"
        with st.chat_message(message["role"], avatar=avatar):
            st.markdown(message["content"], unsafe_allow_html=True)

    # Xử lý input người dùng
    prompt = st.chat_input("Nhập câu hỏi của bạn tại đây...")

    if prompt:
        # 1. Hiển thị câu hỏi người dùng
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user", avatar="🧑‍🎓"):
            st.markdown(prompt)

        # 2. Xử lý RAG
        context_text = ""
        sources_list = []
        relevant_docs = []

        if st.session_state.vector_db:
            # SỬ DỤNG similarity_search_with_score ĐỂ LỌC NHIỄU
            # k=4: lấy 4 đoạn; score càng thấp càng giống (với L2 distance của FAISS)
            results_with_score = st.session_state.vector_db.similarity_search_with_score(prompt, k=4)
            
            for doc, score in results_with_score:
                # Nếu score > ngưỡng (ví dụ 1.5) nghĩa là quá khác biệt -> Bỏ qua
                # Lưu ý: Ngưỡng này cần tinh chỉnh tùy vào dữ liệu thực tế
                if score < SIMILARITY_THRESHOLD: 
                    context_text += f"\n---\n[Nguồn: {doc.metadata['source']} - Tr.{doc.metadata['page']}]\nNội dung: {doc.page_content}"
                    sources_list.append(f"{doc.metadata['source']} (Trang {doc.metadata['page']})")
                    relevant_docs.append(doc)
        
        # 3. Xây dựng Prompt
        if not context_text:
            context_text = "Không tìm thấy thông tin phù hợp trong tài liệu được cung cấp."
            system_instruction = "Bạn là trợ lý ảo. Hiện tại bạn không tìm thấy thông tin trong tài liệu. Hãy trả lời lịch sự rằng bạn chưa có thông tin về vấn đề này trong bộ dữ liệu, nhưng có thể trả lời dựa trên kiến thức xã hội (nếu biết)."
        else:
            system_instruction = """
            Bạn là "Chatbot KTC", trợ lý chuyên gia Tin học của thầy Khanh.
            NHIỆM VỤ:
            1. Dựa CHÍNH XÁC vào "BỐI CẢNH" bên dưới để trả lời.
            2. Nếu bối cảnh tiếng Anh, HÃY DỊCH VÀ TRẢ LỜI BẰNG TIẾNG VIỆT tự nhiên.
            3. Trình bày câu trả lời rõ ràng, dùng Markdown (in đậm, gạch đầu dòng) để dễ đọc.
            4. Cuối câu trả lời, hãy trích dẫn ngắn gọn nguồn tài liệu.
            """

        final_prompt = f"{system_instruction}\n\n--- BỐI CẢNH TÀI LIỆU ---\n{context_text}\n\n--- CÂU HỎI ---\n{prompt}"

        # 4. Gọi API Groq (Streaming)
        with st.chat_message("assistant", avatar="🤖"):
            placeholder = st.empty()
            full_response = ""
            try:
                # Gửi kèm lịch sử chat ngắn (nếu cần) hoặc chỉ gửi prompt hiện tại để tiết kiệm token
                # Ở đây ta gửi prompt hiện tại kèm context RAG là tối ưu nhất cho KHKT
                chat_completion = client.chat.completions.create(
                    messages=[
                        {"role": "system", "content": final_prompt},
                        {"role": "user", "content": prompt}
                    ],
                    model=MODEL_NAME, 
                    stream=True, 
                    temperature=0.3
                )

                for chunk in chat_completion:
                    if chunk.choices[0].delta.content:
                        content = chunk.choices[0].delta.content
                        full_response += content
                        placeholder.markdown(full_response + "▌")
                
                placeholder.markdown(full_response)
                
                # HIỂN THỊ NGUỒN MINH BẠCH (Điểm cộng KHKT)
                if relevant_docs:
                    with st.expander("📚 Xem trích dẫn tài liệu gốc (Dành cho Giám khảo/Kiểm chứng)"):
                        for doc in relevant_docs:
                            st.markdown(f"**📄 {doc.metadata['source']} - Trang {doc.metadata['page']}**")
                            st.caption(doc.page_content[:300] + "...") # Chỉ hiện 300 ký tự đầu
                            st.divider()

                st.session_state.messages.append({"role": "assistant", "content": full_response})
                
            except Exception as e:
                st.error(f"⚠️ Có lỗi kết nối AI: {e}")

    st.markdown('<div class="footer-note">⚠️ Dự án KHKT trường THCS & THPT Phạm Kiệt. AI có thể mắc lỗi, hãy kiểm tra tài liệu gốc.</div>', unsafe_allow_html=True)
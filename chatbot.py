import streamlit as st
from groq import Groq
import os
import glob
from pypdf import PdfReader

# --- CÁC THƯ VIỆN RAG ---
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
SIMILARITY_THRESHOLD = 1.5  # Giữ mức trung bình
TOP_K_RETRIEVAL = 6

# --- 2. CSS ---
st.markdown("""
<style>
    .stApp {background-color: #f8f9fa;}
    [data-testid="stSidebar"] {background-color: #ffffff; border-right: 1px solid #e0e0e0;}
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
</style>
""", unsafe_allow_html=True)

# --- 3. XỬ LÝ KẾT NỐI ---
try:
    api_key = st.secrets.get("GROQ_API_KEY", os.getenv("GROQ_API_KEY"))
    if not api_key: raise KeyError("Missing GROQ_API_KEY")
except Exception:
    st.error("❌ Lỗi: Chưa cấu hình GROQ_API_KEY")
    st.stop()

client = Groq(api_key=api_key)

# Hàm nạp dữ liệu (Không dùng cache_resource ở đây để có thể reload thủ công)
def load_data():
    if not os.path.exists(PDF_DIR):
        os.makedirs(PDF_DIR)
        return None
    
    pdf_files = glob.glob(os.path.join(PDF_DIR, "*.pdf"))
    if not pdf_files:
        return None

    documents = []
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

    # Thêm thanh tiến trình để thầy trò dễ theo dõi
    progress_text = "Đang nạp dữ liệu... Vui lòng đợi."
    my_bar = st.progress(0, text=progress_text)
    
    total_files = len(pdf_files)
    for idx, pdf_path in enumerate(pdf_files):
        try:
            reader = PdfReader(pdf_path)
            file_name = os.path.basename(pdf_path)
            for i, page in enumerate(reader.pages):
                text = page.extract_text()
                if text:
                    # Xử lý text thô một chút để tránh lỗi font cơ bản
                    text = text.replace('\n', ' ').strip()
                    chunks = text_splitter.split_text(text)
                    for chunk in chunks:
                        documents.append(Document(page_content=chunk, metadata={"source": file_name, "page": i + 1}))
        except Exception: pass
        my_bar.progress((idx + 1) / total_files, text=f"Đang đọc file: {file_name}")

    my_bar.empty()
    
    if not documents: return None
    
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    return FAISS.from_documents(documents, embeddings)

# --- KHỞI TẠO STATE ---
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Chào bạn! Chatbot KTC đã sẵn sàng. Hãy hỏi về HTML, AI, Python... nhé!"}]

# Kiểm tra xem Vector DB đã có chưa, nếu chưa thì load
if "vector_db" not in st.session_state:
    with st.spinner("🔄 Đang khởi tạo bộ não lần đầu..."):
        st.session_state.vector_db = load_data()

# --- 4. SIDEBAR ---
with st.sidebar:
    if os.path.exists(LOGO_PATH):
        st.image(LOGO_PATH, use_container_width=True)
    
    st.markdown("<h3 style='text-align: center; color: #0f4c81;'>TRỢ LÝ KTC</h3>", unsafe_allow_html=True)
    
    # Hiển thị số lượng Vectors (Chỉ số quan trọng để biết sách đã vào chưa)
    if st.session_state.vector_db:
        num_vectors = st.session_state.vector_db.index.ntotal
        st.success(f"🟢 Đã học: {num_vectors} đoạn kiến thức")
    else:
        st.error("🔴 Chưa có dữ liệu")

    st.markdown("---")
    
    # --- CÔNG CỤ QUẢN TRỊ (Dành cho thầy Khanh check lỗi) ---
    st.markdown("**🔧 Công cụ quản trị**")
    
    # Nút 1: Ép học lại (Quan trọng khi thêm sách mới)
    if st.button("🔄 Nạp lại dữ liệu gốc (Force Reload)", use_container_width=True):
        st.session_state.vector_db = None # Xóa não cũ
        st.rerun() # Chạy lại trang để hàm load_data chạy lại
        
    # Nút 2: Xóa chat
    if st.button("🧹 Làm mới hội thoại", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    # --- DEBUG: KIỂM TRA XEM MÁY ĐỌC ĐƯỢC GÌ ---
    with st.expander("🕵️ Soi dữ liệu (Debug)"):
        st.write("Dán câu hỏi vào đây để xem máy tìm thấy đoạn nào:")
        debug_query = st.text_input("Câu hỏi test", "HTML là gì")
        if st.button("Kiểm tra tìm kiếm") and st.session_state.vector_db:
            docs = st.session_state.vector_db.similarity_search_with_score(debug_query, k=4)
            for doc, score in docs:
                st.write(f"**Score:** {score:.3f}")
                st.info(doc.page_content)
                st.write("---")

    st.markdown("<div style='margin-top: 20px; font-size: 0.8rem; color: grey'>Sản phẩm KHKT - THCS & THPT Phạm Kiệt</div>", unsafe_allow_html=True)

# --- 5. GIAO DIỆN CHÍNH ---
col1, col2, col3 = st.columns([1, 8, 1])

with col2:
    st.markdown('<h1 class="gradient-text">CHATBOT HỖ TRỢ HỌC TẬP KTC</h1>', unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #64748b; font-style: italic;'>🚀 Hỏi đáp thông minh dựa trên tài liệu Tin học (Anh/Việt)</p>", unsafe_allow_html=True)
    
    for message in st.session_state.messages:
        avatar = "🧑‍🎓" if message["role"] == "user" else "🤖"
        with st.chat_message(message["role"], avatar=avatar):
            st.markdown(message["content"], unsafe_allow_html=True)

    prompt = st.chat_input("Nhập câu hỏi của bạn tại đây...")

    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user", avatar="🧑‍🎓"):
            st.markdown(prompt)

        # --- LOGIC RAG ---
        context_text = ""
        relevant_docs = []

        if st.session_state.vector_db:
            results_with_score = st.session_state.vector_db.similarity_search_with_score(prompt, k=TOP_K_RETRIEVAL)
            for doc, score in results_with_score:
                if score < SIMILARITY_THRESHOLD: 
                    context_text += f"\n---\n[Nguồn: {doc.metadata['source']} - Tr.{doc.metadata['page']}]\nNội dung: {doc.page_content}"
                    relevant_docs.append(doc)
        
        # --- PROMPT ---
        if not context_text:
            # Nếu không tìm thấy trong PDF, cho phép AI trả lời bằng kiến thức nền nhưng phải cảnh báo
            system_instruction = """
            Bạn là Chatbot KTC.
            Hiện tại bạn KHÔNG tìm thấy thông tin này trong tài liệu PDF được cung cấp.
            TUY NHIÊN, hãy trả lời câu hỏi dựa trên kiến thức chung của bạn về Tin học.
            BẮT BUỘC: Bắt đầu câu trả lời bằng dòng chữ in nghiêng: *"⚠️ Nội dung này chưa tìm thấy cụ thể trong tài liệu tải lên, đây là câu trả lời tham khảo:"*
            """
        else:
            system_instruction = f"""
            Bạn là trợ lý Tin học KTC. Dựa vào BỐI CẢNH sau để trả lời học sinh.
            BỐI CẢNH:
            {context_text}
            
            YÊU CẦU:
            1. Trả lời chính xác dựa trên bối cảnh.
            2. Trình bày ngắn gọn, dễ hiểu.
            """

        with st.chat_message("assistant", avatar="🤖"):
            placeholder = st.empty()
            full_response = ""
            try:
                chat_completion = client.chat.completions.create(
                    messages=[
                        {"role": "system", "content": system_instruction},
                        {"role": "user", "content": prompt}
                    ],
                    model=MODEL_NAME, stream=True, temperature=0.3
                )
                for chunk in chat_completion:
                    if chunk.choices[0].delta.content:
                        full_response += chunk.choices[0].delta.content
                        placeholder.markdown(full_response + "▌")
                placeholder.markdown(full_response)
                
                # Hiển thị minh chứng
                if relevant_docs:
                    with st.expander("📚 Minh chứng từ tài liệu (Click để xem)"):
                        for doc in relevant_docs:
                            st.markdown(f"**📄 {doc.metadata['source']} - Trang {doc.metadata['page']}**")
                            st.caption(doc.page_content[:300] + "...") 
                            st.divider()
                
                st.session_state.messages.append({"role": "assistant", "content": full_response})
            except Exception as e:
                st.error(f"Lỗi: {e}")

    st.markdown('<div class="footer-note">⚠️ Dự án KHKT trường THCS & THPT Phạm Kiệt.</div>', unsafe_allow_html=True)
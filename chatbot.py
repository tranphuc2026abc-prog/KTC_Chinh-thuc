import os
import glob
import time
import streamlit as st

# --- Imports tối ưu & Xử lý lỗi thư viện ---
try:
    from pypdf import PdfReader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_community.vectorstores import FAISS
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_core.documents import Document
    from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
    from groq import Groq
except ImportError as e:
    st.error(f"❌ Lỗi thư viện: {e}. Vui lòng kiểm tra file requirements.txt và chạy 'pip install -r requirements.txt'")
    st.stop()

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
    """Cấu hình trung tâm cho ứng dụng."""
    LLM_MODEL = 'llama-3.1-8b-instant'
    EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    TRANSLATION_MODEL = "Helsinki-NLP/opus-mt-vi-en"
    
    PDF_DIR = "PDF_KNOWLEDGE"
    VECTOR_DB_PATH = "faiss_db_index"
    LOGO_PATH = "LOGO.jpg"
    
    CHUNK_SIZE = 1000 
    CHUNK_OVERLAP = 200
    TOP_K_RETRIEVAL = 4

# ==============================================================================
# 2. UI/UX: GIAO DIỆN & CSS
# ==============================================================================

def inject_custom_css():
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;700&display=swap');
        
        html, body, [class*="css"] {
            font-family: 'Roboto', sans-serif;
        }

        .main-header {
            background: linear-gradient(90deg, #005C97 0%, #363795 100%);
            padding: 1.5rem;
            border-radius: 15px;
            color: white;
            text-align: center;
            margin-bottom: 2rem;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        .main-header h1 {
            color: white !important;
            font-weight: 700;
            margin: 0;
            font-size: 2rem;
        }
        .main-header p {
            margin-top: 0.5rem;
            opacity: 0.9;
            font-size: 1.1rem;
        }

        [data-testid="stSidebar"] {
            background-color: #f8f9fa;
        }
        .sidebar-card {
            background: white;
            padding: 15px;
            border-radius: 10px;
            border-left: 5px solid #363795;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
            margin-bottom: 20px;
        }
        .sidebar-card h4 {
            color: #363795;
            margin-top: 0;
            font-size: 1rem;
            font-weight: bold;
        }
        
        .stChatMessage {
            border-radius: 10px;
            border: 1px solid #f0f2f6;
        }
        .stChatMessage[data-testid="stChatMessage"]:nth-child(odd) {
             background-color: #f0f7ff;
        }
    </style>
    """, unsafe_allow_html=True)

# ==============================================================================
# 3. QUẢN LÝ TÀI NGUYÊN (CACHING & RESOURCE)
# ==============================================================================

@st.cache_resource(show_spinner=False)
def load_groq_client():
    api_key = st.secrets.get("GROQ_API_KEY")
    if not api_key:
        return None
    return Groq(api_key=api_key)

@st.cache_resource(show_spinner=False)
def load_embedding_model():
    try:
        return HuggingFaceEmbeddings(model_name=AppConfig.EMBEDDING_MODEL)
    except Exception:
        return None

@st.cache_resource(show_spinner=False)
def load_translator():
    try:
        tokenizer = AutoTokenizer.from_pretrained(AppConfig.TRANSLATION_MODEL)
        model = AutoModelForSeq2SeqLM.from_pretrained(AppConfig.TRANSLATION_MODEL)
        return pipeline("translation", model=model, tokenizer=tokenizer, src_lang="vi", tgt_lang="en")
    except Exception:
        return None

@st.cache_data(show_spinner=False)
def load_and_process_pdfs(pdf_dir):
    docs = []
    if not os.path.exists(pdf_dir):
        return docs
    
    pdf_files = glob.glob(os.path.join(pdf_dir, "*.pdf"))
    for pdf_path in pdf_files:
        try:
            reader = PdfReader(pdf_path)
            filename = os.path.basename(pdf_path)
            for i, page in enumerate(reader.pages):
                text = page.extract_text()
                if text and len(text.strip()) > 50:
                    docs.append(Document(
                        page_content=text,
                        metadata={"source": filename, "page": i + 1}
                    ))
        except Exception:
            continue
            
    if docs:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=AppConfig.CHUNK_SIZE,
            chunk_overlap=AppConfig.CHUNK_OVERLAP
        )
        return splitter.split_documents(docs)
    return []

# ==============================================================================
# 4. CORE LOGIC: RAG & AI PROCESSING
# ==============================================================================

class KnowledgeBase:
    def __init__(self):
        self.embeddings = load_embedding_model()

    def get_vector_store(self):
        if not self.embeddings:
            return None

        if os.path.exists(AppConfig.VECTOR_DB_PATH):
            try:
                return FAISS.load_local(
                    AppConfig.VECTOR_DB_PATH, 
                    self.embeddings, 
                    allow_dangerous_deserialization=True
                )
            except Exception:
                st.toast("⚠️ Database cũ lỗi, đang tạo mới...", icon="🔄")
        
        return self._create_new_db()

    def _create_new_db(self):
        splits = load_and_process_pdfs(AppConfig.PDF_DIR)
        if not splits:
            return None
        
        try:
            vector_db = FAISS.from_documents(splits, self.embeddings)
            vector_db.save_local(AppConfig.VECTOR_DB_PATH)
            return vector_db
        except Exception as e:
            st.error(f"Lỗi tạo Vector DB: {e}")
            return None

def translate_query(text, translator):
    if not translator: 
        return text
    try:
        result = translator(text[:500])
        return result[0]['translation_text'] if result else text
    except Exception:
        return text

def get_context(vector_db, query):
    if not vector_db:
        return "", []
    try:
        results = vector_db.similarity_search(query, k=AppConfig.TOP_K_RETRIEVAL)
        context_text = ""
        sources = []
        
        for doc in results:
            src = doc.metadata.get('source', 'Tài liệu')
            page = doc.metadata.get('page', '1')
            content = doc.page_content.replace("\n", " ")
            
            context_text += f"\n[Nguồn: {src} - Tr.{page}]: {content}"
            sources.append(f"{src} (Trang {page})")
            
        return context_text, list(set(sources))
    except Exception:
        return "", []

def generate_stream(client, context, question):
    system_prompt = f"""
    Bạn là KTC Assistant - Trợ lý ảo hỗ trợ học tập và nghiên cứu khoa học.
    
    NHIỆM VỤ:
    - Trả lời câu hỏi dựa trên thông tin được cung cấp trong [CONTEXT].
    - Nếu [CONTEXT] không có thông tin, hãy dùng kiến thức Tin học chuẩn của bạn (CT GDPT 2018).
    - Văn phong: Thân thiện, sư phạm, khuyến khích học sinh.
    - Định dạng: Sử dụng Markdown (in đậm, danh sách) để dễ đọc.

    [CONTEXT DỮ LIỆU]:
    {context}
    """
    
    try:
        completion = client.chat.completions.create(
            model=AppConfig.LLM_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question}
            ],
            stream=True,
            temperature=0.3
        )
        
        for chunk in completion:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
                
    except Exception as e:
        yield f"⚠️ Lỗi kết nối AI: {str(e)}"

# ==============================================================================
# 5. MAIN APPLICATION
# ==============================================================================

def main():
    inject_custom_css()
    
    with st.sidebar:
        if os.path.exists(AppConfig.LOGO_PATH):
            # SỬA: Bỏ use_container_width để tránh warning
            st.image(AppConfig.LOGO_PATH) 
        else:
            st.header("🤖 KTC AI")

        st.markdown("---")
        
        st.markdown("""
        <div class="sidebar-card">
            <h4>🏆 SẢN PHẨM DỰ THI KHKT<br>CẤP TRƯỜNG</h4>
            <p style="font-size: 0.9rem; margin-bottom: 5px;"><b>🏫 Đơn vị:</b> THCS & THPT Phạm Kiệt</p>
            <p style="font-size: 0.9rem; margin-bottom: 5px;"><b>👨‍💻 Tác giả:</b><br>- Bùi Tá Tùng<br>- Cao Sỹ Bảo Chung</p>
            <p style="font-size: 0.9rem;"><b>🧑‍🏫 GVHD:</b> Thầy Khanh</p>
        </div>
        """, unsafe_allow_html=True)
        
        with st.expander("🛠️ Cài đặt nâng cao"):
            top_k = st.slider("Độ sâu tìm kiếm (Chunks)", 1, 10, AppConfig.TOP_K_RETRIEVAL)
            AppConfig.TOP_K_RETRIEVAL = top_k
            st.info("Tăng độ sâu giúp tìm nhiều thông tin hơn nhưng có thể làm câu trả lời bị loãng.")

        if st.button("🗑️ Xóa lịch sử chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    st.markdown("""
    <div class="main-header">
        <h1>🎓 TRỢ LÝ ẢO KTC AI</h1>
        <p>Hệ thống hỗ trợ tra cứu kiến thức Tin học & Nghiên cứu khoa học</p>
    </div>
    """, unsafe_allow_html=True)

    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Chào bạn! Mình là trợ lý ảo KTC. Mình có thể giúp gì cho dự án KHKT của bạn hôm nay?"}
        ]

    groq_client = load_groq_client()
    translator = load_translator()
    
    if "vector_db" not in st.session_state:
        kb = KnowledgeBase()
        st.session_state.vector_db = kb.get_vector_store()

    if not groq_client:
        st.warning("⚠️ Chưa cấu hình GROQ_API_KEY. Vui lòng kiểm tra secrets.toml")
        st.stop()

    for msg in st.session_state.messages:
        avatar = "🧑‍🎓" if msg["role"] == "user" else "🤖"
        with st.chat_message(msg["role"], avatar=avatar):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Nhập câu hỏi của bạn tại đây..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user", avatar="🧑‍🎓"):
            st.markdown(prompt)

        with st.chat_message("assistant", avatar="🤖"):
            response_container = st.empty()
            
            with st.status("🚀 Đang xử lý...", expanded=True) as status:
                search_query = prompt
                if translator:
                    st.write("🌍 Đang tối ưu hóa câu hỏi (Dịch Việt -> Anh)...")
                    search_query = translate_query(prompt, translator)
                
                st.write("📚 Đang tra cứu tài liệu chuyên ngành...")
                context, sources = get_context(st.session_state.vector_db, search_query)
                
                if context:
                    st.write(f"✅ Tìm thấy {len(sources)} nguồn tài liệu liên quan.")
                else:
                    st.write("⚠️ Không tìm thấy trong tài liệu, sử dụng kiến thức nền.")
                
                status.update(label="✅ Đã xong!", state="complete", expanded=False)

            full_response = ""
            stream = generate_stream(groq_client, context, prompt)
            
            for chunk in stream:
                full_response += chunk
                response_container.markdown(full_response + "▌")
            
            response_container.markdown(full_response)
            
            if sources:
                with st.expander("📖 Xem nguồn tài liệu tham khảo"):
                    for src in sources:
                        st.caption(f"• {src}")

            st.session_state.messages.append({"role": "assistant", "content": full_response})

if __name__ == "__main__":
    main()
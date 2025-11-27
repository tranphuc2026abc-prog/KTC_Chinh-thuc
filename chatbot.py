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
    """Cấu hình trung tâm cho ứng dụng - Dễ dàng điều chỉnh"""
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
    MAX_CONTEXT_LENGTH = 3000  # Giới hạn độ dài context để tránh overload
    ENABLE_TRANSLATION = False  # TẮT dịch thuật để giảm RAM (model multilingual đã đủ tốt)

# ==============================================================================
# 2. UI/UX: GIAO DIỆN HIỆN ĐẠI & ANIMATIONS
# ==============================================================================

def inject_custom_css():
    """CSS tối ưu cho giao diện thi đấu - Modern & Professional"""
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
        
        /* Font chữ hiện đại */
        html, body, [class*="css"] {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        }

        /* Header chính với gradient đẹp mắt */
        .main-header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 2rem 1.5rem;
            border-radius: 20px;
            color: white;
            text-align: center;
            margin-bottom: 2rem;
            box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3);
            animation: fadeInDown 0.6s ease-out;
        }
        
        @keyframes fadeInDown {
            from {
                opacity: 0;
                transform: translateY(-20px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }
        
        .main-header h1 {
            color: white !important;
            font-weight: 700;
            margin: 0;
            font-size: 2.5rem;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
        }
        
        .main-header p {
            margin-top: 0.8rem;
            opacity: 0.95;
            font-size: 1.15rem;
            font-weight: 300;
        }

        /* Sidebar hiện đại */
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #f8f9fa 0%, #e9ecef 100%);
        }
        
        .sidebar-card {
            background: white;
            padding: 20px;
            border-radius: 15px;
            border-left: 5px solid #667eea;
            box-shadow: 0 4px 15px rgba(0,0,0,0.08);
            margin-bottom: 20px;
            transition: transform 0.2s ease;
        }
        
        .sidebar-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(0,0,0,0.12);
        }
        
        .sidebar-card h4 {
            color: #667eea;
            margin-top: 0;
            font-size: 1.1rem;
            font-weight: 700;
        }
        
        /* Chat bubbles đẹp hơn */
        .stChatMessage {
            border-radius: 15px;
            padding: 1rem;
            margin-bottom: 0.5rem;
            border: none;
            animation: fadeIn 0.3s ease-in;
        }
        
        @keyframes fadeIn {
            from { opacity: 0; }
            to { opacity: 1; }
        }
        
        /* Tin nhắn user - màu xanh nhạt */
        [data-testid="stChatMessageContent"]:has(+ [data-testid="stChatMessageAvatar"]) {
            background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
            border-left: 4px solid #2196f3;
        }
        
        /* Nút bấm đẹp hơn */
        .stButton > button {
            border-radius: 10px;
            font-weight: 600;
            transition: all 0.3s ease;
            border: none;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }
        
        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        }
        
        /* Status container đẹp hơn */
        [data-testid="stStatusWidget"] {
            border-radius: 10px;
            border: 2px solid #e0e0e0;
        }
        
        /* Input chat đẹp hơn */
        .stChatInputContainer {
            border-top: 2px solid #e0e0e0;
            padding-top: 1rem;
        }
    </style>
    """, unsafe_allow_html=True)

# ==============================================================================
# 3. QUẢN LÝ TÀI NGUYÊN VỚI CACHE THÔNG MINH
# ==============================================================================

@st.cache_resource(show_spinner=False)
def load_groq_client():
    """Load Groq API client với xử lý lỗi"""
    try:
        api_key = st.secrets.get("GROQ_API_KEY")
        if not api_key:
            st.error("⚠️ Chưa cấu hình GROQ_API_KEY trong Streamlit secrets!")
            return None
        return Groq(api_key=api_key)
    except Exception as e:
        st.error(f"❌ Lỗi kết nối Groq API: {e}")
        return None

@st.cache_resource(show_spinner=False)
def load_embedding_model():
    """Load model embedding với fallback thông minh"""
    try:
        with st.spinner("🔄 Đang tải model embedding (chỉ lần đầu)..."):
            embeddings = HuggingFaceEmbeddings(
                model_name=AppConfig.EMBEDDING_MODEL,
                model_kwargs={'device': 'cpu'},  # Force CPU để tránh lỗi CUDA
                encode_kwargs={'normalize_embeddings': True}  # Cải thiện độ chính xác
            )
        return embeddings
    except Exception as e:
        st.error(f"❌ Không thể load model embedding: {e}")
        st.info("💡 Thử khởi động lại ứng dụng hoặc kiểm tra kết nối mạng.")
        return None

@st.cache_data(show_spinner=False, ttl=3600)  # Cache 1 giờ
def load_and_process_pdfs(pdf_dir):
    """
    Đọc và xử lý tất cả PDF trong thư mục
    TTL=3600s để tự động refresh nếu có PDF mới
    """
    docs = []
    
    # Kiểm tra thư mục tồn tại
    if not os.path.exists(pdf_dir):
        st.warning(f"⚠️ Thư mục {pdf_dir} không tồn tại. Tạo thư mục rỗng.")
        os.makedirs(pdf_dir, exist_ok=True)
        return docs
    
    pdf_files = glob.glob(os.path.join(pdf_dir, "*.pdf"))
    
    if not pdf_files:
        st.info(f"📁 Chưa có file PDF nào trong thư mục '{pdf_dir}'")
        return docs
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for idx, pdf_path in enumerate(pdf_files):
        try:
            filename = os.path.basename(pdf_path)
            status_text.text(f"📄 Đang xử lý: {filename}")
            
            reader = PdfReader(pdf_path)
            for page_num, page in enumerate(reader.pages):
                text = page.extract_text()
                if text and len(text.strip()) > 50:  # Bỏ qua trang trống
                    docs.append(Document(
                        page_content=text,
                        metadata={"source": filename, "page": page_num + 1}
                    ))
            
            progress_bar.progress((idx + 1) / len(pdf_files))
            
        except Exception as e:
            st.warning(f"⚠️ Lỗi đọc file {filename}: {str(e)}")
            continue
    
    progress_bar.empty()
    status_text.empty()
    
    # Split documents thành chunks
    if docs:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=AppConfig.CHUNK_SIZE,
            chunk_overlap=AppConfig.CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        splits = splitter.split_documents(docs)
        st.success(f"✅ Đã xử lý {len(pdf_files)} file PDF → {len(splits)} chunks")
        return splits
    
    return []

# ==============================================================================
# 4. VECTOR DATABASE VỚI QUẢN LÝ THÔNG MINH
# ==============================================================================

class KnowledgeBase:
    """Quản lý Vector Database với các chức năng nâng cao"""
    
    def __init__(self):
        self.embeddings = load_embedding_model()
        self.db_path = AppConfig.VECTOR_DB_PATH

    def get_vector_store(self, force_rebuild=False):
        """
        Lấy hoặc tạo Vector Store
        force_rebuild=True: Xây dựng lại từ đầu (khi thêm PDF mới)
        """
        if not self.embeddings:
            st.error("❌ Model embedding chưa sẵn sàng!")
            return None

        # Kiểm tra DB có tồn tại không
        db_exists = os.path.exists(self.db_path)
        
        if db_exists and not force_rebuild:
            try:
                with st.spinner("🔍 Đang tải cơ sở dữ liệu vector..."):
                    vector_db = FAISS.load_local(
                        self.db_path, 
                        self.embeddings, 
                        allow_dangerous_deserialization=True
                    )
                st.success("✅ Đã tải Vector Database từ cache")
                return vector_db
            except Exception as e:
                st.warning(f"⚠️ Lỗi tải DB cũ: {e}. Đang tạo mới...")
        
        # Tạo DB mới
        return self._create_new_db()

    def _create_new_db(self):
        """Tạo Vector Database mới từ PDF"""
        splits = load_and_process_pdfs(AppConfig.PDF_DIR)
        
        if not splits:
            st.warning("⚠️ Không có dữ liệu để tạo Vector Database")
            return None
        
        try:
            with st.spinner(f"🔨 Đang xây dựng Vector Database ({len(splits)} chunks)..."):
                vector_db = FAISS.from_documents(splits, self.embeddings)
                vector_db.save_local(self.db_path)
            
            st.success("✅ Đã tạo và lưu Vector Database mới!")
            return vector_db
            
        except Exception as e:
            st.error(f"❌ Lỗi tạo Vector DB: {e}")
            return None
    
    def rebuild_database(self):
        """Reset và xây dựng lại toàn bộ Database"""
        # Xóa DB cũ
        if os.path.exists(self.db_path):
            import shutil
            shutil.rmtree(self.db_path)
            st.info("🗑️ Đã xóa Database cũ")
        
        # Xóa cache
        load_and_process_pdfs.clear()
        
        # Tạo mới
        return self.get_vector_store(force_rebuild=True)

# ==============================================================================
# 5. CORE LOGIC: RAG PROCESSING
# ==============================================================================

def get_context(vector_db, query):
    """
    Tìm kiếm context từ Vector DB
    Returns: (context_text, list_sources)
    """
    if not vector_db:
        return "", []
    
    try:
        # Similarity search với điểm số
        results = vector_db.similarity_search_with_score(
            query, 
            k=AppConfig.TOP_K_RETRIEVAL
        )
        
        context_parts = []
        sources = []
        total_length = 0
        
        for doc, score in results:
            # Lọc kết quả có score tốt (càng thấp càng tốt với FAISS)
            if score > 1.5:  # Threshold tùy chỉnh
                continue
            
            src = doc.metadata.get('source', 'Tài liệu')
            page = doc.metadata.get('page', '1')
            content = doc.page_content.replace("\n", " ").strip()
            
            # Giới hạn độ dài context
            if total_length + len(content) > AppConfig.MAX_CONTEXT_LENGTH:
                break
            
            context_parts.append(f"[{src} - Tr.{page}]:\n{content}")
            sources.append(f"{src} (Trang {page})")
            total_length += len(content)
        
        context_text = "\n\n".join(context_parts)
        return context_text, list(set(sources))
        
    except Exception as e:
        st.error(f"❌ Lỗi tìm kiếm: {e}")
        return "", []

def generate_stream(client, context, question):
    """
    Gọi Groq API để sinh câu trả lời streaming
    """
    # System prompt được tối ưu cho giáo dục
    system_prompt = f"""Bạn là KTC Assistant - trợ lý ảo thông minh hỗ trợ học tập môn Tin học THPT.

NHIỆM VỤ:
- Trả lời câu hỏi dựa trên CONTEXT được cung cấp bên dưới
- Nếu CONTEXT không đủ thông tin, hãy dùng kiến thức Tin học chuẩn (Chương trình GDPT 2018)
- Nếu không biết, hãy thành thật nói "Mình chưa có đủ thông tin để trả lời chính xác"

VĂN PHONG:
- Thân thiện, gần gũi như một người bạn học (xưng hô: mình/bạn)
- Giải thích dễ hiểu, có ví dụ cụ thể
- Khuyến khích tinh thần học tập

ĐỊNH DẠNG:
- Sử dụng Markdown: **in đậm** cho thuật ngữ quan trọng
- Dùng bullet points cho danh sách
- Chia đoạn rõ ràng để dễ đọc

[CONTEXT TÀI LIỆU]:
{context if context else "Không có tài liệu liên quan trong cơ sở dữ liệu."}
"""
    
    try:
        completion = client.chat.completions.create(
            model=AppConfig.LLM_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question}
            ],
            stream=True,
            temperature=0.4,  # Tăng nhẹ để câu trả lời tự nhiên hơn
            max_tokens=1500
        )
        
        for chunk in completion:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
                
    except Exception as e:
        yield f"⚠️ Lỗi kết nối AI: {str(e)}\n\nVui lòng thử lại sau!"

# ==============================================================================
# 6. MAIN APPLICATION
# ==============================================================================

def main():
    """Hàm chính chạy ứng dụng"""
    
    # Kiểm tra dependencies
    if not DEPENDENCIES_OK:
        st.error(f"❌ Thiếu thư viện: {IMPORT_ERROR}")
        st.info("💡 Chạy lệnh: `pip install -r requirements.txt`")
        st.stop()
    
    # Inject CSS
    inject_custom_css()
    
    # ============= SIDEBAR =============
    with st.sidebar:
        # Logo
        if os.path.exists(AppConfig.LOGO_PATH):
            st.image(AppConfig.LOGO_PATH, use_container_width=True)
        else:
            st.markdown("### 🤖 KTC AI Assistant")

        st.markdown("---")
        
        # Thông tin dự án
        st.markdown("""
        <div class="sidebar-card">
            <h4>🏆 SẢN PHẨM KHKT CẤP TRƯỜNG</h4>
            <p style="font-size: 0.9rem; margin: 8px 0;"><b>🏫 Đơn vị:</b><br>THCS & THPT Phạm Kiệt</p>
            <p style="font-size: 0.9rem; margin: 8px 0;"><b>👨‍💻 Tác giả:</b><br>• Bùi Tá Tùng<br>• Cao Sỹ Bảo Chung</p>
            <p style="font-size: 0.9rem; margin: 8px 0;"><b>🧑‍🏫 GVHD:</b> Thầy Khanh</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Cài đặt nâng cao
        with st.expander("🛠️ Cài đặt nâng cao"):
            top_k = st.slider(
                "Số lượng chunks tìm kiếm", 
                min_value=1, 
                max_value=10, 
                value=AppConfig.TOP_K_RETRIEVAL,
                help="Tăng để tìm nhiều thông tin hơn, nhưng có thể làm chậm"
            )
            AppConfig.TOP_K_RETRIEVAL = top_k
            
            if st.button("🔄 Làm mới Database", use_container_width=True):
                with st.spinner("Đang xây dựng lại Database..."):
                    kb = KnowledgeBase()
                    st.session_state.vector_db = kb.rebuild_database()
                st.success("✅ Đã làm mới Database!")
                st.rerun()

        st.markdown("---")
        
        # Nút xóa lịch sử
        if st.button("🗑️ Xóa lịch sử chat", use_container_width=True):
            st.session_state.messages = []
            st.success("✅ Đã xóa lịch sử!")
            time.sleep(0.5)
            st.rerun()
        
        # Hướng dẫn sử dụng
        with st.expander("📖 Hướng dẫn sử dụng"):
            st.markdown("""
            **Cách sử dụng:**
            1. Đặt file PDF vào thư mục `PDF_KNOWLEDGE`
            2. Nhấn "Làm mới Database" ở trên
            3. Bắt đầu hỏi câu hỏi!
            
            **Mẹo:**
            - Hỏi câu hỏi cụ thể để được trả lời tốt hơn
            - Kiểm tra "Nguồn tài liệu" để xác minh thông tin
            """)

    # ============= MAIN CONTENT =============
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🎓 TRỢ LÝ ẢO KTC</h1>
        <p>Hệ thống AI hỗ trợ học tập Tin học & Nghiên cứu Khoa học</p>
    </div>
    """, unsafe_allow_html=True)

    # Khởi tạo chat history
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant", 
                "content": "Chào bạn! 👋 Mình là **KTC Assistant**.\n\nMình có thể giúp bạn:\n- Giải đáp thắc mắc về Tin học\n- Hỗ trợ dự án KHKT\n- Tra cứu tài liệu chuyên ngành\n\nHãy đặt câu hỏi để bắt đầu nhé! 😊"
            }
        ]

    # Load resources
    groq_client = load_groq_client()
    
    if not groq_client:
        st.error("❌ Không thể kết nối Groq API. Vui lòng kiểm tra cấu hình!")
        st.stop()

    # Load/Create Vector DB
    if "vector_db" not in st.session_state:
        kb = KnowledgeBase()
        st.session_state.vector_db = kb.get_vector_store()

    # Hiển thị lịch sử chat
    for msg in st.session_state.messages:
        avatar = "🧑‍🎓" if msg["role"] == "user" else "🤖"
        with st.chat_message(msg["role"], avatar=avatar):
            st.markdown(msg["content"])

    # Chat input
    if prompt := st.chat_input("💬 Nhập câu hỏi của bạn tại đây..."):
        # Lưu tin nhắn user
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user", avatar="🧑‍🎓"):
            st.markdown(prompt)

        # Xử lý và trả lời
        with st.chat_message("assistant", avatar="🤖"):
            response_placeholder = st.empty()
            
            # Status processing
            with st.status("🚀 Đang xử lý câu hỏi...", expanded=True) as status:
                st.write("🔍 Đang tìm kiếm tài liệu liên quan...")
                context, sources = get_context(st.session_state.vector_db, prompt)
                
                if sources:
                    st.write(f"✅ Tìm thấy {len(sources)} nguồn tài liệu")
                else:
                    st.write("⚠️ Không tìm thấy trong tài liệu, sử dụng kiến thức nền")
                
                st.write("💭 Đang suy nghĩ và soạn câu trả lời...")
                status.update(label="✨ Hoàn thành!", state="complete", expanded=False)

            # Stream response
            full_response = ""
            for chunk in generate_stream(groq_client, context, prompt):
                full_response += chunk
                response_placeholder.markdown(full_response + "▌")
            
            response_placeholder.markdown(full_response)
            
            # Hiển thị nguồn tài liệu
            if sources:
                with st.expander("📚 Nguồn tài liệu tham khảo"):
                    for idx, src in enumerate(sources, 1):
                        st.caption(f"{idx}. {src}")

            # Lưu response
            st.session_state.messages.append({
                "role": "assistant", 
                "content": full_response
            })

if __name__ == "__main__":
    main()
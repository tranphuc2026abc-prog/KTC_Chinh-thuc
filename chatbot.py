import os
import glob
import time
import streamlit as st
from typing import List, Tuple, Optional, Generator, Any

# --- AI & Data Processing Libraries ---
# Tối ưu import để tránh nạp thư viện không cần thiết nếu chưa dùng
try:
    from pypdf import PdfReader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_community.vectorstores import FAISS
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_core.documents import Document
    from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
    from groq import Groq
except ImportError as e:
    st.error(f"❌ Thiếu thư viện: {e}. Vui lòng chạy: pip install -r requirements.txt")
    st.stop()

# ==============================================================================
# 1. CẤU HÌNH HỆ THỐNG (CONFIG)
# ==============================================================================

st.set_page_config(
    page_title="KTC Assistant - Trợ lý Tin học",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

class AppConfig:
    """Class chứa toàn bộ cấu hình để dễ dàng quản lý và thay đổi."""
    # Model Settings
    LLM_MODEL = 'llama-3.1-8b-instant'
    # Model Embedding nhẹ nhưng hiệu quả cho tiếng Việt/Anh
    EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    TRANSLATION_MODEL = "Helsinki-NLP/opus-mt-vi-en"
    
    # Paths
    PDF_DIR = "PDF_KNOWLEDGE"
    VECTOR_DB_PATH = "faiss_db_index"
    LOGO_PATH = "LOGO.jpg"
    
    # RAG Parameters
    CHUNK_SIZE = 1000 
    CHUNK_OVERLAP = 200
    TOP_K_RETRIEVAL = 5 # Giữ ở mức 5 để cân bằng tốc độ và độ chính xác

# ==============================================================================
# 2. GIAO DIỆN & CSS (UI/UX)
# ==============================================================================

def inject_custom_css():
    """CSS tùy chỉnh để giao diện sạch, đẹp và chuyên nghiệp hơn."""
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
        
        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
        }
        
        /* Tinh chỉnh Header chính */
        .main-header {
            background: linear-gradient(135deg, #0f4c81 0%, #00c6ff 100%);
            padding: 20px;
            border-radius: 12px;
            color: white;
            text-align: center;
            margin-bottom: 25px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        }
        .main-header h1 {
            font-size: 2.2rem;
            font-weight: 800;
            margin: 0;
            color: #ffffff !important;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
        }
        .main-header p {
            font-size: 1.1rem;
            opacity: 0.95;
            margin-top: 5px;
        }

        /* Tinh chỉnh Sidebar */
        [data-testid="stSidebar"] {
            background-color: #f8f9fa;
        }
        .sidebar-info {
            background-color: #ffffff;
            padding: 15px;
            border-radius: 10px;
            border: 1px solid #e0e0e0;
            border-left: 5px solid #0f4c81;
            margin-bottom: 15px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.05);
        }
        .sidebar-title {
            color: #0f4c81;
            font-weight: 800;
            text-align: center;
            font-size: 0.9rem;
            margin-bottom: 10px;
            text-transform: uppercase;
        }
        .sidebar-text {
            font-size: 0.85rem;
            color: #333;
            line-height: 1.5;
        }

        /* Chat bubble tweaks */
        .stChatMessage {
            border-radius: 12px;
            padding: 8px;
        }
    </style>
    """, unsafe_allow_html=True)

# ==============================================================================
# 3. QUẢN LÝ TÀI NGUYÊN (CACHING & INITIALIZATION)
# ==============================================================================

@st.cache_resource(show_spinner=False)
def load_groq_client():
    """Khởi tạo Groq Client an toàn."""
    try:
        api_key = st.secrets.get("GROQ_API_KEY")
        if not api_key:
            st.error("⚠️ Chưa cấu hình GROQ_API_KEY trong Secrets.")
            return None
        return Groq(api_key=api_key)
    except Exception as e:
        st.error(f"❌ Lỗi kết nối Groq: {e}")
        return None

@st.cache_resource(show_spinner=False)
def load_embedding_model():
    """Load model vector hóa (chạy 1 lần)."""
    try:
        return HuggingFaceEmbeddings(model_name=AppConfig.EMBEDDING_MODEL)
    except Exception as e:
        st.error(f"❌ Lỗi tải Embedding Model: {e}")
        return None

@st.cache_resource(show_spinner=False)
def load_translator():
    """Load model dịch thuật (chạy 1 lần)."""
    try:
        # Sử dụng device=-1 cho CPU (Streamlit Cloud thường không có GPU)
        tokenizer = AutoTokenizer.from_pretrained(AppConfig.TRANSLATION_MODEL)
        model = AutoModelForSeq2SeqLM.from_pretrained(AppConfig.TRANSLATION_MODEL)
        translator = pipeline("translation", model=model, tokenizer=tokenizer, src_lang="vi", tgt_lang="en")
        return translator
    except Exception as e:
        # Không return None để app vẫn chạy được dù không có dịch
        print(f"Translator Warning: {e}") 
        return None

@st.cache_data(show_spinner=False)
def load_pdf_documents(pdf_dir: str) -> List[Document]:
    """Đọc tất cả PDF và trả về list Document (cached để tránh đọc lại nhiều lần)."""
    docs: List[Document] = []
    try:
        if not os.path.exists(pdf_dir):
            return docs
        pdf_files = glob.glob(os.path.join(pdf_dir, "*.pdf"))
        for pdf_path in pdf_files:
            try:
                reader = PdfReader(pdf_path)
                filename = os.path.basename(pdf_path)
                for i, page in enumerate(reader.pages):
                    try:
                        text = page.extract_text()
                        if text and len(text.strip()) > 50:
                            docs.append(Document(
                                page_content=text,
                                metadata={"source": filename, "page": i + 1}
                            ))
                    except Exception:
                        # Bỏ qua 1 trang nếu lỗi
                        continue
            except Exception:
                # Bỏ qua file nếu lỗi
                continue
    except Exception:
        # Bảo vệ tổng thể
        pass
    return docs

# ==============================================================================
# 4. LOGIC XỬ LÝ DỮ LIỆU & RAG (CORE)
# ==============================================================================

class KnowledgeBaseManager:
    """Quản lý việc đọc PDF và tạo Vector DB."""
    
    def __init__(self):
        self.embeddings = load_embedding_model()
    
    def get_vector_store(self):
        """Lấy Vector Store, nếu chưa có thì tự build."""
        if not self.embeddings:
            return None
            
        # 1. Thử load từ ổ cứng
        try:
            if os.path.exists(AppConfig.VECTOR_DB_PATH):
                try:
                    return FAISS.load_local(
                        AppConfig.VECTOR_DB_PATH, 
                        self.embeddings, 
                        allow_dangerous_deserialization=True
                    )
                except Exception:
                    # báo nhưng không crash
                    st.toast("⚠️ Database lỗi, đang tạo lại...", icon="🔄")
        except Exception:
            # ignore và build mới
            pass
        
        # 2. Nếu chưa có hoặc lỗi, build mới
        return self._build_new_vector_store()

    def _build_new_vector_store(self):
        """Hàm nội bộ để đọc PDF và tạo index."""
        try:
            docs = load_pdf_documents(AppConfig.PDF_DIR)
            if not docs:
                return None

            # Chia nhỏ văn bản
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=AppConfig.CHUNK_SIZE,
                chunk_overlap=AppConfig.CHUNK_OVERLAP
            )
            splits = splitter.split_documents(docs)

            # Tạo và lưu DB (bọc try-except)
            try:
                vector_db = FAISS.from_documents(splits, self.embeddings)
                # đảm bảo folder tồn tại
                if not os.path.exists(AppConfig.VECTOR_DB_PATH):
                    os.makedirs(AppConfig.VECTOR_DB_PATH, exist_ok=True)
                vector_db.save_local(AppConfig.VECTOR_DB_PATH)
                return vector_db
            except Exception as e:
                st.warning(f"Không thể tạo hoặc lưu Vector DB: {e}")
                return None
        except Exception as e:
            st.warning(f"Lỗi khi xây dựng Vector DB: {e}")
            return None

# ==============================================================================
# 5. UTILITIES (HÀM HỖ TRỢ)
# ==============================================================================

def translate_query(text: str, translator) -> str:
    """Dịch câu hỏi sang tiếng Anh."""
    if not translator or not text:
        return text
    try:
        # giới hạn độ dài tránh OOM
        short_text = text[:512]
        out = translator(short_text)
        # một số pipeline trả dict hoặc list
        if isinstance(out, list) and len(out) > 0:
            # key có thể là 'translation_text' hoặc 'translation'
            t = out[0]
            if isinstance(t, dict):
                return t.get('translation_text') or t.get('translation') or text
            elif isinstance(t, str):
                return t
        elif isinstance(out, dict):
            return out.get('translation_text') or out.get('translation') or text
        return text
    except Exception:
        return text

def retrieve_info(vector_db, query: str) -> Tuple[str, List[str]]:
    """Tìm kiếm thông tin trong Vector DB."""
    if not vector_db:
        return "", []
    try:
        # Tìm kiếm similarity (bọc try)
        results = vector_db.similarity_search(query, k=AppConfig.TOP_K_RETRIEVAL)
        context_parts = []
        sources = []
        for d in results:
            try:
                src = d.metadata.get('source', 'Unknown')
                page = d.metadata.get('page', '?')
                content = d.page_content or ""
                context_parts.append(f"[Nguồn: {src} - Tr. {page}]\n{content}")
                sources.append(f"{src} (Trang {page})")
            except Exception:
                continue
        context = "\n\n".join(context_parts)
        # unique sources while keeping order
        seen = set()
        unique_sources = []
        for s in sources:
            if s not in seen:
                unique_sources.append(s)
                seen.add(s)
        return context, unique_sources
    except Exception:
        return "", []

def _safe_stream_iter(stream_obj: Any) -> Generator[str, None, None]:
    """
    Duyệt generator stream một cách an toàn.
    Trả về từng content fragment (string).
    """
    try:
        for chunk in stream_obj:
            try:
                # nhiều SDK khác nhau - lấy content trong nhiều cấu trúc
                content = ""
                # cấu trúc theo OpenAI-like
                if hasattr(chunk, "choices"):
                    choice0 = chunk.choices[0]
                    # delta có thể là dict hay object
                    delta = getattr(choice0, "delta", None) or (choice0.get("delta") if isinstance(choice0, dict) else None)
                    if delta:
                        if isinstance(delta, dict):
                            content = delta.get("content", "") or delta.get("text", "")
                        else:
                            content = getattr(delta, "content", "") or getattr(delta, "text", "")
                    else:
                        # có thể là full text in choices[0].message/content
                        text = getattr(choice0, "text", None)
                        if text:
                            content = text
                # fallback cho dict chunk
                if isinstance(chunk, dict):
                    # openai-style
                    choices = chunk.get("choices")
                    if choices and isinstance(choices, list):
                        delta = choices[0].get("delta", {})
                        content = delta.get("content") or delta.get("text") or content
                        if not content:
                            content = choices[0].get("text") or content
                if content:
                    yield content
            except Exception:
                continue
    except Exception:
        return

def generate_stream_response(client, context, question):
    """Gọi LLM trả về Stream."""
    system_prompt = f"""
    Bạn là KTC Assistant, một trợ lý giáo dục ảo, chuyên gia về Tin học.
    
    NHIỆM VỤ: Trả lời câu hỏi dựa trên ngữ cảnh được cung cấp.
    
    NGUYÊN TẮC:
    1. Ưu tiên dùng thông tin trong [CONTEXT]. Nếu không có, hãy dùng kiến thức chuẩn của bạn về Tin học (GDPT 2018).
    2. Trả lời bằng Tiếng Việt, văn phong sư phạm, dễ hiểu, thân thiện.
    3. Dùng Markdown để trình bày (in đậm từ khóa, gạch đầu dòng).
    
    [CONTEXT - DỮ LIỆU TRA CỨU]:
    {context}
    """
    
    try:
        # Gọi client, trả về generator/iterable
        return client.chat.completions.create(
            model=AppConfig.LLM_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question}
            ],
            stream=True,
            temperature=0.3 # Giảm nhiệt độ để câu trả lời chính xác hơn với tài liệu
        )
    except Exception as e:
        return f"❌ Lỗi kết nối AI: {str(e)}"

# ==============================================================================
# 6. MAIN APP
# ==============================================================================

def main():
    inject_custom_css()
    
    # --- Sidebar ---
    with st.sidebar:
        # Logo căn giữa đẹp mắt
        if os.path.exists(AppConfig.LOGO_PATH):
            col1, col2, col3 = st.columns([1, 4, 1])
            with col2:
                st.image(AppConfig.LOGO_PATH, use_container_width=True)
        else:
            st.markdown("<div style='text-align:center; font-size: 50px;'>🤖</div>", unsafe_allow_html=True)

        st.markdown("---")
        
        # Thông tin dự án clean và chuyên nghiệp
        st.markdown("""
        <div class="sidebar-info">
            <div class="sidebar-title">🏆 SẢN PHẨM DỰ THI<br>KHKT CẤP TRƯỜNG</div>
            <div class="sidebar-text">
                <b>Đơn vị:</b> THCS & THPT Phạm Kiệt<br>
                <b>Tác giả:</b> Bùi Tá Tùng & Cao Sỹ Bảo Chung<br>
                <b>GVHD:</b> Thầy Khanh
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Hướng dẫn sử dụng nhanh trong expander
        with st.expander("ℹ️ Hướng dẫn sử dụng"):
            st.write("""
            - Gõ câu hỏi vào ô chat bên dưới (ví dụ: "Cấu trúc rẽ nhánh là gì?")  
            - Ứng dụng sẽ tìm trong tài liệu PDF (nếu đã nạp) và trả lời dựa trên ngữ cảnh.  
            - Nếu không có tài liệu, AI sẽ trả lời dựa trên kiến thức nền.  
            - Dùng nút "🗑️ Xóa lịch sử" để reset phiên làm việc.
            """)

        st.markdown("---")
        # Nút chức năng
        if st.button("🗑️ Xóa lịch sử trò chuyện", use_container_width=True):
            st.session_state.messages = []
            # giữ vector_db nếu có; chỉ xóa messages
            st.experimental_rerun()

    # --- Main Interface Header ---
    st.markdown("""
    <div class="main-header">
        <h1>🎓 TRỢ LÝ ẢO KTC AI</h1>
        <p>Hỗ trợ tra cứu kiến thức Tin học & Nghiên cứu khoa học</p>
    </div>
    """, unsafe_allow_html=True)

    # Khởi tạo Session State (giữ nguyên key "messages")
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Chào bạn! Thầy Khanh và nhóm KHKT đã nạp đầy đủ dữ liệu. Bạn cần tìm hiểu kiến thức gì nào? 🧑‍💻"}
        ]

    # Load Resources (Chỉ load 1 lần) - safe wrapped
    groq_client = load_groq_client()
    translator = load_translator()

    # Check Vector DB (Lazy loading để app mở nhanh hơn)
    if "vector_db" not in st.session_state:
        try:
            kb = KnowledgeBaseManager()
            db = kb.get_vector_store()
            st.session_state.vector_db = db
            if db:
                # Không gây crash nếu toast fails
                try:
                    st.toast("✅ Đã nạp dữ liệu thành công!", icon="✅")
                except Exception:
                    pass
            else:
                # vẫn cho phép chat (AI không có context từ PDF)
                st.session_state.vector_db = None
        except Exception as e:
            st.session_state.vector_db = None
            st.warning(f"⚠️ Lỗi khi khởi tạo KB: {e}")

    # Nếu không có kết nối GROQ thì cảnh báo và dừng (giữ logic gốc)
    if not groq_client:
        st.warning("⚠️ Hệ thống đang bảo trì kết nối AI. Vui lòng kiểm tra lại sau.")
        st.stop()

    # Bố cục chính: left chat history, right controls (nếu muốn)
    col_left, col_right = st.columns([3, 1])

    # Hiển thị lịch sử chat (left)
    with col_left:
        for msg in st.session_state.messages:
            avatar = "🧑‍🎓" if msg["role"] == "user" else "🤖"
            # dùng st.chat_message để hiệu ứng mượt
            try:
                with st.chat_message(msg["role"], avatar=avatar):
                    st.markdown(msg["content"])
            except Exception:
                # fallback an toàn
                st.write(f"{avatar} {msg['content']}")

        # Input chat (giữ luồng)
        prompt = st.chat_input("Nhập câu hỏi của bạn (Ví dụ: Cấu trúc rẽ nhánh là gì?)...")

        if prompt:
            # 1. Hiển thị câu hỏi người dùng
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user", avatar="🧑‍🎓"):
                st.markdown(prompt)

            # 2. AI xử lý (Dùng st.status để hiển thị quy trình)
            with st.chat_message("assistant", avatar="🤖"):
                response_placeholder = st.empty()
                full_response = ""
                sources = []

                with st.status("🔍 Hệ thống đang phân tích...", expanded=True) as status:
                    # Bước 1: Dịch thuật (Nếu cần)
                    search_query = prompt
                    if translator:
                        try:
                            st.write("🇬🇧 Đang dịch câu hỏi sang tiếng Anh để tra cứu sâu hơn...")
                            translated = translate_query(prompt, translator)
                            if translated and translated != prompt:
                                search_query = translated
                        except Exception:
                            # ignore dịch nếu lỗi
                            search_query = prompt

                    # Bước 2: Truy xuất dữ liệu (RAG)
                    st.write("📚 Đang quét cơ sở dữ liệu PDF...")
                    try:
                        context_text, sources = retrieve_info(st.session_state.get("vector_db"), search_query)
                    except Exception:
                        context_text, sources = "", []

                    if not context_text:
                        context_text = "Không tìm thấy dữ liệu trong sách. Sử dụng kiến thức nền tảng."
                        st.write("⚠️ Không tìm thấy trong tài liệu, sử dụng kiến thức AI.")
                    else:
                        st.write("✅ Đã tìm thấy thông tin liên quan.")
                    
                    status.update(label="✅ Đã xử lý xong!", state="complete", expanded=False)

                # Bước 3: Streaming câu trả lời
                stream = generate_stream_response(groq_client, context_text, prompt)
                
                if isinstance(stream, str):
                    # Trường hợp lỗi trả về string
                    full_response = stream
                    response_placeholder.markdown(full_response)
                else:
                    # stream có thể là iterable/generator SDK-specific
                    try:
                        for fragment in _safe_stream_iter(stream):
                            full_response += fragment
                            # hiển thị tạm thời với con trỏ
                            response_placeholder.markdown(full_response + "▌")
                        # cuối cùng hiển thị đầy đủ (không hiển thị con trỏ)
                        response_placeholder.markdown(full_response)
                    except Exception as e:
                        # fallback nếu streaming hỏng
                        response_placeholder.markdown(f"❌ Lỗi khi stream kết quả: {e}")

                # Bước 4: Hiển thị nguồn (Minh chứng khoa học)
                if sources:
                    with st.expander("📖 Nguồn tài liệu tham khảo"):
                        for src in sources:
                            st.markdown(f"- {src}")

                # Lưu vào session history
                try:
                    st.session_state.messages.append({"role": "assistant", "content": full_response})
                except Exception:
                    # Nếu không lưu được thì ít nhất vẫn hiển thị nhưng cảnh báo
                    st.warning("⚠️ Không thể lưu lịch sử chat vào session_state.")

    # Right column: một số control / thống kê nhanh
    with col_right:
        st.markdown("### ⚙️ Trạng thái hệ thống")
        emb_status = "OK" if load_embedding_model() else "Không sẵn sàng"
        st.markdown(f"- Embedding: **{emb_status}**")
        st.markdown(f"- Vector DB: **{'Có' if st.session_state.get('vector_db') else 'Chưa có'}**")
        st.markdown(f"- Translator: **{'Có' if translator else 'Không'}**")

        st.markdown("---")
        st.markdown("### 🔎 Tùy chọn tìm kiếm")
        topk = st.number_input("Số kết quả truy xuất (TOP K)", min_value=1, max_value=20, value=AppConfig.TOP_K_RETRIEVAL, step=1)
        AppConfig.TOP_K_RETRIEVAL = int(topk)

if __name__ == "__main__":
    main()

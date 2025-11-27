# app.py
import os
import glob
from typing import List, Tuple, Any, Generator
import streamlit as st

# --- Third-party imports wrapped for friendly error message ---
try:
    from pypdf import PdfReader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_community.vectorstores import FAISS
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_core.documents import Document
    from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
    from groq import Groq
except ImportError as e:
    st.error(f"❌ Thiếu thư viện: {e}. Chạy: pip install -r requirements.txt rồi thử lại.")
    st.stop()

# =========================
# Config
# =========================
st.set_page_config(page_title="KTC Assistant - RAG (6 PDF)", page_icon="🎓", layout="wide")

class AppConfig:
    PDF_DIR = "PDF_KNOWLEDGE"           # đặt 6 file pdf vào đây
    VECTOR_DB_PATH = "faiss_db_index"
    LOGO_PATH = "LOGO.jpg"
    EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    TRANSLATION_MODEL = "Helsinki-NLP/opus-mt-vi-en"
    LLM_MODEL = "llama-3.1-8b-instant"
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200
    TOP_K = 5

# =========================
# Minimal CSS / UI helper
# =========================
def inject_css():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
        html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
        .header { background: linear-gradient(90deg,#0f4c81,#00c6ff); padding:18px; border-radius:10px; color:white; }
        .sidebar-info { padding:12px; border-radius:8px; background:#fff; border-left:6px solid #0f4c81; }
        </style>
        """, unsafe_allow_html=True
    )

# =========================
# Caching resources
# =========================

@st.cache_resource(show_spinner=False)
def get_groq_client() -> Any:
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
def read_all_pdfs(pdf_dir: str) -> List[Document]:
    """Đọc tất cả PDF trong thư mục và trả về list Document. Cached để không đọc lại."""
    docs: List[Document] = []
    if not os.path.exists(pdf_dir):
        return docs
    files = sorted(glob.glob(os.path.join(pdf_dir, "*.pdf")))
    # Nếu bạn mong đợi chính xác 6 file, ta có thể cảnh báo nhưng vẫn tiếp tục
    for path in files:
        try:
            reader = PdfReader(path)
            name = os.path.basename(path)
            for i, page in enumerate(reader.pages):
                try:
                    txt = page.extract_text()
                    if txt and txt.strip():
                        docs.append(Document(page_content=txt, metadata={"source": name, "page": i+1}))
                except Exception:
                    continue
        except Exception:
            continue
    return docs

@st.cache_resource(show_spinner=False)
def build_or_load_faiss(docs: List[Document], embeddings) -> Any:
    """Nếu đã lưu local thì load, nếu chưa thì build rồi lưu. Cached resource vì nặng."""
    try:
        if embeddings is None:
            return None
        # Nếu đã có folder DB (FAISS.save_local lưu nhiều file), thử load
        if os.path.exists(AppConfig.VECTOR_DB_PATH):
            try:
                return FAISS.load_local(AppConfig.VECTOR_DB_PATH, embeddings, allow_dangerous_deserialization=True)
            except Exception:
                # load thất bại -> rebuild
                pass
        # Nếu không có docs thì trả None
        if not docs:
            return None
        # Chia text thành chunks
        splitter = RecursiveCharacterTextSplitter(chunk_size=AppConfig.CHUNK_SIZE, chunk_overlap=AppConfig.CHUNK_OVERLAP)
        splits = splitter.split_documents(docs)
        if not splits:
            return None
        db = FAISS.from_documents(splits, embeddings)
        # Save local (lưu nhiều file -> tạo folder)
        try:
            if not os.path.exists(AppConfig.VECTOR_DB_PATH):
                os.makedirs(AppConfig.VECTOR_DB_PATH, exist_ok=True)
            db.save_local(AppConfig.VECTOR_DB_PATH)
        except Exception:
            pass
        return db
    except Exception:
        return None

# =========================
# Utilities
# =========================

def translate_query(text: str, translator) -> str:
    if not translator or not text:
        return text
    try:
        out = translator(text[:512])
        if isinstance(out, list) and len(out) > 0:
            item = out[0]
            if isinstance(item, dict):
                return item.get("translation_text") or item.get("translation") or text
            elif isinstance(item, str):
                return item
        if isinstance(out, dict):
            return out.get("translation_text") or out.get("translation") or text
        return text
    except Exception:
        return text

def retrieve_from_db(db: Any, query: str, top_k: int = AppConfig.TOP_K) -> Tuple[str, List[str]]:
    if not db:
        return "", []
    try:
        docs = db.similarity_search(query, k=top_k)
        parts = []
        sources = []
        for d in docs:
            src = d.metadata.get("source", "Unknown")
            page = d.metadata.get("page", "?")
            parts.append(f"[Nguồn: {src} - Tr. {page}]\n{d.page_content}")
            sources.append(f"{src} (Trang {page})")
        # unique sources keep order
        seen = set()
        uniq = []
        for s in sources:
            if s not in seen:
                uniq.append(s); seen.add(s)
        return "\n\n".join(parts), uniq
    except Exception:
        return "", []

def _safe_stream_iter(stream_obj: Any) -> Generator[str, None, None]:
    """Duyệt generator stream một cách an toàn, yield fragment string."""
    try:
        for chunk in stream_obj:
            try:
                # OpenAI-like or SDK-like handling
                if isinstance(chunk, dict):
                    choices = chunk.get("choices", [])
                    if choices:
                        delta = choices[0].get("delta", {}) or {}
                        text = delta.get("content") or delta.get("text") or choices[0].get("text")
                        if text:
                            yield text
                            continue
                # object-like
                if hasattr(chunk, "choices"):
                    choice0 = chunk.choices[0]
                    delta = getattr(choice0, "delta", None) or (choice0.get("delta") if isinstance(choice0, dict) else None)
                    if delta:
                        if isinstance(delta, dict):
                            content = delta.get("content") or delta.get("text")
                        else:
                            content = getattr(delta, "content", None) or getattr(delta, "text", None)
                        if content:
                            yield content
                            continue
                    # fallback
                    text = getattr(choice0, "text", None) or (choice0.get("text") if isinstance(choice0, dict) else None)
                    if text:
                        yield text
                        continue
            except Exception:
                continue
    except Exception:
        return

def generate_stream_response(client, context: str, question: str) -> Any:
    """Gọi LLM (groq client) giống luồng cũ; trả về iterable/generator hoặc string lỗi."""
    system_prompt = f"""
Bạn là KTC Assistant, trợ lý giáo dục về Tin học.
Ưu tiên dùng thông tin trong CONTEXT dưới đây. Nếu không có, hãy dùng kiến thức nền.
Trả lời bằng tiếng Việt, rõ ràng, sư phạm.
[CONTEXT]:
{context}
"""
    try:
        return client.chat.completions.create(
            model=AppConfig.LLM_MODEL,
            messages=[{"role":"system","content":system_prompt},{"role":"user","content":question}],
            stream=True,
            temperature=0.2
        )
    except Exception as e:
        return f"❌ Lỗi kết nối AI: {e}"

# =========================
# Main app (gọn, tối ưu cho 6 file)
# =========================
def main():
    inject_css()

    # Sidebar minimal
    with st.sidebar:
        if os.path.exists(AppConfig.LOGO_PATH):
            st.image(AppConfig.LOGO_PATH, use_container_width=True)
        else:
            st.markdown("<div style='text-align:center; font-size:48px;'>🤖</div>", unsafe_allow_html=True)
        st.markdown("---")
        st.markdown('<div class="sidebar-info"><b>Kho tri thức:</b> 6 file PDF (đưa vào thư mục <code>PDF_KNOWLEDGE</code>)</div>', unsafe_allow_html=True)
        st.markdown("")

    st.markdown("<div class='header'><h2 style='margin:0'>🎓 TRỢ LÝ ẢO KTC - RAG</h2><div style='opacity:0.9'>Tìm câu trả lời từ 6 file PDF (tối ưu)</div></div>", unsafe_allow_html=True)
    st.write("")  # spacing

    # session state for messages
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role":"assistant","content":"Chào! Tôi sẽ tra cứu trong kho 6 PDF. Hãy nhập câu hỏi của bạn."}
        ]

    # Load models/resources (cached)
    groq_client = get_groq_client()
    translator = load_translator()
    embeddings = load_embedding_model()

    # Load docs (cached) and build/load vector DB (cached resource)
    docs = read_all_pdfs(AppConfig.PDF_DIR)
    if docs and len(docs) < 6:
        st.warning(f"📌 Phát hiện {len(docs)} file PDF trong {AppConfig.PDF_DIR}. (Ưu tiên là 6 file).")
    if not docs:
        st.info("📁 Chưa có file PDF trong thư mục 'PDF_KNOWLEDGE'. Bật app vẫn hoạt động nhưng sẽ trả lời dựa trên kiến thức nền.")
    # Build or load faiss (resource cached)
    with st.spinner("🔧 Kiểm tra/khởi tạo index..."):
        vector_db = build_or_load_faiss(docs, embeddings)
    # Save to session so retrieval code uses same instance
    st.session_state.vector_db = vector_db

    # If groq not configured, stop (giữ luồng gốc)
    if not groq_client:
        st.warning("⚠️ GROQ API key chưa cấu hình trong secrets. Vui lòng thêm GROQ_API_KEY để kích hoạt LLM.")
        st.stop()

    # Display chat history
    for m in st.session_state.messages:
        try:
            with st.chat_message(m["role"]):
                st.markdown(m["content"])
        except Exception:
            st.write(f"{m['role']}: {m['content']}")

    # Chat input
    prompt = st.chat_input("Nhập câu hỏi của bạn...")
    if prompt:
        # append user message
        st.session_state.messages.append({"role":"user","content":prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # assistant loading message
        with st.chat_message("assistant"):
            placeholder = st.empty()
            full = ""
            # Translate (if translator available)
            search_query = prompt
            if translator:
                try:
                    translated = translate_query(prompt, translator)
                    if translated and translated != prompt:
                        search_query = translated
                except Exception:
                    search_query = prompt

            # Retrieve context from vector DB
            context_text, sources = retrieve_from_db(st.session_state.get("vector_db"), search_query, top_k=AppConfig.TOP_K)
            if not context_text:
                context_text = ""  # let LLM answer from background knowledge
            # Generate stream response
            stream = generate_stream_response(groq_client, context_text, prompt)
            if isinstance(stream, str):
                full = stream
                placeholder.markdown(full)
            else:
                try:
                    for frag in _safe_stream_iter(stream):
                        full += frag
                        placeholder.markdown(full + "▌")
                    placeholder.markdown(full)
                except Exception as e:
                    placeholder.markdown(f"❌ Lỗi khi lấy kết quả: {e}")

            # show sources if any (collapsed)
            if sources:
                with st.expander("📖 Nguồn tham khảo"):
                    for s in sources:
                        st.markdown(f"- {s}")

            # Save assistant response to history
            st.session_state.messages.append({"role":"assistant","content":full})

if __name__ == "__main__":
    main()

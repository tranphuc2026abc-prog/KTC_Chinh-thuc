# ============================================
# KTC Assistant – RAG Chatbot (UI tối ưu mới)
# ============================================

import os
import glob
from typing import List, Tuple, Any, Generator
import streamlit as st

# --------- Import kiểm soát lỗi -----------
try:
    from pypdf import PdfReader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_community.vectorstores import FAISS
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_core.documents import Document
    from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
    from groq import Groq
except ImportError as e:
    st.error(f"❌ Thiếu thư viện: {e}. Hãy chạy: pip install -r requirements.txt")
    st.stop()

# --------- Cài đặt chung -----------
st.set_page_config(page_title="KTC Assistant", page_icon="🤖", layout="wide")

class AppConfig:
    PDF_DIR = "PDF_KNOWLEDGE"
    VECTOR_DB = "faiss_db_index"
    LOGO_PATH = "LOGO.jpg"

    EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    TRANSLATION_MODEL = "Helsinki-NLP/opus-mt-vi-en"
    LLM_MODEL = "llama-3.1-8b-instant"

    CHUNK = 1000
    OVERLAP = 200
    TOP_K = 5


# ============================================
#  UI / CSS – giao diện theo phong cách ChatGPT
# ============================================
def inject_css():
    st.markdown("""
    <style>

    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

    html, body, .css-18e3th9, .css-1d391kg {
        font-family: 'Inter', sans-serif !important;
    }

    /* Header */
    .main-header {
        background: linear-gradient(90deg, #1640F0, #4CB0FF);
        padding: 20px;
        border-radius: 14px;
        color: white;
        margin-bottom: 18px;
        box-shadow: 0px 4px 14px rgba(0,0,0,0.12);
    }

    /* Chat bubble style */
    .chat-bubble-user {
        background: #DCF2FF;
        padding: 12px 16px;
        border-radius: 12px;
        margin-bottom: 8px;
        width: fit-content;
        max-width: 80%;
        animation: fadeIn 0.25s ease;
    }

    .chat-bubble-assistant {
        background: #FFFFFF;
        padding: 12px 16px;
        border-radius: 12px;
        margin-bottom: 8px;
        width: fit-content;
        max-width: 80%;
        border-left: 4px solid #4CB0FF;
        box-shadow: 0px 2px 6px rgba(0,0,0,0.05);
        animation: fadeIn 0.25s ease;
    }

    /* Animation */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(4px); }
        to { opacity: 1; transform: translateY(0); }
    }

    /* Sidebar card */
    .sb-card {
        background: white;
        padding: 14px;
        border-radius: 10px;
        border-left: 5px solid #1640F0;
    }

    /* Chat input area */
    .stChatInput > div > div textarea {
        border-radius: 10px !important;
        padding: 14px !important;
        font-size: 16px !important;
    }

    </style>
    """, unsafe_allow_html=True)


# ============================================
#  Tải mô hình (cache)
# ============================================

@st.cache_resource
def get_client():
    key = st.secrets.get("GROQ_API_KEY")
    return Groq(api_key=key) if key else None

@st.cache_resource
def embed_model():
    return HuggingFaceEmbeddings(model_name=AppConfig.EMBEDDING_MODEL)

@st.cache_resource
def load_translator():
    tokenizer = AutoTokenizer.from_pretrained(AppConfig.TRANSLATION_MODEL)
    model = AutoModelForSeq2SeqLM.from_pretrained(AppConfig.TRANSLATION_MODEL)
    return pipeline("translation", model=model, tokenizer=tokenizer, src_lang="vi", tgt_lang="en")

@st.cache_data
def read_pdfs(folder: str):
    docs = []
    if not os.path.exists(folder):
        return docs

    for path in sorted(glob.glob(folder + "/*.pdf")):
        reader = PdfReader(path)
        name = os.path.basename(path)
        for i, p in enumerate(reader.pages):
            text = p.extract_text() or ""
            if text.strip():
                docs.append(Document(page_content=text, metadata={"source": name, "page": i+1}))
    return docs

@st.cache_resource
def build_db(docs, embeddings):
    if not docs:
        return None

    if os.path.exists(AppConfig.VECTOR_DB):
        try:
            return FAISS.load_local(AppConfig.VECTOR_DB, embeddings, allow_dangerous_deserialization=True)
        except:
            pass

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=AppConfig.CHUNK,
        chunk_overlap=AppConfig.OVERLAP
    )
    chunks = splitter.split_documents(docs)

    db = FAISS.from_documents(chunks, embeddings)
    db.save_local(AppConfig.VECTOR_DB)
    return db


# ============================================
#  Xử lý truy vấn
# ============================================

def translate(text, translator):
    try:
        return translator(text[:500])[0]["translation_text"]
    except:
        return text

def retrieve(db, query):
    if not db:
        return "", []

    results = db.similarity_search(query, k=AppConfig.TOP_K)
    parts, src_list = [], []

    for d in results:
        parts.append(f"[Nguồn: {d.metadata['source']} – Trang {d.metadata['page']}]\n{d.page_content}")
        src_list.append(f"{d.metadata['source']} (Trang {d.metadata['page']})")

    uniq = list(dict.fromkeys(src_list))
    return "\n\n".join(parts), uniq


def stream_answer(client, ctx, question):
    system = f"""
Bạn là KTC Assistant – trợ lý giáo dục.
Ưu tiên dùng dữ liệu từ CONTEXT, sau đó mới tới kiến thức nền.

[CONTEXT]:
{ctx}
"""
    return client.chat.completions.create(
        model=AppConfig.LLM_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": question}
        ],
        stream=True,
        temperature=0.2
    )


def safe_stream(gen):
    for chunk in gen:
        try:
            delta = chunk["choices"][0]["delta"].get("content", "")
            if delta:
                yield delta
        except:
            pass


# ============================================
#  MAIN UI
# ============================================
def main():
    inject_css()

    # Sidebar
    with st.sidebar:
        if os.path.exists(AppConfig.LOGO_PATH):
            st.image(AppConfig.LOGO_PATH, use_container_width=True)
        st.markdown("<div class='sb-card'><b>Kho tri thức:</b> 6 file PDF trong thư mục <code>PDF_KNOWLEDGE</code></div>", unsafe_allow_html=True)

    # Header
    st.markdown("""
        <div class="main-header">
            <h2 style="margin:0;">🤖 KTC Assistant – RAG Chatbot</h2>
            <div style="opacity:0.9">Trí tuệ nhân tạo hỗ trợ tra cứu kiến thức từ PDF</div>
        </div>
    """, unsafe_allow_html=True)

    # Session state
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Load resources
    client = get_client()
    translator = load_translator()
    embeddings = embed_model()

    docs = read_pdfs(AppConfig.PDF_DIR)
    db = build_db(docs, embeddings)

    # Hiển thị lịch sử chat
    for m in st.session_state.messages:
        role = m["role"]
        bubble = "chat-bubble-user" if role == "user" else "chat-bubble-assistant"
        st.markdown(f"<div class='{bubble}'>{m['content']}</div>", unsafe_allow_html=True)

    # Nhập câu hỏi
    question = st.chat_input("Nhập câu hỏi...")

    if question:

        # Lưu câu hỏi
        st.session_state.messages.append({"role": "user", "content": question})
        st.markdown(f"<div class='chat-bubble-user'>{question}</div>", unsafe_allow_html=True)

        # Dịch truy vấn trước khi search
        q_trans = translate(question, translator)

        ctx, src = retrieve(db, q_trans)

        # Stream câu trả lời
        with st.chat_message("assistant"):
            placeholder = st.empty()
            full = ""

            stream = stream_answer(client, ctx, question)

            for t in safe_stream(stream):
                full += t
                placeholder.markdown(f"<div class='chat-bubble-assistant'>{full}▌</div>", unsafe_allow_html=True)

            placeholder.markdown(f"<div class='chat-bubble-assistant'>{full}</div>", unsafe_allow_html=True)

        st.session_state.messages.append({"role":"assistant","content":full})

        if src:
            with st.expander("📚 Nguồn tham khảo"):
                for s in src:
                    st.write("- " + s)


if __name__ == "__main__":
    main()

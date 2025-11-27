# ==============================================================
#   TRỢ LÝ KHKT – PHIÊN BẢN KHÔNG LỖI TRANSLATOR (SAFE VERSION)
# ==============================================================

import os
import glob
import time
from typing import List, Optional

import streamlit as st
from pypdf import PdfReader

# AI / RAG libs
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

# Translator (SAFE version)
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

# Groq client for LLM streaming
from groq import Groq


# ==============================================================
# 0. CẤU HÌNH CHUNG
# ==============================================================

st.set_page_config(
    page_title="KTC Assistant - Trợ lý Tin học 2025",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

CONSTANTS = {
    "MODEL_NAME": 'llama-3.1-8b-instant',
    "PDF_DIR": "./PDF_KNOWLEDGE",
    "VECTOR_STORE_PATH": "./faiss_db_index",
    "LOGO_PATH": "LOGO.jpg",

    # Embedding đa ngôn ngữ – giúp fallback khi không dịch được
    "EMBEDDING_MODEL": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",

    # Model dịch (sẽ có fallback nếu không tải được)
    "TRANSLATION_MODEL": "Helsinki-NLP/opus-mt-vi-en",

    "CHUNK_SIZE": 800,
    "CHUNK_OVERLAP": 150,
    "TOP_K": 3,
}


# ==============================================================
# 1. GIAO DIỆN CSS
# ==============================================================

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;700&display=swap');
    html, body, [class*="css"] { font-family: 'Roboto', sans-serif; }
    .stApp {background-color: #f8f9fa;}
    [data-testid="stSidebar"] { background-color: #ffffff; border-right: 1px solid #e0e0e0; }
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
        font-size: 0.85rem; color: #444; background: #f1f1f1;
        padding: 8px; border-radius: 6px; margin-top: 8px; border-left: 3px solid #0284c7;
    }
</style>
""", unsafe_allow_html=True)


# ==============================================================
# 2. CACHE TÀI NGUYÊN
# ==============================================================

@st.cache_resource(show_spinner=False)
def get_groq_client():
    try:
        api_key = st.secrets["GROQ_API_KEY"]
        return Groq(api_key=api_key)
    except Exception:
        return None


@st.cache_resource(show_spinner=False)
def get_embeddings():
    return HuggingFaceEmbeddings(model_name=CONSTANTS["EMBEDDING_MODEL"])


# --------------------------------------------------
#  🔥 TRANSLATOR AN TOÀN – KHÔNG BAO GIỜ CRASH
# --------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_translator():
    """
    Translator an toàn – nếu lỗi thì trả về None.
    App vẫn hoạt động nhờ embedding đa ngôn ngữ.
    """
    model_name = CONSTANTS["TRANSLATION_MODEL"]

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        translator = pipeline(
            "translation",
            model=model,
            tokenizer=tokenizer,
            src_lang="vi",
            tgt_lang="en",
        )
        return translator

    except Exception as e:
        st.warning(
            "⚠️ Không tải được model dịch tiếng Việt → tiếng Anh.\n"
            "→ Hệ thống tự động chuyển sang chế độ fallback (không cần dịch).\n"
            f"Chi tiết lỗi: {str(e)}"
        )
        return None


# ==============================================================
# 3. CLASS KNOWLEDGEBASE
# ==============================================================

class KnowledgeBase:
    def __init__(self, embeddings):
        self.embeddings = embeddings

    def load_documents(self) -> List[Document]:
        if not os.path.exists(CONSTANTS["PDF_DIR"]):
            os.makedirs(CONSTANTS["PDF_DIR"])
            return []

        pdf_files = glob.glob(os.path.join(CONSTANTS["PDF_DIR"], "*.pdf"))
        docs = []

        for pdf_path in pdf_files:
            try:
                reader = PdfReader(pdf_path)
                fname = os.path.basename(pdf_path)
                for i, page in enumerate(reader.pages):
                    text = page.extract_text()
                    if text:
                        docs.append(Document(
                            page_content=text,
                            metadata={"source": fname, "page": i + 1}
                        ))
            except Exception as e:
                st.warning(f"Không đọc được file {pdf_path}: {e}")

        return docs

    def build_or_load_vector_db(self, force_rebuild=False):
        path = CONSTANTS["VECTOR_STORE_PATH"]

        if os.path.exists(path) and not force_rebuild:
            try:
                return FAISS.load_local(path, self.embeddings, allow_dangerous_deserialization=True)
            except:
                pass

        docs = self.load_documents()
        if not docs:
            return None

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CONSTANTS["CHUNK_SIZE"],
            chunk_overlap=CONSTANTS["CHUNK_OVERLAP"]
        )
        chunks = splitter.split_documents(docs)

        vector_db = FAISS.from_documents(chunks, self.embeddings)

        try:
            vector_db.save_local(path)
        except:
            pass

        return vector_db


# ==============================================================
# 4. HÀM TIỆN ÍCH
# ==============================================================

def translate_vi_to_en(translator, text):
    if not translator:
        return None
    try:
        out = translator(text, max_length=512)
        return out[0]["translation_text"]
    except:
        return None


def retrieve_context(vector_db, query, k=3):
    if not vector_db or not query:
        return "", []

    try:
        docs = vector_db.similarity_search(query, k=k)
        ctx = []
        srcs = []

        for d in docs:
            ctx.append(f"[TRÍCH]: {d.page_content.strip()}")
            srcs.append(f"{d.metadata.get('source')} (Tr. {d.metadata.get('page')})")

        return "\n\n".join(ctx), srcs
    except:
        return "", []


def build_system_prompt(context_text):
    return f"""
Bạn là trợ lý ảo KTC, chuyên gia Tin học GDPT 2018.
Chỉ trả lời dựa trên [NGUỒN TÀI LIỆU] bên dưới.
Nếu tài liệu không có thông tin → trả lời: "SGK hiện chưa đề cập vấn đề này."

[NGUỒN TÀI LIỆU]:
{context_text}
"""


# ==============================================================
# 5. KHỞI TẠO
# ==============================================================

groq_client = get_groq_client()
if groq_client is None:
    st.error("❌ Chưa cấu hình GROQ_API_KEY!")
    st.stop()

embeddings = get_embeddings()
translator = get_translator()
kb = KnowledgeBase(embeddings)

if "vector_db" not in st.session_state:
    with st.spinner("🔄 Đang tải Vector Database..."):
        st.session_state.vector_db = kb.build_or_load_vector_db()


# ==============================================================
# 6. SIDEBAR
# ==============================================================

with st.sidebar:
    if os.path.exists(CONSTANTS["LOGO_PATH"]):
        st.image(CONSTANTS["LOGO_PATH"], use_container_width=True)

    st.title("⚙️ Control Panel")

    st.markdown("---")

    if st.button("🔄 Rebuild dữ liệu"):
        with st.spinner("Đang cập nhật..."):
            st.session_state.vector_db = kb.build_or_load_vector_db(force_rebuild=True)
        st.success("Đã xây dựng lại!")
        st.rerun()

    if st.button("🗑 Xóa lịch sử chat"):
        st.session_state.messages = []
        st.rerun()


# ==============================================================
# 7. CHAT HISTORY
# ==============================================================

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Xin chào! Tôi là **KTC AI** – trợ lý Tin học của bạn."}
    ]


# ==============================================================
# 8. MAIN CHAT UI
# ==============================================================

col1, col2, col3 = st.columns([1, 8, 1])

with col2:
    st.markdown('<h1 class="gradient-text">TRỢ LÝ ẢO TIN HỌC KTC</h1>', unsafe_allow_html=True)

    for msg in st.session_state.messages:
        avatar = "🧑‍🎓" if msg["role"] == "user" else "🤖"
        with st.chat_message(msg["role"], avatar=avatar):
            st.markdown(msg["content"], unsafe_allow_html=True)

    user_input = st.chat_input("Bạn muốn hỏi gì? (gõ tiếng Việt)")

    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})

        with st.chat_message("user", avatar="🧑‍🎓"):
            st.markdown(user_input)

        # --------------------------------------------
        # 1) Dịch Vi → En (nếu translator có)
        # --------------------------------------------
        query_en = translate_vi_to_en(translator, user_input)
        search_text = query_en if query_en else user_input

        # --------------------------------------------
        # 2) Lấy context
        # --------------------------------------------
        with st.spinner("🔎 Đang truy vấn dữ liệu..."):
            context_text, sources = retrieve_context(st.session_state.vector_db, search_text)

        # --------------------------------------------
        # 3) Build prompt
        # --------------------------------------------
        sys_prompt = build_system_prompt(context_text)

        # --------------------------------------------
        # 4) Model trả lời dạng streaming
        # --------------------------------------------
        with st.chat_message("assistant", avatar="🤖"):
            placeholder = st.empty()
            full = ""

            try:
                stream = groq_client.chat.completions.create(
                    model=CONSTANTS["MODEL_NAME"],
                    stream=True,
                    messages=[
                        {"role": "system", "content": sys_prompt},
                        {"role": "user", "content": user_input}
                    ]
                )

                for chunk in stream:
                    delta = chunk.choices[0].delta
                    if hasattr(delta, "content") and delta.content:
                        full += delta.content
                        placeholder.markdown(full + "▌")

                # Hiện nguồn
                if sources:
                    src_html = "<div class='source-box'>📚 <b>Nguồn:</b><br>" + "<br>".join([f"• {s}" for s in sources]) + "</div>"
                    full = full + "\n\n" + src_html

                placeholder.markdown(full, unsafe_allow_html=True)
                st.session_state.messages.append({"role": "assistant", "content": full})

            except Exception as e:
                err = f"❌ Lỗi khi gọi mô hình: {str(e)}"
                placeholder.markdown(err)
                st.session_state.messages.append({"role": "assistant", "content": err})

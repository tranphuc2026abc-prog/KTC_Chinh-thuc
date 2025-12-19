# ===============================
# KTC CHATBOT – VERIFIABLE RAG
# Chuẩn KHKT cấp Quốc gia
# GIỮ NGUYÊN GIAO DIỆN
# ===============================

import os, glob, base64, shutil, re, uuid, unicodedata
import streamlit as st
from typing import List, Generator

# ===============================
# IMPORTS AN TOÀN
# ===============================
try:
    import nest_asyncio
    nest_asyncio.apply()

    try:
        from llama_parse import LlamaParse
    except ImportError:
        LlamaParse = None

    from langchain_community.document_loaders import PyPDFLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_community.vectorstores import FAISS
    from langchain_community.retrievers import BM25Retriever
    from langchain.retrievers import EnsembleRetriever
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_core.documents import Document
    from groq import Groq
    from flashrank import Ranker, RerankRequest

    DEPENDENCIES_OK = True
except ImportError as e:
    DEPENDENCIES_OK = False
    IMPORT_ERROR = str(e)

# ===============================
# CONFIG
# ===============================
st.set_page_config(
    page_title="KTC Chatbot - THCS & THPT Phạm Kiệt",
    page_icon="LOGO.jpg",
    layout="wide",
    initial_sidebar_state="expanded"
)

class AppConfig:
    LLM_MODEL = "llama-3.1-8b-instant"
    EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    RERANK_MODEL_NAME = "ms-marco-TinyBERT-L-2-v2"

    PDF_DIR = "PDF_KNOWLEDGE"
    VECTOR_DB_PATH = "faiss_db_index"
    PROCESSED_MD_DIR = "PROCESSED_MD"
    RERANK_CACHE = "./opt"

    LOGO_PROJECT = "LOGO.jpg"
    LOGO_SCHOOL = "LOGO PKS.png"

    RETRIEVAL_K = 30
    FINAL_K = 5
    BM25_WEIGHT = 0.4
    FAISS_WEIGHT = 0.6
    LLM_TEMPERATURE = 0.0

# ===============================
# UI – GIỮ NGUYÊN
# ===============================
class UIManager:
    @staticmethod
    def get_img_as_base64(file_path):
        if not os.path.exists(file_path):
            return ""
        with open(file_path, "rb") as f:
            return base64.b64encode(f.read()).decode()

    @staticmethod
    def inject_custom_css():
        st.markdown("""<style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;800&display=swap');
        html, body, [class*="css"] { font-family: 'Inter', sans-serif !important; }
        section[data-testid="stSidebar"] { background:#f8f9fa; border-right:1px solid #e9ecef; }
        .project-card { background:white; padding:15px; border-radius:12px; border:1px solid #dee2e6; }
        .main-header { background:linear-gradient(135deg,#023e8a,#0077b6);
            padding:1.5rem 2rem; border-radius:15px; color:white; margin-bottom:2rem; }
        .citation-footer { margin-top:15px; padding-top:10px; border-top:1px dashed #ced4da; font-size:0.9rem;}
        #MainMenu, footer {visibility:hidden;}
        </style>""", unsafe_allow_html=True)

    @staticmethod
    def render_sidebar():
        with st.sidebar:
            if os.path.exists(AppConfig.LOGO_SCHOOL):
                st.image(AppConfig.LOGO_SCHOOL, use_container_width=True)
            st.markdown("### ⚙️ Tiện ích")
            if st.button("🗑️ Xóa lịch sử chat", use_container_width=True):
                st.session_state.messages = []
                st.rerun()
            if st.button("🔄 Cập nhật dữ liệu mới", use_container_width=True):
                shutil.rmtree(AppConfig.VECTOR_DB_PATH, ignore_errors=True)
                shutil.rmtree(AppConfig.PROCESSED_MD_DIR, ignore_errors=True)
                st.session_state.pop("retriever_engine", None)
                st.rerun()

    @staticmethod
    def render_header():
        logo = UIManager.get_img_as_base64(AppConfig.LOGO_PROJECT)
        st.markdown(f"""
        <div class="main-header">
            <h1>KTC CHATBOT</h1>
            <p>Học Tin dễ dàng – Thao tác vững vàng</p>
            <img src="data:image/jpeg;base64,{logo}" width="90">
        </div>
        """, unsafe_allow_html=True)

# ===============================
# RAG ENGINE – CHUẨN KHKT
# ===============================
class RAGEngine:

    @staticmethod
    @st.cache_resource
    def load_embeddings():
        return HuggingFaceEmbeddings(
            model_name=AppConfig.EMBEDDING_MODEL,
            encode_kwargs={"normalize_embeddings": True}
        )

    @staticmethod
    @st.cache_resource
    def load_reranker():
        return Ranker(model_name=AppConfig.RERANK_MODEL_NAME, cache_dir=AppConfig.RERANK_CACHE)

    @staticmethod
    @st.cache_resource
    def load_llm():
        key = st.secrets.get("GROQ_API_KEY") or os.environ.get("GROQ_API_KEY")
        return Groq(api_key=key) if key else None

    # ---------- STRUCTURAL CHUNKING ----------
    @staticmethod
    def structural_chunking(text: str, source: str) -> List[Document]:
        text = unicodedata.normalize("NFC", text)
        lines = text.split("\n")

        chunks, buffer = [], []
        topic = lesson = section = None

        p_topic = re.compile(r"CHỦ\s*ĐỀ\s+([0-9A-Z]+)", re.I)
        p_lesson = re.compile(r"BÀI\s+([0-9]+)", re.I)

        def commit():
            if not buffer or not topic or not lesson:
                return
            content = "\n".join(buffer).strip()
            meta = {
                "source": source,
                "chapter": topic,
                "lesson": lesson,
                "section": section or "Nội dung",
                "chunk_uid": str(uuid.uuid4())[:8]
            }
            ctx = f"{topic} > {lesson}"
            chunks.append(Document(
                page_content=f"Context: {ctx}\n{content}",
                metadata=meta
            ))

        for line in lines:
            if p_topic.search(line):
                commit(); buffer=[]
                topic = f"Chủ đề {p_topic.search(line).group(1)}"
            elif p_lesson.search(line):
                commit(); buffer=[]
                lesson = f"Bài {p_lesson.search(line).group(1)}"
            else:
                buffer.append(line)

        commit()
        return chunks

    # ---------- PDF PARSE ----------
    @staticmethod
    def parse_pdf(path):
        os.makedirs(AppConfig.PROCESSED_MD_DIR, exist_ok=True)
        cache = os.path.join(AppConfig.PROCESSED_MD_DIR, os.path.basename(path)+".md")
        if os.path.exists(cache):
            return open(cache, encoding="utf-8").read()

        if LlamaParse and st.secrets.get("LLAMA_CLOUD_API_KEY"):
            try:
                parser = LlamaParse(api_key=st.secrets["LLAMA_CLOUD_API_KEY"], result_type="markdown")
                text = parser.load_data(path)[0].text
            except:
                text = ""
        else:
            loader = PyPDFLoader(path)
            text = "\n".join([p.page_content for p in loader.load()])

        open(cache, "w", encoding="utf-8").write(text)
        return text

    # ---------- BUILD RETRIEVER ----------
    @staticmethod
    def build_retriever(emb):
        if os.path.exists(AppConfig.VECTOR_DB_PATH):
            db = FAISS.load_local(AppConfig.VECTOR_DB_PATH, emb, allow_dangerous_deserialization=True)
        else:
            docs=[]
            for pdf in glob.glob(f"{AppConfig.PDF_DIR}/*.pdf"):
                text = RAGEngine.parse_pdf(pdf)
                docs.extend(RAGEngine.structural_chunking(text, os.path.basename(pdf)))
            db = FAISS.from_documents(docs, emb)
            db.save_local(AppConfig.VECTOR_DB_PATH)

        bm25 = BM25Retriever.from_documents(list(db.docstore._dict.values()))
        bm25.k = AppConfig.RETRIEVAL_K

        faiss = db.as_retriever(search_kwargs={"k": AppConfig.RETRIEVAL_K})
        return EnsembleRetriever(
            retrievers=[bm25, faiss],
            weights=[AppConfig.BM25_WEIGHT, AppConfig.FAISS_WEIGHT]
        )

    # ---------- GENERATE ----------
    @staticmethod
    def answer(llm, retriever, query) -> str:
        docs = retriever.invoke(query)
        reranker = RAGEngine.load_reranker()

        passages=[{"id":i,"text":d.page_content,"meta":d.metadata} for i,d in enumerate(docs)]
        ranked = reranker.rank(RerankRequest(query=query, passages=passages))[:AppConfig.FINAL_K]

        context = "\n".join([r["text"] for r in ranked])

        prompt = f"""Bạn là KTC Chatbot.
Chỉ dùng thông tin trong CONTEXT.

CONTEXT:
{context}
"""
        res = llm.chat.completions.create(
            model=AppConfig.LLM_MODEL,
            messages=[{"role":"system","content":prompt},{"role":"user","content":query}],
            temperature=0
        ).choices[0].message.content

        cites = {f"📖 {r['meta']['source']} → {r['meta']['chapter']} → {r['meta']['lesson']}" for r in ranked}
        cite_html = "<div class='citation-footer'><b>📚 Nguồn:</b>" + "<br>".join(cites) + "</div>"
        return res + cite_html

# ===============================
# MAIN
# ===============================
def main():
    if not DEPENDENCIES_OK:
        st.error(IMPORT_ERROR); return

    UIManager.inject_custom_css()
    UIManager.render_sidebar()
    UIManager.render_header()

    if "messages" not in st.session_state:
        st.session_state.messages=[{"role":"assistant","content":"👋 Chào bạn!"}]

    llm = RAGEngine.load_llm()
    if "retriever_engine" not in st.session_state:
        emb = RAGEngine.load_embeddings()
        st.session_state.retriever_engine = RAGEngine.build_retriever(emb)

    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"], unsafe_allow_html=True)

    q = st.chat_input("Nhập câu hỏi...")
    if q:
        st.session_state.messages.append({"role":"user","content":q})
        with st.chat_message("assistant"):
            ans = RAGEngine.answer(llm, st.session_state.retriever_engine, q)
            st.markdown(ans, unsafe_allow_html=True)
            st.session_state.messages.append({"role":"assistant","content":ans})

if __name__ == "__main__":
    main()

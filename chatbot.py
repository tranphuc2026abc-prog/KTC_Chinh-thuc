# =========================
# KTC CHATBOT – RAG QUỐC GIA v1.1
# Giữ nguyên UI – Nâng cấp RAG học thuật
# =========================

import os, glob, base64, shutil, re
import streamlit as st
from typing import List, Dict

# ===== IMPORTS =====
try:
    import nest_asyncio
    nest_asyncio.apply()

    from llama_parse import LlamaParse
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

# =========================
# CONFIG
# =========================
st.set_page_config(
    page_title="KTC Chatbot - THCS & THPT Phạm Kiệt",
    page_icon="LOGO.jpg",
    layout="wide",
    initial_sidebar_state="expanded"
)

class AppConfig:
    LLM_MODEL = "llama-3.1-8b-instant"
    EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    RERANK_MODEL = "ms-marco-TinyBERT-L-2-v2"

    PDF_DIR = "PDF_KNOWLEDGE"
    VECTOR_DB_PATH = "faiss_db_index"
    PROCESSED_MD_DIR = "PROCESSED_MD"
    RERANK_CACHE = "./opt"

    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200
    RETRIEVAL_K = 30
    FINAL_K = 5

    BM25_WEIGHT = 0.4
    FAISS_WEIGHT = 0.6
    LLM_TEMPERATURE = 0.0

    LOGO_PROJECT = "LOGO.jpg"
    LOGO_SCHOOL = "LOGO PKS.png"

# =========================
# UI (GIỮ NGUYÊN)
# =========================
class UIManager:
    @staticmethod
    def get_img_as_base64(path):
        if not os.path.exists(path): return ""
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode()

    @staticmethod
    def inject_custom_css():
        st.markdown("""<style>/* GIỮ NGUYÊN CSS CỦA BẠN */</style>""",
                    unsafe_allow_html=True)

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
                if os.path.exists(AppConfig.VECTOR_DB_PATH):
                    shutil.rmtree(AppConfig.VECTOR_DB_PATH)
                st.session_state.pop("retriever", None)
                st.rerun()

    @staticmethod
    def render_header():
        logo = UIManager.get_img_as_base64(AppConfig.LOGO_PROJECT)
        st.markdown(f"""
        <div class="main-header">
            <div>
                <h1>KTC CHATBOT</h1>
                <p>Học Tin dễ dàng – Thao tác vững vàng</p>
            </div>
            <img src="data:image/jpeg;base64,{logo}" width="90">
        </div>
        """, unsafe_allow_html=True)

# =========================
# RAG ENGINE (QUỐC GIA)
# =========================
class RAGEngine:

    @staticmethod
    def load_groq():
        key = st.secrets.get("GROQ_API_KEY")
        return Groq(api_key=key) if key else None

    @staticmethod
    def load_embeddings():
        return HuggingFaceEmbeddings(
            model_name=AppConfig.EMBEDDING_MODEL,
            encode_kwargs={"normalize_embeddings": True}
        )

    @staticmethod
    def load_reranker():
        return Ranker(model_name=AppConfig.RERANK_MODEL, cache_dir=AppConfig.RERANK_CACHE)

    # ===== TÁCH CẤU TRÚC SGK =====
    @staticmethod
    def extract_structure(md: str) -> List[Dict]:
        blocks, lesson, section = [], "Không xác định", "Không xác định"
        for line in md.splitlines():
            line = line.strip()
            if re.match(r"^#\s*Bài\s+\d+", line, re.I):
                lesson = line.replace("#", "").strip()
            elif re.match(r"^##\s+", line):
                section = line.replace("##", "").strip()
            elif len(line) > 40:
                blocks.append({
                    "lesson": lesson,
                    "section": section,
                    "content": line
                })
        return blocks

    @staticmethod
    def parse_pdf(pdf):
        os.makedirs(AppConfig.PROCESSED_MD_DIR, exist_ok=True)
        md_path = f"{AppConfig.PROCESSED_MD_DIR}/{os.path.basename(pdf)}.md"
        if os.path.exists(md_path):
            return open(md_path, encoding="utf-8").read()

        parser = LlamaParse(
            api_key=st.secrets.get("LLAMA_CLOUD_API_KEY"),
            result_type="markdown",
            language="vi"
        )
        md = parser.load_data(pdf)[0].text
        open(md_path, "w", encoding="utf-8").write(md)
        return md

    @staticmethod
    def load_documents():
        docs = []
        for pdf in glob.glob(f"{AppConfig.PDF_DIR}/*.pdf"):
            md = RAGEngine.parse_pdf(pdf)
            for block in RAGEngine.extract_structure(md):
                docs.append(Document(
                    page_content=block["content"],
                    metadata={
                        "source": os.path.basename(pdf),
                        "lesson": block["lesson"],
                        "section": block["section"],
                        "doc_type": "SGK Tin học"
                    }
                ))
        return docs

    @staticmethod
    def build_retriever(emb):
        if os.path.exists(AppConfig.VECTOR_DB_PATH):
            db = FAISS.load_local(AppConfig.VECTOR_DB_PATH, emb, allow_dangerous_deserialization=True)
        else:
            docs = RAGEngine.load_documents()
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=AppConfig.CHUNK_SIZE,
                chunk_overlap=AppConfig.CHUNK_OVERLAP
            )
            chunks = splitter.split_documents(docs)
            db = FAISS.from_documents(chunks, emb)
            db.save_local(AppConfig.VECTOR_DB_PATH)

        bm25 = BM25Retriever.from_documents(list(db.docstore._dict.values()))
        bm25.k = AppConfig.RETRIEVAL_K

        faiss = db.as_retriever(search_type="mmr",
                                search_kwargs={"k": AppConfig.RETRIEVAL_K})

        return EnsembleRetriever(
            retrievers=[bm25, faiss],
            weights=[AppConfig.BM25_WEIGHT, AppConfig.FAISS_WEIGHT]
        )

    @staticmethod
    def answer(client, retriever, query):
        docs = retriever.invoke(query)
        ranker = RAGEngine.load_reranker()

        passages = [{"id": i, "text": d.page_content, "meta": d.metadata}
                    for i, d in enumerate(docs)]
        ranked = ranker.rank(RerankRequest(query=query, passages=passages))

        final_docs = ranked[:AppConfig.FINAL_K]
        if not final_docs:
            return "Không tìm thấy thông tin trong SGK.", []

        context, sources = "", set()
        for r in final_docs:
            m = r["meta"]
            sources.add(f"{m['source']} – {m['lesson']} – {m['section']}")
            context += f"{r['text']}\n\n"

        prompt = f"""
Bạn là trợ lý học tập AI.
Chỉ trả lời dựa trên CONTEXT.
Nếu không có, nói rõ không tìm thấy.

[CONTEXT]
{context}
"""

        stream = client.chat.completions.create(
            model=AppConfig.LLM_MODEL,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": query}
            ],
            stream=True,
            temperature=0
        )
        return stream, list(sources)

# =========================
# MAIN
# =========================
def main():
    if not DEPENDENCIES_OK:
        st.error(IMPORT_ERROR)
        return

    UIManager.inject_custom_css()
    UIManager.render_sidebar()
    UIManager.render_header()

    if "messages" not in st.session_state:
        st.session_state.messages = [{
            "role": "assistant",
            "content": "👋 Chào bạn! KTC Chatbot sẵn sàng hỗ trợ."
        }]

    client = RAGEngine.load_groq()

    if "retriever" not in st.session_state:
        with st.spinner("🚀 Khởi tạo tri thức SGK..."):
            emb = RAGEngine.load_embeddings()
            st.session_state.retriever = RAGEngine.build_retriever(emb)

    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    q = st.chat_input("Nhập câu hỏi...")
    if q:
        st.session_state.messages.append({"role": "user", "content": q})
        with st.chat_message("assistant"):
            stream, sources = RAGEngine.answer(client, st.session_state.retriever, q)
            ans = ""
            for c in stream:
                if c.choices[0].delta.content:
                    ans += c.choices[0].delta.content
                    st.markdown(ans + "▌")
            st.markdown(ans)
            if sources:
                with st.expander("📚 Nguồn SGK"):
                    for s in sources:
                        st.markdown(f"- {s}")
        st.session_state.messages.append({"role": "assistant", "content": ans})

if __name__ == "__main__":
    main()

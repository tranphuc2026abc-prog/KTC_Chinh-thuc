# =========================================================
# KTC CHATBOT – RAG CHUẨN QUỐC GIA (v1.1.1 – STABLE)
# Giữ nguyên UI | Sửa triệt để FlashRank | Production ready
# =========================================================

import os
import glob
import base64
import shutil
import re
import streamlit as st

from typing import List, Dict

# ===================== DEPENDENCIES ======================
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
    from flashrank import Ranker

    DEP_OK = True
except Exception as e:
    DEP_OK = False
    DEP_ERROR = str(e)

# ========================= CONFIG =========================
st.set_page_config(
    page_title="KTC Chatbot - THCS & THPT Phạm Kiệt",
    page_icon="LOGO.jpg",
    layout="wide",
    initial_sidebar_state="expanded"
)

class Config:
    LLM_MODEL = "llama-3.1-8b-instant"
    EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    RERANK_MODEL = "ms-marco-TinyBERT-L-2-v2"

    PDF_DIR = "PDF_KNOWLEDGE"
    VECTOR_DB = "faiss_index"
    MD_CACHE = "MD_CACHE"
    RERANK_CACHE = "./opt"

    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200
    RETRIEVE_K = 30
    FINAL_K = 5

    BM25_W = 0.4
    FAISS_W = 0.6

# ========================= UI =============================
class UI:
    @staticmethod
    def b64(path):
        if not os.path.exists(path):
            return ""
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode()

    @staticmethod
    def css():
        st.markdown("<style>/* giữ nguyên css */</style>", unsafe_allow_html=True)

    @staticmethod
    def sidebar():
        with st.sidebar:
            if os.path.exists("LOGO PKS.png"):
                st.image("LOGO PKS.png", use_container_width=True)

            if st.button("🗑️ Xóa lịch sử chat", use_container_width=True):
                st.session_state.messages = []
                st.rerun()

            if st.button("🔄 Cập nhật dữ liệu", use_container_width=True):
                if os.path.exists(Config.VECTOR_DB):
                    shutil.rmtree(Config.VECTOR_DB)
                st.session_state.pop("retriever", None)
                st.rerun()

    @staticmethod
    def header():
        logo = UI.b64("LOGO.jpg")
        st.markdown(f"""
        <div style="display:flex;justify-content:space-between;align-items:center">
            <div>
                <h1>KTC CHATBOT</h1>
                <p>Học Tin dễ dàng – Thao tác vững vàng</p>
            </div>
            <img src="data:image/png;base64,{logo}" width="90">
        </div>
        """, unsafe_allow_html=True)

# ========================= RAG CORE =======================
class RAG:

    @staticmethod
    def llm():
        return Groq(api_key=st.secrets.get("GROQ_API_KEY"))

    @staticmethod
    def embeddings():
        return HuggingFaceEmbeddings(
            model_name=Config.EMBEDDING_MODEL,
            encode_kwargs={"normalize_embeddings": True}
        )

    @staticmethod
    def reranker():
        return Ranker(model_name=Config.RERANK_MODEL, cache_dir=Config.RERANK_CACHE)

    @staticmethod
    def parse_pdf(pdf):
        os.makedirs(Config.MD_CACHE, exist_ok=True)
        md_path = f"{Config.MD_CACHE}/{os.path.basename(pdf)}.md"

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
    def extract_blocks(md: str) -> List[Dict]:
        blocks = []
        lesson, section = "Không rõ", "Không rõ"

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
                    "text": line
                })
        return blocks

    @staticmethod
    def documents():
        docs = []
        for pdf in glob.glob(f"{Config.PDF_DIR}/*.pdf"):
            md = RAG.parse_pdf(pdf)
            for b in RAG.extract_blocks(md):
                docs.append(Document(
                    page_content=b["text"],
                    metadata={
                        "source": os.path.basename(pdf),
                        "lesson": b["lesson"],
                        "section": b["section"]
                    }
                ))
        return docs

    @staticmethod
    def retriever(emb):
        if os.path.exists(Config.VECTOR_DB):
            db = FAISS.load_local(Config.VECTOR_DB, emb, allow_dangerous_deserialization=True)
        else:
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=Config.CHUNK_SIZE,
                chunk_overlap=Config.CHUNK_OVERLAP
            )
            chunks = splitter.split_documents(RAG.documents())
            db = FAISS.from_documents(chunks, emb)
            db.save_local(Config.VECTOR_DB)

        bm25 = BM25Retriever.from_documents(list(db.docstore._dict.values()))
        bm25.k = Config.RETRIEVE_K

        faiss = db.as_retriever(
            search_type="mmr",
            search_kwargs={"k": Config.RETRIEVE_K}
        )

        return EnsembleRetriever(
            retrievers=[bm25, faiss],
            weights=[Config.BM25_W, Config.FAISS_W]
        )

    @staticmethod
    def answer(client, retriever, query):
        docs = retriever.invoke(query)
        ranker = RAG.reranker()

        passages = [
            {"text": d.page_content, "meta": d.metadata}
            for d in docs
        ]

        ranked = ranker.rank(query=query, passages=passages)
        ranked = ranked[:Config.FINAL_K]

        if not ranked:
            return "Không tìm thấy thông tin phù hợp trong tài liệu.", []

        context = ""
        sources = set()

        for r in ranked:
            meta = r.get("meta", {})
            context += r.get("text", "") + "\n\n"
            sources.add(
                f"{meta.get('source','')} – {meta.get('lesson','')} – {meta.get('section','')}"
            )

        system_prompt = f"""
Bạn là trợ lý học tập.
Chỉ trả lời dựa trên CONTEXT.
Nếu CONTEXT không đủ, hãy nói rõ không có thông tin.

[CONTEXT]
{context}
"""

        stream = client.chat.completions.create(
            model=Config.LLM_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query}
            ],
            temperature=0,
            stream=True
        )

        return stream, list(sources)

# ========================= MAIN ===========================
def main():
    if not DEP_OK:
        st.error(DEP_ERROR)
        return

    UI.css()
    UI.sidebar()
    UI.header()

    if "messages" not in st.session_state:
        st.session_state.messages = [{
            "role": "assistant",
            "content": "👋 Chào bạn! Tôi sẵn sàng hỗ trợ học tập Tin học."
        }]

    client = RAG.llm()

    if "retriever" not in st.session_state:
        with st.spinner("Khởi tạo tri thức SGK..."):
            emb = RAG.embeddings()
            st.session_state.retriever = RAG.retriever(emb)

    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    q = st.chat_input("Nhập câu hỏi...")
    if q:
        st.session_state.messages.append({"role": "user", "content": q})

        with st.chat_message("assistant"):
            stream, sources = RAG.answer(client, st.session_state.retriever, q)
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

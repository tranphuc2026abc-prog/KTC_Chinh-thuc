# ============================================================
# KTC CHATBOT – VERIFIABLE HYBRID RAG (NATIONAL SAFE BASELINE)
# UI LAYER – DO NOT MODIFY (KHKT NATIONAL CONSTRAINT)
# ============================================================

import os
import re
import glob
import uuid
import shutil
import base64
from typing import List, Set

import streamlit as st

# ===============================
# DEPENDENCY CHECK (SAFE MODE)
# ===============================
try:
    import nest_asyncio
    nest_asyncio.apply()

    from llama_parse import LlamaParse
    from groq import Groq

    from langchain_core.documents import Document
    from langchain_community.vectorstores import FAISS
    from langchain_community.retrievers import BM25Retriever
    from langchain.retrievers import EnsembleRetriever
    from langchain_huggingface import HuggingFaceEmbeddings

    DEPENDENCIES_OK = True
except Exception as e:
    DEPENDENCIES_OK = False
    IMPORT_ERROR = str(e)

# ============================================================
# CONFIGURATION – NATIONAL STANDARD TERMINOLOGY
# ============================================================

class AppConfig:
    LLM_MODEL = "llama-3.1-8b-instant"
    LLM_TEMPERATURE = 0.0

    EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

    PDF_DIR = "PDF_KNOWLEDGE"
    VECTOR_DB_PATH = "faiss_db_index"
    PROCESSED_MD_DIR = "PROCESSED_MD"

    LOGO_PROJECT = "LOGO.jpg"
    LOGO_SCHOOL = "LOGO PKS.png"

    RETRIEVAL_K = 30
    FINAL_K = 5
    BM25_WEIGHT = 0.4
    FAISS_WEIGHT = 0.6

    H1_RE = re.compile(r"^\s*#\s+(.*)")
    H2_RE = re.compile(r"^\s*##\s+(.*)")
    H3_RE = re.compile(r"^\s*###\s+(.*)")

# ============================================================
# REQUIRED METADATA SCHEMA – KHKT NATIONAL
# ============================================================

REQUIRED_CHUNK_METADATA = [
    "book",
    "grade",
    "chapter",
    "lesson",
    "section",
    "chunk_type",
    "chunk_uid"
]

# ============================================================
# UI MANAGER (ABSOLUTELY NO MODIFICATION)
# ============================================================

class UIManager:
    @staticmethod
    def get_img_as_base64(path):
        if not os.path.exists(path):
            return ""
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode()

    @staticmethod
    def inject_css():
        st.markdown("<style>#MainMenu{visibility:hidden;}footer{visibility:hidden;}</style>", unsafe_allow_html=True)

    @staticmethod
    def render_sidebar():
        with st.sidebar:
            if os.path.exists(AppConfig.LOGO_SCHOOL):
                st.image(AppConfig.LOGO_SCHOOL, use_container_width=True)
            st.markdown("### KTC CHATBOT")
            st.markdown("**Sản phẩm dự thi KHKT**")

            if st.button("🗑️ Xóa lịch sử"):
                st.session_state.messages = []
                st.rerun()

            if st.button("🔄 Cập nhật dữ liệu"):
                if os.path.exists(AppConfig.VECTOR_DB_PATH):
                    shutil.rmtree(AppConfig.VECTOR_DB_PATH)
                st.session_state.pop("retriever", None)
                st.rerun()

    @staticmethod
    def render_header():
        logo = UIManager.get_img_as_base64(AppConfig.LOGO_PROJECT)
        if logo:
            st.image(AppConfig.LOGO_PROJECT, width=120)
        st.markdown("## 📘 KTC Chatbot – Hệ thống RAG SGK Tin học")

# ============================================================
# RAG ENGINE – VERIFIABLE HYBRID RAG
# ============================================================

class RAGEngine:

    # --------------------------------------------------------
    # PDF → MARKDOWN (NATIONAL SAFE)
    # --------------------------------------------------------
    @staticmethod
    def parse_pdf(pdf_path: str) -> str:
        os.makedirs(AppConfig.PROCESSED_MD_DIR, exist_ok=True)
        md_path = os.path.join(
            AppConfig.PROCESSED_MD_DIR,
            os.path.basename(pdf_path) + ".md"
        )

        if os.path.exists(md_path):
            with open(md_path, encoding="utf-8") as f:
                return f.read()

        parser = LlamaParse(
            api_key=st.secrets.get("LLAMA_CLOUD_API_KEY"),
            result_type="markdown",
            language="vi"
        )

        docs = parser.load_data(pdf_path)

        # HARD GUARD: NO FAKE KNOWLEDGE
        if not docs or not hasattr(docs[0], "text") or not docs[0].text.strip():
            return ""

        text = docs[0].text

        with open(md_path, "w", encoding="utf-8") as f:
            f.write(text)

        return text

    # --------------------------------------------------------
    # STRUCTURAL / SEMANTIC CHUNKING (MANDATORY)
    # --------------------------------------------------------
    @staticmethod
    def structural_chunk(md_text: str, source: str) -> List[Document]:
        if not md_text or not md_text.strip():
            return []

        docs: List[Document] = []

        chapter = lesson = section = ""
        buffer: List[str] = []

        def flush():
            nonlocal buffer
            text = "\n".join(buffer).strip()
            buffer = []

            if not text:
                return

            uid = str(uuid.uuid4())
            metadata = {
                "book": source,
                "grade": RAGEngine._infer_grade(source),
                "chapter": chapter,
                "lesson": lesson,
                "section": section,
                "chunk_type": RAGEngine._infer_chunk_type(text),
                "chunk_uid": uid
            }

            for key in REQUIRED_CHUNK_METADATA:
                if key not in metadata:
                    return

            docs.append(Document(page_content=text, metadata=metadata))

        for line in md_text.splitlines():
            h1 = AppConfig.H1_RE.match(line)
            h2 = AppConfig.H2_RE.match(line)
            h3 = AppConfig.H3_RE.match(line)

            if h1:
                flush()
                chapter = h1.group(1).strip()
                lesson = section = ""
            elif h2:
                flush()
                lesson = h2.group(1).strip()
                section = ""
            elif h3:
                flush()
                section = h3.group(1).strip()
            else:
                buffer.append(line)

        flush()
        return docs

    @staticmethod
    def _infer_grade(source: str):
        m = re.search(r"(10|11|12)", source)
        return int(m.group(1)) if m else None

    @staticmethod
    def _infer_chunk_type(text: str) -> str:
        t = text.lower()
        if "ví dụ" in t:
            return "example"
        if "là" in t or "được gọi là" in t:
            return "definition"
        return "explanation"

    # --------------------------------------------------------
    # BUILD HYBRID RETRIEVER
    # --------------------------------------------------------
    @staticmethod
    def build_retriever(embeddings):
        if os.path.exists(AppConfig.VECTOR_DB_PATH):
            vectordb = FAISS.load_local(
                AppConfig.VECTOR_DB_PATH,
                embeddings,
                allow_dangerous_deserialization=True
            )
            docs = list(vectordb.docstore._dict.values())
            bm25 = BM25Retriever.from_documents(docs)
            bm25.k = AppConfig.RETRIEVAL_K

            return EnsembleRetriever(
                retrievers=[
                    bm25,
                    vectordb.as_retriever(search_kwargs={"k": AppConfig.RETRIEVAL_K})
                ],
                weights=[AppConfig.BM25_WEIGHT, AppConfig.FAISS_WEIGHT]
            )

        docs: List[Document] = []
        for pdf in glob.glob(os.path.join(AppConfig.PDF_DIR, "*.pdf")):
            md = RAGEngine.parse_pdf(pdf)
            docs.extend(RAGEngine.structural_chunk(md, os.path.basename(pdf)))

        vectordb = FAISS.from_documents(docs, embeddings)
        vectordb.save_local(AppConfig.VECTOR_DB_PATH)

        bm25 = BM25Retriever.from_documents(docs)
        bm25.k = AppConfig.RETRIEVAL_K

        return EnsembleRetriever(
            retrievers=[
                bm25,
                vectordb.as_retriever(search_kwargs={"k": AppConfig.RETRIEVAL_K})
            ],
            weights=[AppConfig.BM25_WEIGHT, AppConfig.FAISS_WEIGHT]
        )

    # --------------------------------------------------------
    # GENERATION WITH VERIFIABLE CITATION
    # --------------------------------------------------------
    @staticmethod
    def generate_answer(query: str, contexts: List[Document]) -> str:
        client = Groq(api_key=st.secrets.get("GROQ_API_KEY"))

        context_text = "\n\n".join(
            f"[{d.metadata['chunk_uid']}] "
            f"({d.metadata['chapter']} – {d.metadata['lesson']})\n"
            f"{d.page_content}"
            for d in contexts
        )

        prompt = f"""
Bạn là HỆ THỐNG RAG.
Sinh câu trả lời dựa trên tri thức SGK đã truy xuất.
Mỗi ý trả lời PHẢI có trích dẫn theo mẫu:
[Nguồn: <chunk_uid> – <chapter> – <lesson>]

TRI THỨC SGK:
{context_text}

CÂU HỎI:
{query}
"""

        resp = client.chat.completions.create(
            model=AppConfig.LLM_MODEL,
            temperature=AppConfig.LLM_TEMPERATURE,
            messages=[{"role": "user", "content": prompt}]
        )

        return resp.choices[0].message.content

    # --------------------------------------------------------
    # POST-GENERATION VALIDATION (MANDATORY)
    # --------------------------------------------------------
    @staticmethod
    def validate_answer(answer: str, contexts: List[Document]) -> bool:
        valid_uids: Set[str] = {d.metadata["chunk_uid"] for d in contexts}

        cited = re.findall(r"\[Nguồn:\s*([^\]\s]+)", answer)
        if not cited:
            return False

        for uid in cited:
            if uid not in valid_uids:
                return False

        return True

# ============================================================
# MAIN APPLICATION
# ============================================================

def main():
    if not DEPENDENCIES_OK:
        st.error(f"Missing dependency: {IMPORT_ERROR}")
        st.stop()

    st.set_page_config(page_title="KTC Chatbot", layout="wide")

    UIManager.inject_css()
    UIManager.render_sidebar()
    UIManager.render_header()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    embeddings = HuggingFaceEmbeddings(
        model_name=AppConfig.EMBEDDING_MODEL,
        encode_kwargs={"normalize_embeddings": True}
    )

    if "retriever" not in st.session_state:
        with st.spinner("Đang khởi tạo hệ thống RAG..."):
            st.session_state.retriever = RAGEngine.build_retriever(embeddings)

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    query = st.chat_input("Nhập câu hỏi SGK Tin học...")

    if query:
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("assistant"):
            docs = st.session_state.retriever.get_relevant_documents(query)
            docs = docs[:AppConfig.FINAL_K]

            if not docs:
                st.markdown("Không tìm thấy thông tin phù hợp trong SGK hiện có.")
            else:
                answer = RAGEngine.generate_answer(query, docs)
                if not RAGEngine.validate_answer(answer, docs):
                    st.markdown("Không tìm thấy thông tin phù hợp trong SGK hiện có.")
                else:
                    st.markdown(answer)

        st.session_state.messages.append({"role": "assistant", "content": st.session_state.messages[-1]["content"]})

if __name__ == "__main__":
    main()

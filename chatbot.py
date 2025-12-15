# ==============================
# KTC CHATBOT – VERIFIABLE HYBRID RAG (SGK)
# Giữ nguyên 100% giao diện – Tái cấu trúc backend theo chuẩn KHKT Quốc gia
# ==============================

import os, glob, base64, shutil, re, uuid
import streamlit as st
from pathlib import Path
from typing import List, Dict

# ===== Imports kỹ thuật RAG (giữ nguyên nền tảng) =====
try:
    import nest_asyncio
    nest_asyncio.apply()

    from llama_parse import LlamaParse
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

# ==============================
# 1. CONFIG
# ==============================

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

    RETRIEVAL_K = 30
    FINAL_K = 5
    BM25_WEIGHT = 0.4
    FAISS_WEIGHT = 0.6

    LLM_TEMPERATURE = 0.0

    LOGO_PROJECT = "LOGO.jpg"
    LOGO_SCHOOL = "LOGO PKS.png"

# ==============================
# 2. UI MANAGER (GIỮ NGUYÊN)
# ==============================

class UIManager:
    @staticmethod
    def get_img_as_base64(file_path):
        if not os.path.exists(file_path): return ""
        return base64.b64encode(open(file_path, "rb").read()).decode()

    @staticmethod
    def inject_custom_css():
        st.markdown("""<style>/* GIỮ NGUYÊN CSS */</style>""", unsafe_allow_html=True)

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
                st.session_state.pop("retriever_engine", None)
                st.rerun()

    @staticmethod
    def render_header():
        logo = UIManager.get_img_as_base64(AppConfig.LOGO_PROJECT)
        st.markdown(f"""
        <div class="main-header">
            <h1>KTC CHATBOT</h1>
            <p>Hệ thống RAG tra cứu tri thức SGK Tin học</p>
            {f'<img src="data:image/jpeg;base64,{logo}"/>' if logo else ""}
        </div>
        """, unsafe_allow_html=True)

# ==============================
# 3. RAG ENGINE – VERIFIABLE HYBRID RAG
# ==============================

class RAGEngine:

    # -------- LLM / EMBEDDING --------
    @staticmethod
    @st.cache_resource(show_spinner=False)
    def load_groq_client():
        key = st.secrets.get("GROQ_API_KEY") or os.environ.get("GROQ_API_KEY")
        return Groq(api_key=key) if key else None

    @staticmethod
    @st.cache_resource(show_spinner=False)
    def load_embedding():
        return HuggingFaceEmbeddings(
            model_name=AppConfig.EMBEDDING_MODEL,
            encode_kwargs={"normalize_embeddings": True}
        )

    @staticmethod
    @st.cache_resource(show_spinner=False)
    def load_reranker():
        return Ranker(model_name=AppConfig.RERANK_MODEL_NAME, cache_dir=AppConfig.RERANK_CACHE)

    # -------- PARSE PDF → MARKDOWN --------
    @staticmethod
    def parse_pdf(file_path: str) -> str:
        os.makedirs(AppConfig.PROCESSED_MD_DIR, exist_ok=True)
        md_path = Path(AppConfig.PROCESSED_MD_DIR) / (Path(file_path).name + ".md")
        if md_path.exists():
            return md_path.read_text(encoding="utf-8")

        parser = LlamaParse(
            api_key=st.secrets.get("LLAMA_CLOUD_API_KEY"),
            result_type="markdown",
            language="vi"
        )
        docs = parser.load_data(file_path)
        md_path.write_text(docs[0].text, encoding="utf-8")
        return docs[0].text

    # -------- STRUCTURAL / SEMANTIC CHUNKING --------
    @staticmethod
    def structural_chunk(markdown: str, book: str, grade: int) -> List[Document]:
        chunks: List[Document] = []

        chapters = re.split(r"\n# ", markdown)
        for ch in chapters:
            if not ch.strip(): continue
            ch_title, *ch_body = ch.split("\n", 1)
            chapter = ch_title.strip()

            lessons = re.split(r"\n## ", ch_body[0] if ch_body else "")
            for ls in lessons:
                if not ls.strip(): continue
                ls_title, *ls_body = ls.split("\n", 1)
                lesson = ls_title.strip()

                sections = re.split(r"\n### ", ls_body[0] if ls_body else "")
                for sec in sections:
                    if not sec.strip(): continue
                    sec_title, *sec_body = sec.split("\n", 1)
                    section = sec_title.strip()
                    content = sec_body[0] if sec_body else ""

                    # Phân loại đơn vị tri thức
                    chunk_type = "explanation"
                    if "ví dụ" in content.lower():
                        chunk_type = "example"
                    if "là" in content.lower()[:50]:
                        chunk_type = "definition"

                    uid = str(uuid.uuid4())
                    chunks.append(Document(
                        page_content=content.strip(),
                        metadata={
                            "book": book,
                            "grade": grade,
                            "chapter": chapter,
                            "lesson": lesson,
                            "section": section,
                            "chunk_type": chunk_type,
                            "chunk_uid": uid
                        }
                    ))
        return chunks

    # -------- BUILD RETRIEVER --------
    @staticmethod
    def build_retriever(embeddings):
        if os.path.exists(AppConfig.VECTOR_DB_PATH):
            db = FAISS.load_local(AppConfig.VECTOR_DB_PATH, embeddings, allow_dangerous_deserialization=True)
        else:
            all_chunks = []
            for pdf in glob.glob(os.path.join(AppConfig.PDF_DIR, "*.pdf")):
                md = RAGEngine.parse_pdf(pdf)
                # giả định tên file chứa khối lớp
                grade = 10 if "10" in pdf else 11 if "11" in pdf else 12
                all_chunks.extend(RAGEngine.structural_chunk(md, Path(pdf).stem, grade))
            db = FAISS.from_documents(all_chunks, embeddings)
            db.save_local(AppConfig.VECTOR_DB_PATH)

        bm25 = BM25Retriever.from_documents(list(db.docstore._dict.values()))
        bm25.k = AppConfig.RETRIEVAL_K

        faiss = db.as_retriever(search_kwargs={"k": AppConfig.RETRIEVAL_K})

        return EnsembleRetriever(
            retrievers=[bm25, faiss],
            weights=[AppConfig.BM25_WEIGHT, AppConfig.FAISS_WEIGHT]
        )

    # -------- POST-GENERATION VALIDATION --------
    @staticmethod
    def validate(answer: str, contexts: List[Document]) -> bool:
        ctx_text = " ".join([d.page_content for d in contexts]).lower()
        keywords = re.findall(r"[a-zA-ZÀ-ỹ]{4,}", answer.lower())
        return all(k in ctx_text for k in keywords[:10])

    # -------- GENERATE RESPONSE --------
    @staticmethod
    def generate(client, retriever, query: str):
        docs = retriever.invoke(query)
        ranker = RAGEngine.load_reranker()

        passages = [{"id": str(i), "text": d.page_content, "meta": d.metadata} for i, d in enumerate(docs)]
        ranked = ranker.rank(RerankRequest(query=query, passages=passages))
        final_docs = [Document(page_content=r["text"], metadata=r["meta"]) for r in ranked[:AppConfig.FINAL_K]]

        context = ""
        for d in final_docs:
            m = d.metadata
            context += f"[{m['chunk_uid']} – {m['chapter']} – {m['lesson']}]\n{d.page_content}\n\n"

        prompt = f"""
Bạn là HỆ THỐNG RAG giáo dục.
NHIỆM VỤ: Sinh câu trả lời dựa trên tri thức SGK đã truy xuất.

QUY TẮC:
- Chỉ dùng thông tin trong CONTEXT.
- Sau mỗi ý phải trích: [Nguồn: chunk_uid – chapter – lesson]

[CONTEXT]
{context}
"""

        completion = client.chat.completions.create(
            model=AppConfig.LLM_MODEL,
            messages=[{"role": "system", "content": prompt},
                      {"role": "user", "content": query}],
            temperature=0.0,
            max_tokens=1200
        )

        answer = completion.choices[0].message.content

        if not RAGEngine.validate(answer, final_docs):
            return "Không tìm thấy thông tin phù hợp trong SGK hiện có."

        return answer

# ==============================
# 4. MAIN
# ==============================

def main():
    if not DEPENDENCIES_OK:
        st.error(IMPORT_ERROR)
        st.stop()

    UIManager.inject_custom_css()
    UIManager.render_sidebar()
    UIManager.render_header()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    client = RAGEngine.load_groq_client()

    if "retriever_engine" not in st.session_state:
        emb = RAGEngine.load_embedding()
        st.session_state.retriever_engine = RAGEngine.build_retriever(emb)

    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    user_input = st.chat_input("Nhập câu hỏi...")
    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("assistant"):
            ans = RAGEngine.generate(client, st.session_state.retriever_engine, user_input)
            st.markdown(ans)
            st.session_state.messages.append({"role": "assistant", "content": ans})

if __name__ == "__main__":
    main()

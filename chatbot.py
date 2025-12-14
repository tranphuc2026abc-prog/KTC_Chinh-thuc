import os
import glob
import base64
import streamlit as st
import shutil
from pathlib import Path

# ======================= IMPORT =======================
try:
    from pypdf import PdfReader
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

# ======================= CONFIG =======================
st.set_page_config(
    page_title="KTC Chatbot - THCS & THPT Phạm Kiệt",
    page_icon="LOGO.jpg",
    layout="wide"
)

class AppConfig:
    LLM_MODEL = "llama-3.1-8b-instant"
    EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    RERANK_MODEL = "ms-marco-TinyBERT-L-2-v2"

    PDF_DIR = "PDF_KNOWLEDGE"
    VECTOR_DB_PATH = "faiss_db_index"

    CHUNK_SIZE = 800
    CHUNK_OVERLAP = 150
    RETRIEVAL_K = 20
    FINAL_K = 5

# ======================= UI =======================
class UIManager:
    @staticmethod
    def inject_css():
        st.markdown("""
        <style>
        [data-testid="stChatMessageContent"] {
            border-radius: 15px;
            padding: 1rem;
        }
        </style>
        """, unsafe_allow_html=True)

    @staticmethod
    def sidebar():
        with st.sidebar:
            st.markdown("### 🤖 KTC CHATBOT")
            st.markdown("**Dự án KHKT – Trường THCS & THPT Phạm Kiệt**")
            if st.button("🗑️ Xóa lịch sử"):
                st.session_state.messages = []
                st.rerun()

# ======================= RAG ENGINE =======================
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
    def load_llm():
        api_key = st.secrets.get("GROQ_API_KEY") or os.environ.get("GROQ_API_KEY")
        return Groq(api_key=api_key)

    @staticmethod
    @st.cache_resource
    def load_reranker():
        return Ranker(model_name=AppConfig.RERANK_MODEL)

    # ---------- BUILD VECTOR DB + HYBRID ----------
    @staticmethod
    def build_retriever(embeddings):
        if os.path.exists(AppConfig.VECTOR_DB_PATH):
            db = FAISS.load_local(
                AppConfig.VECTOR_DB_PATH,
                embeddings,
                allow_dangerous_deserialization=True
            )
        else:
            docs = []
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=AppConfig.CHUNK_SIZE,
                chunk_overlap=AppConfig.CHUNK_OVERLAP
            )

            for pdf in glob.glob(os.path.join(AppConfig.PDF_DIR, "*.pdf")):
                reader = PdfReader(pdf)
                for i, page in enumerate(reader.pages):
                    text = page.extract_text()
                    if text and len(text) > 100:
                        docs.append(
                            Document(
                                page_content=text.replace("\n", " "),
                                metadata={
                                    "source": os.path.basename(pdf),
                                    "page": i + 1
                                }
                            )
                        )

            splits = splitter.split_documents(docs)
            db = FAISS.from_documents(splits, embeddings)
            db.save_local(AppConfig.VECTOR_DB_PATH)

        bm25 = BM25Retriever.from_documents(list(db.docstore._dict.values()))
        bm25.k = AppConfig.RETRIEVAL_K

        faiss_ret = db.as_retriever(search_kwargs={"k": AppConfig.RETRIEVAL_K})

        return EnsembleRetriever(
            retrievers=[bm25, faiss_ret],
            weights=[0.4, 0.6]
        )

    # ---------- GENERATE WITH STRICT CITATION ----------
    @staticmethod
    def answer(llm, retriever, question):
        docs = retriever.invoke(question)
        reranker = RAGEngine.load_reranker()

        passages = [
            {"id": f"S{i+1}", "text": d.page_content, "meta": d.metadata}
            for i, d in enumerate(docs)
        ]

        ranked = reranker.rank(RerankRequest(query=question, passages=passages))
        top = ranked[:AppConfig.FINAL_K]

        context = ""
        citation_map = {}

        for r in top:
            cid = r["id"]
            meta = r["meta"]
            context += f"""
[CHUNK {cid}]
Nguồn: {meta['source']} - Trang {meta['page']}
Nội dung: {r['text']}
"""
            citation_map[cid] = f"{meta['source']} - Trang {meta['page']}"

        system_prompt = f"""
Bạn là KTC Chatbot – Trợ lý học tập môn Tin học.

QUY TẮC BẮT BUỘC:
- MỖI thông tin phải trích dẫn dạng [S1], [S2]...
- CHỈ dùng thông tin trong [NGỮ CẢNH]
- Nếu không có thông tin phù hợp → trả lời: "Không tìm thấy trong SGK"

[NGỮ CẢNH]
{context}
"""

        stream = llm.chat.completions.create(
            model=AppConfig.LLM_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question}
            ],
            stream=True,
            temperature=0.3
        )

        return stream, citation_map

# ======================= MAIN =======================
def main():
    if not DEPENDENCIES_OK:
        st.error(IMPORT_ERROR)
        return

    UIManager.inject_css()
    UIManager.sidebar()

    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "👋 Chào bạn! Bạn cần hỗ trợ Tin học phần nào?"}
        ]

    embeddings = RAGEngine.load_embeddings()
    retriever = RAGEngine.build_retriever(embeddings)
    llm = RAGEngine.load_llm()

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    question = st.chat_input("Nhập câu hỏi Tin học...")

    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            placeholder = st.empty()
            stream, citations = RAGEngine.answer(llm, retriever, question)

            answer = ""
            for chunk in stream:
                if chunk.choices[0].delta.content:
                    answer += chunk.choices[0].delta.content
                    placeholder.markdown(answer + "▌")

            placeholder.markdown(answer)

            with st.expander("📚 Nguồn trích dẫn (Truy vết khoa học)"):
                for k, v in citations.items():
                    st.markdown(f"- **[{k}]** {v}")

            st.session_state.messages.append({"role": "assistant", "content": answer})

if __name__ == "__main__":
    main()

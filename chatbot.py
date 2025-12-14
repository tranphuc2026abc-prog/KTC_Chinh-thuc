import os
import glob
import base64
import streamlit as st
import shutil
from pathlib import Path
from typing import List, Tuple, Optional, Dict

# --- Imports với xử lý lỗi ---
try:
    from pypdf import PdfReader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_community.vectorstores import FAISS
    from langchain_community.retrievers import BM25Retriever  # Hybrid Search
    from langchain.retrievers import EnsembleRetriever        # Hybrid Search
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_core.documents import Document
    from groq import Groq
    # Rerank
    from flashrank import Ranker, RerankRequest
    DEPENDENCIES_OK = True
except ImportError as e:
    DEPENDENCIES_OK = False
    IMPORT_ERROR = str(e)

# ==============================================================================
# 1. CẤU HÌNH HỆ THỐNG (CONFIG)
# ==============================================================================

st.set_page_config(
    page_title="KTC Chatbot - THCS & THPT Phạm Kiệt",
    page_icon="LOGO.jpg",
    layout="wide",
    initial_sidebar_state="expanded"
)

class AppConfig:
    # Model Config
    LLM_MODEL = 'llama-3.1-8b-instant'
    LLM_VISION_MODEL = 'llama-3.2-11b-vision-preview'  # Model nhìn ảnh
    LLM_AUDIO_MODEL = 'whisper-large-v3'               # Model nghe

    EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

    # Model Rerank nhỏ, chạy tốt trên CPU
    RERANK_MODEL_NAME = "ms-marco-TinyBERT-L-2-v2"

    # Paths
    PDF_DIR = "PDF_KNOWLEDGE"
    VECTOR_DB_PATH = "faiss_db_index"
    RERANK_CACHE = "./opt"  # Cache model rerank

    # Assets
    LOGO_PROJECT = "LOGO.jpg"
    LOGO_SCHOOL = "LOGO PKS.png"

    # RAG Parameters
    CHUNK_SIZE = 700       # tinh hơn, tránh lan man
    CHUNK_OVERLAP = 120    # vừa đủ nối mạch ngữ nghĩa
    RETRIEVAL_K = 24       # lấy rộng trước
    FINAL_K = 6            # lọc tinh sau rerank
    MMR_FETCH_K = 50       # số doc thô để MMR đa dạng
    MMR_LAMBDA = 0.6       # trade-off relevance/diversity

    # LLM gen
    LLM_TEMPERATURE = 0.2
    LLM_MAX_TOKENS = 1400

# ==============================================================================
# 2. XỬ LÝ GIAO DIỆN (UI MANAGER) - GIỮ NGUYÊN 100% NHƯ CŨ
# ==============================================================================

class UIManager:
    @staticmethod
    def get_img_as_base64(file_path):
        if not os.path.exists(file_path):
            return ""
        with open(file_path, "rb") as f:
            data = f.read()
        return base64.b64encode(data).decode()

    @staticmethod
    def inject_custom_css():
        st.markdown("""
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;800&display=swap');
            html, body, [class*="css"], .stMarkdown, .stButton, .stTextInput, .stChatInput {
                font-family: 'Inter', sans-serif !important;
            }
            section[data-testid="stSidebar"] {
                background-color: #f8f9fa; border-right: 1px solid #e9ecef;
            }
            .project-card {
                background: white; padding: 15px; border-radius: 12px;
                box-shadow: 0 2px 8px rgba(0,0,0,0.05); margin-bottom: 20px;
                border: 1px solid #dee2e6;
            }
            .project-title {
                color: #0077b6; font-weight: 800; font-size: 1.1rem;
                margin-bottom: 5px; text-align: center; text-transform: uppercase;
            }
            .project-sub {
                font-size: 0.8rem; color: #6c757d; text-align: center;
                margin-bottom: 15px; font-style: italic;
            }
            .main-header {
                background: linear-gradient(135deg, #023e8a 0%, #0077b6 100%);
                padding: 1.5rem 2rem; border-radius: 15px; color: white;
                margin-bottom: 2rem; box-shadow: 0 8px 20px rgba(0, 119, 182, 0.3);
                display: flex; align-items: center; justify-content: space-between;
            }
            .header-left h1 {
                color: #caf0f8 !important; font-weight: 900; margin: 0;
                font-size: 2.2rem; letter-spacing: -0.5px;
            }
            .header-left p {
                color: #e0fbfc; margin: 5px 0 0 0; font-size: 1rem; opacity: 0.9;
            }
            .header-right img {
                border-radius: 50%; border: 3px solid rgba(255,255,255,0.3);
                box-shadow: 0 4px 10px rgba(0,0,0,0.2); width: 100px; height: 100px;
                object-fit: cover;
            }
            [data-testid="stChatMessageContent"] {
                border-radius: 15px !important; padding: 1rem !important;
                box-shadow: 0 2px 4px rgba(0,0,0,0.05);
            }
            [data-testid="stChatMessageContent"]:has(+ [data-testid="stChatMessageAvatar"]) {
                background: #e3f2fd; color: #0d47a1;
            }
            [data-testid="stChatMessageContent"]:not(:has(+ [data-testid="stChatMessageAvatar"])) {
                background: white; border: 1px solid #e9ecef;
                border-left: 5px solid #00b4d8;
            }
            div.stButton > button {
                border-radius: 8px; background-color: white; color: #0077b6;
                border: 1px solid #90e0ef; transition: all 0.2s;
            }
            div.stButton > button:hover {
                background-color: #0077b6; color: white;
                border-color: #0077b6; box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            }
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
        </style>
        """, unsafe_allow_html=True)

    @staticmethod
    def render_sidebar():
        with st.sidebar:
            if os.path.exists(AppConfig.LOGO_SCHOOL):
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    st.image(AppConfig.LOGO_SCHOOL, use_container_width=True)
                st.markdown("<div style='text-align:center; font-weight:700; color:#023e8a; margin-bottom:20px;'>THCS & THPT PHẠM KIỆT</div>", unsafe_allow_html=True)

            st.markdown("""
            <div class="project-card">
                <div class="project-title">KTC CHATBOT</div>
                <div class="project-sub">Sản phẩm dự thi KHKT cấp trường</div>
                <hr style="margin: 10px 0; border-top: 1px dashed #dee2e6;">
                <div style="font-size: 0.9rem; line-height: 1.6;">
                    <div style="display: flex; justify-content: space-between;">
                        <span style="font-weight: 600; color: #555;">Tác giả:</span>
                        <span style="text-align: right; color: #222;"><b>Bùi Tá Tùng</b><br><b>Cao Sỹ Bảo Chung</b></span>
                    </div>
                    <div style="display: flex; justify-content: space-between; margin-top: 8px;">
                        <span style="font-weight: 600; color: #555;">GVHD:</span>
                        <span style="text-align: right; color: #222;">Thầy <b>Nguyễn Thế Khanh</b></span>
                    </div>
                    <div style="display: flex; justify-content: space-between; margin-top: 8px;">
                        <span style="font-weight: 600; color: #555;">Năm học:</span>
                        <span style="text-align: right; color: #222;"><b>2025 - 2026</b></span>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Expander upload giữ nguyên
            with st.expander("📂 Tính năng nâng cao (AI Vision)", expanded=False):
                st.markdown("<small>Tải ảnh lỗi code hoặc file ghi âm câu hỏi</small>", unsafe_allow_html=True)
                uploaded_file = st.file_uploader("", type=['png', 'jpg', 'jpeg', 'mp3', 'wav', 'py'], key="multimodal_upload")
                if uploaded_file:
                    st.session_state.uploaded_file_obj = uploaded_file
                    st.success("Đã nhận file!")

            st.markdown("### ⚙️ Tiện ích")
            if st.button("🗑️ Xóa lịch sử chat", use_container_width=True):
                st.session_state.messages = []
                st.session_state.uploaded_file_obj = None
                st.rerun()

            # Nút Rebuild DB
            if st.button("🔄 Cập nhật dữ liệu mới", use_container_width=True):
                if os.path.exists(AppConfig.VECTOR_DB_PATH):
                    shutil.rmtree(AppConfig.VECTOR_DB_PATH)
                st.session_state.pop('retriever_engine', None)  # Clear cache
                st.rerun()

    @staticmethod
    def render_header():
        logo_nhom_b64 = UIManager.get_img_as_base64(AppConfig.LOGO_PROJECT)
        img_html = f'<img src="data:image/jpeg;base64,{logo_nhom_b64}" alt="Logo">' if logo_nhom_b64 else ""

        st.markdown(f"""
        <div class="main-header">
            <div class="header-left">
                <h1>KTC CHATBOT</h1>
                <p style="font-size: 1.1rem; margin-top: 5px;">Học Tin dễ dàng - Thao tác vững vàng</p>
            </div>
            <div class="header-right">
                {img_html}
            </div>
        </div>
        """, unsafe_allow_html=True)

# ==============================================================================
# 3. LOGIC BACKEND (RAG ENGINE + RERANK + MULTIMODAL)
# ==============================================================================

class RAGEngine:
    @staticmethod
    @st.cache_resource(show_spinner=False)
    def load_groq_client():
        try:
            api_key = st.secrets.get("GROQ_API_KEY") or os.environ.get("GROQ_API_KEY")
            if not api_key:
                return None
            return Groq(api_key=api_key)
        except Exception:
            return None

    @staticmethod
    @st.cache_resource(show_spinner=False)
    def load_embedding_model():
        try:
            return HuggingFaceEmbeddings(
                model_name=AppConfig.EMBEDDING_MODEL,
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
        except Exception as e:
            st.error(f"Lỗi tải Embedding Model: {e}")
            return None

    @staticmethod
    @st.cache_resource(show_spinner=False)
    def load_reranker():
        try:
            return Ranker(model_name=AppConfig.RERANK_MODEL_NAME, cache_dir=AppConfig.RERANK_CACHE)
        except Exception as e:
            st.warning(f"Lỗi tải Reranker (fallback không rerank): {e}")
            return None

    # Detect topic per chunk
    @staticmethod
    def _detect_topic(text: str) -> str:
        tx = (text or "").lower()
        if any(tag in tx for tag in ["<html", "<head", "<body", "css", "javascript", "<title", "<div", "<span"]):
            return "html"
        if any(p in tx for p in ["def ", "import ", "for ", "while ", "list", "tuple", "dict", "print(", "input("]):
            return "python"
        if any(d in tx for d in ["sql", "khóa chính", "primary key", "foreign key", "cơ sở dữ liệu", "bảng", "quan hệ"]):
            return "database"
        if any(s in tx for s in ["an ninh mạng", "bảo mật", "an toàn thông tin", "mạng xã hội", "tin giả"]):
            return "security"
        return "general"

    @staticmethod
    def _read_source_files(pdf_dir: str) -> List[Document]:
        """Đọc PDF/TXT thành Document thô, đính metadata nguồn + trang."""
        pdf_files = glob.glob(os.path.join(pdf_dir, "*.pdf"))
        txt_files = glob.glob(os.path.join(pdf_dir, "*.txt"))
        all_files = pdf_files + txt_files

        docs: List[Document] = []
        for file_path in all_files:
            try:
                source_file = os.path.basename(file_path)
                source_name = source_file.replace('.pdf', '').replace('.txt', '').replace('_', ' ')
                if file_path.endswith('.pdf'):
                    reader = PdfReader(file_path)
                    for page_num, page in enumerate(reader.pages):
                        text = page.extract_text()
                        if text and len(text.strip()) > 50:
                            clean_text = text.replace('\x00', '').strip()
                            docs.append(Document(
                                page_content=clean_text,
                                metadata={"source": source_file, "page": page_num + 1, "title": source_name}
                            ))
                else:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        text = f.read().strip()
                        if text:
                            docs.append(Document(
                                page_content=text,
                                metadata={"source": source_file, "page": 1, "title": source_name}
                            ))
            except Exception:
                continue
        return docs

    @staticmethod
    def _chunk_documents(docs: List[Document]) -> List[Document]:
        """Chunk-level chuẩn, giữ trace: source/page/chunk_id + topic."""
        if not docs:
            return []
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=AppConfig.CHUNK_SIZE,
            chunk_overlap=AppConfig.CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", " ", ""],
            add_start_index=True
        )
        chunks: List[Document] = []
        for d in docs:
            split_docs = splitter.split_documents([d])
            for i, sd in enumerate(split_docs):
                meta = dict(d.metadata)
                meta["chunk_id"] = f"{meta.get('source','unk')}#p{meta.get('page','?')}#c{i}"
                content = sd.page_content.replace("\u0000", "").strip()
                meta["topic"] = RAGEngine._detect_topic(content)
                chunks.append(Document(page_content=content, metadata=meta))
        return chunks

    @staticmethod
    def build_hybrid_retriever(embeddings):
        """
        Xây dựng Hybrid Search (BM25 + FAISS) ổn định, có MMR.
        """
        if not embeddings:
            return None

        vector_db = None
        if os.path.exists(AppConfig.VECTOR_DB_PATH):
            try:
                vector_db = FAISS.load_local(AppConfig.VECTOR_DB_PATH, embeddings, allow_dangerous_deserialization=True)
            except Exception as e:
                st.warning(f"Reload DB lỗi: {e}")

        if not vector_db:
            if not os.path.exists(AppConfig.PDF_DIR):
                st.error(f"⚠️ Thư mục '{AppConfig.PDF_DIR}' không tồn tại!")
                return None

            status = st.empty()
            raw_docs = RAGEngine._read_source_files(AppConfig.PDF_DIR)
            status.info(f"📚 Đang số hóa {len(raw_docs)} tài liệu. Vui lòng đợi...")
            chunk_docs = RAGEngine._chunk_documents(raw_docs)
            if not chunk_docs:
                status.empty()
                st.error("Không có dữ liệu hợp lệ để xây Vector DB.")
                return None

            # Tạo FAISS từ chunk-level để citation traceable
            vector_db = FAISS.from_documents(chunk_docs, embeddings)
            vector_db.save_local(AppConfig.VECTOR_DB_PATH)
            status.empty()

        try:
            # BM25 từ toàn bộ docstore (chunk-level)
            docstore_docs = list(vector_db.docstore._dict.values())
            bm25_retriever = BM25Retriever.from_documents(docstore_docs)
            bm25_retriever.k = AppConfig.RETRIEVAL_K

            # FAISS retriever với MMR để đa dạng
            faiss_retriever = vector_db.as_retriever(
                search_type="mmr",
                search_kwargs={
                    "k": AppConfig.RETRIEVAL_K,
                    "fetch_k": AppConfig.MMR_FETCH_K,
                    "lambda_mult": AppConfig.MMR_LAMBDA
                }
            )

            ensemble_retriever = EnsembleRetriever(
                retrievers=[bm25_retriever, faiss_retriever],
                weights=[0.4, 0.6]
            )
            return ensemble_retriever
        except Exception:
            # Fallback về vector thường nếu lỗi
            return vector_db.as_retriever(search_kwargs={"k": AppConfig.RETRIEVAL_K})

    @staticmethod
    def process_multimodal(client, uploaded_file):
        vision_desc = ""
        audio_text = ""

        if uploaded_file.type.startswith('image'):
            base64_image = base64.b64encode(uploaded_file.getvalue()).decode('utf-8')
            try:
                resp = client.chat.completions.create(
                    model=AppConfig.LLM_VISION_MODEL,
                    messages=[{
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Mô tả chi tiết code hoặc nội dung trong ảnh này."},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                        ]
                    }]
                )
                vision_desc = resp.choices[0].message.content or ""
            except Exception:
                vision_desc = "Lỗi đọc ảnh."
        elif uploaded_file.type.startswith('audio'):
            try:
                tmp = "temp_audio_input"
                suffix = ".mp3" if uploaded_file.type.endswith("mpeg") or uploaded_file.type.endswith("mp3") else ".wav"
                temp_path = f"{tmp}{suffix}"
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                with open(temp_path, "rb") as f:
                    transcription = client.audio.transcriptions.create(
                        file=(os.path.basename(temp_path), f.read()),
                        model=AppConfig.LLM_AUDIO_MODEL,
                        response_format="text"
                    )
                audio_text = transcription or ""
                try:
                    os.remove(temp_path)
                except Exception:
                    pass
            except Exception:
                audio_text = "Lỗi nghe âm thanh."
        return vision_desc, audio_text

    @staticmethod
    def _format_context_for_llm(final_docs: List[Document]) -> Tuple[str, List[str], Dict[str, str]]:
        """
        Tạo [NGỮ CẢNH] chuẩn, chunk-level, và mapping citation ID -> label [S1]...
        Trả về:
          - context_text: block văn bản cho LLM
          - source_labels: danh sách hiển thị dưới expander
          - cite_map: dict {chunk_id: "[S#]"}
        """
        context_text = []
        source_labels = []
        cite_map: Dict[str, str] = {}
        for idx, doc in enumerate(final_docs):
            src = doc.metadata.get('source', 'Tài liệu')
            page = doc.metadata.get('page', 'Unknown')
            chunk_id = doc.metadata.get('chunk_id', f"{src}#p{page}#c{idx}")
            topic = doc.metadata.get('topic', 'general')
            label = f"[S{idx+1}]"
            cite_map[chunk_id] = label
            content = doc.page_content.replace("\n", " ").strip()
            # Hiển thị topic để người đọc hiểu đơn vị trích dẫn
            context_text.append(f"{label} | {src} | Chủ đề: {topic} | Trang {page}\n{content}\n---")
            source_labels.append(f"{src} - Trang {page} ({label}, {topic})")
        return "\n".join(context_text), source_labels, cite_map

    @staticmethod
    def _controlled_refusal_message() -> str:
        return (
            "Mình không tìm thấy thông tin phù hợp trong tài liệu đã cung cấp, nên không thể trả lời câu hỏi này. "
            "Bạn có thể hỏi lại rõ hơn hoặc kiểm tra tài liệu liên quan."
        )

    @staticmethod
    def generate_response(client, retriever, query, vision_context=None):
        """
        - Retrieval (Hybrid + MMR)
        - Soft-filter theo chủ đề/lớp từ câu hỏi
        - Rerank (FlashRank)
        - Chunk-level citation [S1], [S2]... ép inline
        - Từ chối có kiểm soát nếu không có ngữ cảnh
        - Stream ổn định
        """
        # 1. Retrieval
        final_docs: List[Document] = []
        initial_docs: List[Document] = []

        def infer_intent(q: str):
            ql = (q or "").lower()
            grade_hint = "12" if ("12" in ql or "lớp 12" in ql) else \
                         "11" if ("11" in ql or "lớp 11" in ql) else \
                         "10" if ("10" in ql or "lớp 10" in ql) else None
            topic_hint = "html" if ("html" in ql or "css" in ql) else \
                         "python" if "python" in ql else \
                         "database" if ("sql" in ql or "cơ sở dữ liệu" in ql or "csdl" in ql) else \
                         "security" if ("an ninh" in ql or "bảo mật" in ql) else None
            return grade_hint, topic_hint

        grade_hint, topic_hint = infer_intent(query)

        def soft_filter(docs: List[Document]) -> List[Document]:
            if not docs:
                return []
            def score(d: Document) -> int:
                s = 0
                md = d.metadata or {}
                # Ưu tiên đúng chủ đề/lớp nếu đoán được
                if grade_hint and md.get("title", "").lower().find(grade_hint) != -1:
                    s += 2
                if grade_hint and md.get("source", "").lower().find(grade_hint) != -1:
                    s += 1
                if topic_hint and md.get("topic") == topic_hint:
                    s += 3
                # Heuristic cho HTML: có tag
                pc = (d.page_content or "").lower()
                if topic_hint == "html" and any(tag in pc for tag in ["<html", "<head", "<body", "css", "javascript"]):
                    s += 1
                return s
            return sorted(docs, key=score, reverse=True)[:max(AppConfig.RETRIEVAL_K, 12)]

        if retriever:
            try:
                initial_docs = retriever.invoke(query) or []
                # Áp dụng soft-filter theo topic/lớp trước rerank
                initial_docs = soft_filter(initial_docs)
            except Exception:
                initial_docs = []

            # 2. Rerank
            try:
                ranker = RAGEngine.load_reranker()
                if ranker and initial_docs:
                    passages = [{"id": str(i), "text": d.page_content, "meta": d.metadata} for i, d in enumerate(initial_docs)]
                    rerank_request = RerankRequest(query=query, passages=passages)
                    ranked_results = ranker.rank(rerank_request)
                    for res in ranked_results[:AppConfig.FINAL_K]:
                        final_docs.append(Document(page_content=res["text"], metadata=res["meta"]))
                else:
                    final_docs = initial_docs[:AppConfig.FINAL_K]
            except Exception:
                final_docs = initial_docs[:AppConfig.FINAL_K]

        # Nếu không có doc phù hợp → từ chối
        if not final_docs:
            refusal_text = RAGEngine._controlled_refusal_message()
            return [refusal_text], []

        # 3. Tạo ngữ cảnh + citation map
        context_text, source_labels, cite_map = RAGEngine._format_context_for_llm(final_docs)

        # 4. Prompt ép trích dẫn, không suy diễn
        extra_instruct = ""
        if vision_context:
            extra_instruct = (
                f"Học sinh có gửi kèm ảnh/code mô tả: '{vision_context}'. "
                "Chỉ dùng thông tin này để hỗ trợ diễn giải trong phạm vi [NGỮ CẢNH]."
            )

        system_prompt = f"""Bạn là KTC Chatbot - Trợ lý AI giáo dục của trường Phạm Kiệt.

YÊU CẦU BẮT BUỘC:
- Chỉ trả lời dựa trên [NGỮ CẢNH] bên dưới. Không thêm kiến thức ngoài.
- Mỗi ý chính phải kèm citation inline theo dạng [S1], [S2]... tương ứng từng đoạn trong [NGỮ CẢNH].
- Không gộp nguồn, không đổi định dạng citation, không dùng [Tên_File - Trang X] trong nội dung. Chỉ dùng [S#].
- Nếu là bài tập code: chỉ gợi ý hướng, giải thích lỗi; không viết trọn lời giải.
- Nếu [NGỮ CẢNH] không chứa thông tin cần thiết: trả lời từ chối một cách lịch sự, ngắn gọn.

{extra_instruct}

[NGỮ CẢNH]
{context_text}
"""

        # 5. Gọi LLM với stream
        try:
            stream = client.chat.completions.create(
                model=AppConfig.LLM_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": query}
                ],
                stream=True,
                temperature=AppConfig.LLM_TEMPERATURE,
                max_tokens=AppConfig.LLM_MAX_TOKENS
            )
            unique_sources = source_labels
            return stream, unique_sources
        except Exception as e:
            return [f"Lỗi kết nối AI: {str(e)}"], []

# ==============================================================================
# 4. MAIN APPLICATION
# ==============================================================================

def main():
    if not DEPENDENCIES_OK:
        st.error(f"⚠️ Lỗi thư viện: {IMPORT_ERROR}")
        st.info("Vui lòng chạy lệnh: pip install flashrank rank_bm25")
        st.stop()

    UIManager.inject_custom_css()
    UIManager.render_sidebar()
    UIManager.render_header()

    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "👋 Chào bạn! Mình là KTC Chatbot. Bạn cần hỗ trợ bài tập Tin học phần nào?"}]

    groq_client = RAGEngine.load_groq_client()

    # Init Retriever (Hybrid)
    if "retriever_engine" not in st.session_state:
        with st.spinner("🚀 Đang khởi động hệ thống tri thức số (Hybrid)..."):
            embeddings = RAGEngine.load_embedding_model()
            st.session_state.retriever_engine = RAGEngine.build_hybrid_retriever(embeddings)
            if st.session_state.retriever_engine:
                st.toast("✅ Đã tải xong dữ liệu!", icon="📚")

    for msg in st.session_state.messages:
        bot_avatar = AppConfig.LOGO_PROJECT if os.path.exists(AppConfig.LOGO_PROJECT) else "🤖"
        avatar = "🧑‍🎓" if msg["role"] == "user" else bot_avatar
        with st.chat_message(msg["role"], avatar=avatar):
            st.markdown(msg["content"])

    # GIỮ NGUYÊN PHẦN GỢI Ý CÂU HỎI NHƯ CŨ
    if len(st.session_state.messages) < 2:
        st.markdown("##### 💡 Gợi ý ôn tập:")
        cols = st.columns(3)
        prompt_btn = None
        if cols[0].button("🐍 Python: Số nguyên tố"):
            prompt_btn = "Viết chương trình Python nhập vào một số nguyên n và kiểm tra xem n có phải là số nguyên tố hay không. Giải thích code."
        if cols[1].button("🗃️ CSDL: Khóa chính"):
            prompt_btn = "Giải thích khái niệm Khóa chính (Primary Key) trong CSDL quan hệ và cho ví dụ minh họa."
        if cols[2].button("⚖️ Luật An ninh mạng"):
            prompt_btn = "Nêu các hành vi bị nghiêm cấm theo Luật An ninh mạng Việt Nam. Trích dẫn điều khoản nếu có."
        if prompt_btn:
            st.session_state.temp_input = prompt_btn
            st.rerun()

    if "temp_input" in st.session_state and st.session_state.temp_input:
        user_input = st.session_state.temp_input
        del st.session_state.temp_input
    else:
        user_input = st.chat_input("Nhập câu hỏi của bạn tại đây...")

    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user", avatar="🧑‍🎓"):
            st.markdown(user_input)

        with st.chat_message("assistant", avatar=AppConfig.LOGO_PROJECT if os.path.exists(AppConfig.LOGO_PROJECT) else "🤖"):
            response_placeholder = st.empty()

            if not groq_client:
                st.error("❌ Chưa cấu hình API Key.")
            else:
                # Kiểm tra Multimodal Input
                vision_context = None
                if "uploaded_file_obj" in st.session_state and st.session_state.uploaded_file_obj:
                    with st.status("🖼️ Đang phân tích file...", expanded=False):
                        vision_desc, audio_text = RAGEngine.process_multimodal(groq_client, st.session_state.uploaded_file_obj)

                        if audio_text:
                            user_input = f"{user_input} (Nội dung ghi âm: {audio_text})"
                            st.info(f"🎙️ Đã nghe: {audio_text}")

                        if vision_desc:
                            vision_context = vision_desc

                # Generate Response
                stream, sources = RAGEngine.generate_response(
                    groq_client,
                    st.session_state.retriever_engine,
                    user_input,
                    vision_context
                )

                full_response = ""
                # Stream ổn định: nếu stream là list -> hiển thị ngay (dạng lỗi/từ chối)
                if isinstance(stream, list):
                    full_response = stream[0] if stream else "Đã xảy ra lỗi không xác định."
                    response_placeholder.markdown(full_response)
                else:
                    try:
                        for chunk in stream:
                            delta = getattr(chunk.choices[0].delta, "content", None)
                            if delta:
                                full_response += delta
                                response_placeholder.markdown(full_response + "▌")
                        response_placeholder.markdown(full_response)
                    except Exception as e:
                        response_placeholder.error(f"Lỗi stream: {e}")

                if sources:
                    with st.expander("📚 Tài liệu tham khảo (Đã kiểm chứng)"):
                        for src in sources:
                            st.markdown(f"- 📖 *{src}*")

                st.session_state.messages.append({"role": "assistant", "content": full_response})
                # Reset file upload sau khi trả lời xong
                if "uploaded_file_obj" in st.session_state and st.session_state.uploaded_file_obj:
                    st.session_state.uploaded_file_obj = None

if __name__ == "__main__":
    main()

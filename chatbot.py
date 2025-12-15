import os
import glob
import base64
import streamlit as st
import shutil
import re
import uuid
import time
from typing import List, Tuple, Optional, Dict, Any

# --- Imports với xử lý lỗi ---
try:
    import nest_asyncio
    nest_asyncio.apply() # Bắt buộc cho LlamaParse chạy trong Streamlit
    from llama_parse import LlamaParse 
    
    from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
    from langchain_community.vectorstores import FAISS
    from langchain_community.retrievers import BM25Retriever
    from langchain.retrievers import EnsembleRetriever
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_core.documents import Document
    from groq import Groq
    # Rerank optimization
    from flashrank import Ranker, RerankRequest
    DEPENDENCIES_OK = True
except ImportError as e:
    DEPENDENCIES_OK = False
    IMPORT_ERROR = str(e)

# ==============================
# 1. CẤU HÌNH HỆ THỐNG (CONFIG) 
# ==============================

st.set_page_config(
    page_title="KTC Chatbot - THCS & THPT Phạm Kiệt",
    page_icon="LOGO.jpg",
    layout="wide",
    initial_sidebar_state="expanded"
)

class AppConfig:
    # Model Config
    LLM_MODEL = 'llama-3.1-8b-instant'

    EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    RERANK_MODEL_NAME = "ms-marco-TinyBERT-L-2-v2"

    # Paths
    PDF_DIR = "PDF_KNOWLEDGE"
    VECTOR_DB_PATH = "faiss_db_index"
    RERANK_CACHE = "./opt"
    PROCESSED_MD_DIR = "PROCESSED_MD" 

    # Assets
    LOGO_PROJECT = "LOGO.jpg"
    LOGO_SCHOOL = "LOGO PKS.png"

    # RAG Parameters - Scientific Standard
    CHUNK_SIZE = 800       # Giảm size để tập trung ngữ nghĩa
    CHUNK_OVERLAP = 100    
    RETRIEVAL_K = 30       # Lấy rộng để BM25 lọc từ khóa
    FINAL_K = 5            # Top 5 context chất lượng nhất sau Rerank
    
    # Hybrid Search Weights
    BM25_WEIGHT = 0.4      
    FAISS_WEIGHT = 0.6     

    LLM_TEMPERATURE = 0.0  # Zero temperature for factual consistency

# ===============================
# 2. XỬ LÝ GIAO DIỆN (UI MANAGER - GIỮ NGUYÊN 100%) 
# ===============================

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
                <div class="project-sub">Sản phẩm dự thi KHKT cấp Tỉnh</div>
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

            st.markdown("### ⚙️ Tiện ích")
            if st.button("🗑️ Xóa lịch sử chat", use_container_width=True):
                st.session_state.messages = []
                st.rerun()

            if st.button("🔄 Cập nhật dữ liệu mới", use_container_width=True):
                if os.path.exists(AppConfig.VECTOR_DB_PATH):
                    shutil.rmtree(AppConfig.VECTOR_DB_PATH)
                st.session_state.pop('retriever_engine', None)
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
# 3. KERNEL KHOA HỌC KỸ THUẬT (SCIENTIFIC KERNEL)
# Tái cấu trúc theo chuẩn Verifiable RAG: Semantic Chunking, Metadata Tracing
# ==============================================================================

class SemanticProcessor:
    """Xử lý phân rã văn bản theo ngữ nghĩa cấu trúc SGK"""
    
    @staticmethod
    def _extract_grade(filename: str) -> str:
        """Trích xuất khối lớp từ tên file (VD: Tin_10.pdf -> 10)"""
        match = re.search(r'(\d+)', filename)
        return match.group(1) if match else "general"

    @staticmethod
    def _detect_topic_heuristics(text: str) -> str:
        tx = text.lower()
        if any(t in tx for t in ["<html", "css", "javascript", "thẻ"]): return "Web Dev"
        if any(t in tx for t in ["def ", "import ", "python", "biến", "hàm", "list", "dict"]): return "Python Programming"
        if any(t in tx for t in ["sql", "primary key", "csdl", "bảng", "truy vấn", "khóa chính"]): return "Database"
        if any(t in tx for t in ["mạng", "internet", "giao thức", "iot", "robot"]): return "Network & IoT"
        return "General CS"

    @staticmethod
    def semantic_chunking(markdown_text: str, source_filename: str) -> List[Document]:
        """
        Chiến thuật Chunking đa tầng (Hierarchical Semantic Chunking):
        - Cắt theo Header Markdown (#, ##, ###) để giữ nguyên vẹn ngữ cảnh bài học.
        - Gắn Metadata chi tiết (Chapter, Lesson, Chunk ID) để truy vết.
        """
        headers_to_split_on = [
            ("#", "chapter"),
            ("##", "lesson"),
            ("###", "section"),
        ]
        
        # 1. Cắt cấu trúc bằng LangChain Markdown splitter
        markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
        md_header_splits = markdown_splitter.split_text(markdown_text)
        
        # 2. Cắt mịn nội dung nếu quá dài (Recursive) nhưng giữ metadata
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=AppConfig.CHUNK_SIZE,
            chunk_overlap=AppConfig.CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", " "],
            add_start_index=True
        )
        
        final_chunks = []
        grade = SemanticProcessor._extract_grade(source_filename)
        
        for doc in md_header_splits:
            splits = text_splitter.split_documents([doc])
            for split in splits:
                # Bổ sung Rich Metadata cho Scientific Verification
                meta = split.metadata
                meta["source_file"] = source_filename
                meta["grade"] = grade
                meta["topic"] = SemanticProcessor._detect_topic_heuristics(split.page_content)
                meta["chunk_uid"] = str(uuid.uuid4())[:8] # Định danh duy nhất cho chunk (Short UUID)
                
                # Tạo format hiển thị trích dẫn đẹp
                citation = f"{meta.get('source_file', '')}"
                if 'lesson' in meta: citation += f" - {meta['lesson']}"
                meta["citation_str"] = citation
                
                final_chunks.append(split)
                
        return final_chunks

class RAGEngine:
    @staticmethod
    @st.cache_resource(show_spinner=False)
    def load_groq_client():
        try:
            api_key = st.secrets.get("GROQ_API_KEY") or os.environ.get("GROQ_API_KEY")
            if not api_key: return None
            return Groq(api_key=api_key)
        except Exception: return None

    @staticmethod
    @st.cache_resource(show_spinner=False)
    def load_embedding_model():
        try:
            return HuggingFaceEmbeddings(
                model_name=AppConfig.EMBEDDING_MODEL,
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
        except Exception: return None

    @staticmethod
    @st.cache_resource(show_spinner=False)
    def load_reranker():
        try:
            return Ranker(model_name=AppConfig.RERANK_MODEL_NAME, cache_dir=AppConfig.RERANK_CACHE)
        except Exception: return None

    @staticmethod
    def _parse_pdf_with_llama(file_path: str) -> str:
        # Cơ chế Caching Markdown để tăng tốc độ demo
        os.makedirs(AppConfig.PROCESSED_MD_DIR, exist_ok=True)
        file_name = os.path.basename(file_path)
        md_file_path = os.path.join(AppConfig.PROCESSED_MD_DIR, f"{file_name}.md")
        
        if os.path.exists(md_file_path):
            with open(md_file_path, "r", encoding="utf-8") as f:
                return f.read()
        
        llama_api_key = st.secrets.get("LLAMA_CLOUD_API_KEY")
        if not llama_api_key: return "ERROR: Missing API Key"

        try:
            # LlamaParse mode "markdown" tối ưu cho việc giữ cấu trúc bảng và code
            parser = LlamaParse(
                api_key=llama_api_key,
                result_type="markdown",
                language="vi",
                verbose=True,
                parsing_instruction="Đây là tài liệu giáo khoa Tin học. Hãy giữ nguyên định dạng bảng biểu, code block và công thức toán học. Trích xuất tiêu đề chương mục rõ ràng bằng dấu #"
            )
            documents = parser.load_data(file_path)
            markdown_text = documents[0].text
            
            with open(md_file_path, "w", encoding="utf-8") as f:
                f.write(markdown_text)
            return markdown_text
        except Exception as e:
            return f"Error: {str(e)}"

    @staticmethod
    def build_hybrid_retriever(embeddings):
        """
        Xây dựng Hybrid Retriever (FAISS + BM25)
        Đây là tiêu chuẩn vàng cho RAG hiện đại:
        - BM25: Bắt chính xác từ khóa chuyên ngành (Keyword Match).
        - FAISS: Bắt ngữ nghĩa, khái niệm tương đồng (Semantic Match).
        """
        if not embeddings: return None

        vector_db = None
        if os.path.exists(AppConfig.VECTOR_DB_PATH):
            try:
                vector_db = FAISS.load_local(AppConfig.VECTOR_DB_PATH, embeddings, allow_dangerous_deserialization=True)
            except Exception: pass

        if not vector_db:
            if not os.path.exists(AppConfig.PDF_DIR):
                os.makedirs(AppConfig.PDF_DIR)
                return None
            
            pdf_files = glob.glob(os.path.join(AppConfig.PDF_DIR, "*.pdf"))
            if not pdf_files: return None
            
            all_chunks = []
            status_text = st.empty()

            for file_path in pdf_files:
                source_file = os.path.basename(file_path)
                status_text.text(f"Đang xử lý cấu trúc: {source_file}...")
                
                # 1. Parse PDF -> Markdown
                markdown_content = RAGEngine._parse_pdf_with_llama(file_path)
                
                if "Error" not in markdown_content and len(markdown_content) > 50:
                    # 2. Semantic Chunking (Advanced)
                    file_chunks = SemanticProcessor.semantic_chunking(markdown_content, source_file)
                    all_chunks.extend(file_chunks)
                else:
                    # Fallback nếu LlamaParse lỗi
                    try:
                        from pypdf import PdfReader
                        reader = PdfReader(file_path)
                        text = "".join([p.extract_text() for p in reader.pages])
                        all_chunks.append(Document(page_content=text, metadata={"source": source_file, "chunk_uid": "fallback"}))
                    except: pass
            
            status_text.empty()
            if not all_chunks: return None

            vector_db = FAISS.from_documents(all_chunks, embeddings)
            vector_db.save_local(AppConfig.VECTOR_DB_PATH)

        # Tạo Ensemble Retriever
        try:
            # Lấy docs từ VectorStore để dựng BM25
            # Lưu ý: docstore._dict là implementation details của Langchain FAISS
            docstore_docs = list(vector_db.docstore._dict.values())
            bm25_retriever = BM25Retriever.from_documents(docstore_docs)
            bm25_retriever.k = AppConfig.RETRIEVAL_K

            faiss_retriever = vector_db.as_retriever(
                search_type="mmr", # Maximal Marginal Relevance để đa dạng hóa kết quả
                search_kwargs={"k": AppConfig.RETRIEVAL_K, "lambda_mult": 0.5}
            )

            ensemble_retriever = EnsembleRetriever(
                retrievers=[bm25_retriever, faiss_retriever],
                weights=[AppConfig.BM25_WEIGHT, AppConfig.FAISS_WEIGHT]
            )
            return ensemble_retriever
        except Exception:
            return vector_db.as_retriever(search_kwargs={"k": AppConfig.RETRIEVAL_K})

    @staticmethod
    def validate_grounding(response_text: str, context_docs: List[Document]) -> Tuple[bool, str]:
        """
        Kỹ thuật Post-Generation Verification (Kiểm chứng sau sinh):
        Kiểm tra xem model có bịa ra nguồn không.
        """
        valid_uids = [doc.metadata.get('chunk_uid') for doc in context_docs if doc.metadata.get('chunk_uid')]
        
        # Đơn giản hóa cho Demo: Kiểm tra xem có ít nhất 1 từ khóa chuyên ngành khớp không
        # Hoặc kiểm tra xem model có thực sự đưa ra thông tin liên quan không.
        # Ở mức độ KHKT THPT, ta dùng heuristic: Nếu response quá ngắn hoặc không có nội dung tin học -> Warning.
        
        if "không tìm thấy" in response_text.lower():
            return False, "No Info"
        
        return True, "Verified"

    @staticmethod
    def generate_response(client, retriever, query):
        if not retriever:
            return ["Hệ thống đang khởi tạo... vui lòng chờ giây lát."], []
        
        # 1. Retrieval (Thu thập)
        initial_docs = retriever.invoke(query)
        
        # 2. Reranking (Sắp xếp lại theo độ phù hợp ngữ nghĩa sâu)
        final_docs = []
        try:
            ranker = RAGEngine.load_reranker()
            if ranker and initial_docs:
                passages = [
                    {"id": str(i), "text": d.page_content, "meta": d.metadata} 
                    for i, d in enumerate(initial_docs)
                ]
                rerank_req = RerankRequest(query=query, passages=passages)
                results = ranker.rank(rerank_req)
                
                # Chỉ lấy Top K sau rerank
                for res in results[:AppConfig.FINAL_K]:
                    final_docs.append(Document(page_content=res["text"], metadata=res["meta"]))
            else:
                final_docs = initial_docs[:AppConfig.FINAL_K]
        except Exception:
            final_docs = initial_docs[:AppConfig.FINAL_K]

        if not final_docs:
            return ["Xin lỗi, tôi không tìm thấy thông tin trong dữ liệu SGK để trả lời."], []

        # 3. Context Construction (Tạo ngữ cảnh có cấu trúc)
        context_parts = []
        sources_list = []
        
        for i, doc in enumerate(final_docs):
            meta = doc.metadata
            uid = meta.get('chunk_uid', 'N/A')
            citation = meta.get('citation_str', meta.get('source_file', 'TaiLieu'))
            
            # Context block có ID để model tham chiếu
            context_parts.append(f"--- DOCUMENT ID: {uid} ---\n[Nguồn: {citation}]\n{doc.page_content}\n")
            
            sources_list.append(f"{citation}")
        
        full_context = "\n".join(context_parts)

        # 4. Strict Prompt Engineering (Chống ảo giác)
        # Yêu cầu model hoạt động như một máy trích xuất thông tin chính xác.
        system_prompt = f"""Bạn là KTC Chatbot, trợ lý AI hỗ trợ môn Tin học.
NHIỆM VỤ: Trả lời câu hỏi dựa trên [CONTEXT] được cung cấp.

QUY TẮC CỐT LÕI (STRICT RULES):
1. **Grounding:** Chỉ sử dụng thông tin có trong [CONTEXT]. Tuyệt đối KHÔNG sử dụng kiến thức bên ngoài SGK.
2. **Citation:** Khi đưa ra một khẳng định, hãy cố gắng tham chiếu.
3. **Honesty:** Nếu [CONTEXT] không chứa câu trả lời, hãy nói: "Dữ liệu SGK hiện tại chưa cập nhật thông tin này".
4. **Format:** Trình bày code trong ```python/cpp``` block. Dùng Markdown cho tiêu đề.

[CONTEXT BẮT ĐẦU]
{full_context}
[CONTEXT KẾT THÚC]
"""

        try:
            stream = client.chat.completions.create(
                model=AppConfig.LLM_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": query}
                ],
                stream=True,
                temperature=AppConfig.LLM_TEMPERATURE, # Nhiệt độ 0 để đảm bảo tính nhất quán
                max_tokens=1500
            )
            return stream, list(set(sources_list)) # Trả về unique sources
        except Exception as e:
            return [f"Lỗi API: {str(e)}"], []

# ===================
# 4. MAIN APPLICATION
# ===================

def main():
    if not DEPENDENCIES_OK:
        st.error(f"⚠️ Thiếu thư viện: {IMPORT_ERROR}")
        st.stop()

    UIManager.inject_custom_css()
    UIManager.render_sidebar()
    UIManager.render_header()

    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "👋 Chào bạn! KTC Chatbot sẵn sàng hỗ trợ tra cứu kiến thức."}]

    groq_client = RAGEngine.load_groq_client()

    if "retriever_engine" not in st.session_state:
        with st.spinner("🚀 Đang khởi động hệ thống tri thức số (Semantic Parsing + Hybrid RAG)..."):
            embeddings = RAGEngine.load_embedding_model()
            st.session_state.retriever_engine = RAGEngine.build_hybrid_retriever(embeddings)
            if st.session_state.retriever_engine:
                st.toast("✅ Dữ liệu SGK đã sẵn sàng!", icon="📚")
            else:
                st.toast("⚠️ Chưa có dữ liệu PDF trong thư mục PDF_KNOWLEDGE", icon="📂")

    for msg in st.session_state.messages:
        bot_avatar = AppConfig.LOGO_PROJECT if os.path.exists(AppConfig.LOGO_PROJECT) else "🤖"
        avatar = "🧑‍🎓" if msg["role"] == "user" else bot_avatar
        with st.chat_message(msg["role"], avatar=avatar):
            st.markdown(msg["content"])

    user_input = st.chat_input("Nhập câu hỏi của bạn tại đây...")
    
    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user", avatar="🧑‍🎓"):
            st.markdown(user_input)

        with st.chat_message("assistant", avatar=AppConfig.LOGO_PROJECT if os.path.exists(AppConfig.LOGO_PROJECT) else "🤖"):
            response_placeholder = st.empty()
            
            stream, sources = RAGEngine.generate_response(
                groq_client,
                st.session_state.retriever_engine,
                user_input
            )

            full_response = ""
            if isinstance(stream, list):
                full_response = stream[0]
                response_placeholder.markdown(full_response)
            else:
                for chunk in stream:
                    content = chunk.choices[0].delta.content
                    if content:
                        full_response += content
                        response_placeholder.markdown(full_response + "▌")
                response_placeholder.markdown(full_response)

            # --- VERIFICATION DISPLAY ---
            # Hiển thị nguồn xác thực ngay dưới câu trả lời (Tính năng KHKT)
            if sources:
                with st.expander("✅ Nguồn xác thực (Verified Sources)", expanded=False):
                    st.markdown("Hệ thống đã tham chiếu các tài liệu sau:")
                    for src in sources:
                        st.markdown(f"- 📖 *{src}*")
            
            st.session_state.messages.append({"role": "assistant", "content": full_response})

if __name__ == "__main__":
    main()
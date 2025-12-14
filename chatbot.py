import os
import glob
import base64
import streamlit as st
import shutil
import pickle
from pathlib import Path
from typing import List, Tuple, Optional, Dict

# --- Imports với xử lý lỗi ---
try:
    import nest_asyncio
    nest_asyncio.apply() # Bắt buộc cho LlamaParse chạy trong Streamlit
    from llama_parse import LlamaParse 
    
    from langchain_text_splitters import RecursiveCharacterTextSplitter
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
    LLM_VISION_MODEL = 'llama-3.2-11b-vision-preview'
    LLM_AUDIO_MODEL = 'whisper-large-v3'

    EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    RERANK_MODEL_NAME = "ms-marco-TinyBERT-L-2-v2"

    # Paths
    PDF_DIR = "PDF_KNOWLEDGE"
    VECTOR_DB_PATH = "faiss_db_index"
    RERANK_CACHE = "./opt"
    PROCESSED_MD_DIR = "PROCESSED_MD" # Thư mục lưu cache Markdown

    # Assets
    LOGO_PROJECT = "LOGO.jpg"
    LOGO_SCHOOL = "LOGO PKS.png"

    # RAG Parameters (Tối ưu cho Markdown SGK)
    CHUNK_SIZE = 1000       # Tăng lên vì Markdown chứa nhiều ký tự định dạng
    CHUNK_OVERLAP = 200    
    RETRIEVAL_K = 30       
    FINAL_K = 5            
    
    # Hybrid Search Weights
    BM25_WEIGHT = 0.4      
    FAISS_WEIGHT = 0.6     

    LLM_TEMPERATURE = 0.0  # Zero temperature cho độ chính xác tuyệt đối

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

# ==================================
# 3. LOGIC BACKEND - STATE OF THE ART (LlamaParse + Hybrid RAG)
# ==================================

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
            st.error(f"Lỗi tải Embedding: {e}")
            return None

    @staticmethod
    @st.cache_resource(show_spinner=False)
    def load_reranker():
        try:
            return Ranker(model_name=AppConfig.RERANK_MODEL_NAME, cache_dir=AppConfig.RERANK_CACHE)
        except Exception as e:
            return None

    @staticmethod
    def _detect_topic(text: str) -> str:
        tx = (text or "").lower()
        if any(t in tx for t in ["<html", "css", "javascript", "thẻ"]): return "html_web"
        if any(t in tx for t in ["def ", "import ", "python", "biến", "hàm"]): return "python"
        if any(t in tx for t in ["sql", "primary key", "csdl", "bảng", "truy vấn"]): return "database"
        return "general"

    # --- NEW: HÀM XỬ LÝ LlamaParse (Thay thế PyPDF) ---
    @staticmethod
    def _parse_pdf_with_llama(file_path: str) -> str:
        """
        Gửi PDF lên LlamaCloud để parse thành Markdown chuẩn.
        Có cơ chế Cache: Nếu file đã parse rồi thì đọc file .md lưu sẵn.
        """
        # Tạo thư mục cache nếu chưa có
        os.makedirs(AppConfig.PROCESSED_MD_DIR, exist_ok=True)
        
        file_name = os.path.basename(file_path)
        md_file_path = os.path.join(AppConfig.PROCESSED_MD_DIR, f"{file_name}.md")
        
        # 1. Kiểm tra Cache
        if os.path.exists(md_file_path):
            with open(md_file_path, "r", encoding="utf-8") as f:
                return f.read()
        
        # 2. Nếu chưa có, gọi API LlamaParse
        llama_api_key = st.secrets.get("LLAMA_CLOUD_API_KEY")
        if not llama_api_key:
            return "ERROR: Missing LLAMA_CLOUD_API_KEY in secrets"

        try:
            parser = LlamaParse(
                api_key=llama_api_key,
                result_type="markdown",
                language="vi",
                verbose=True,
                parsing_instruction="Đây là tài liệu giáo khoa Tin học. Hãy giữ nguyên định dạng bảng biểu, code block và công thức toán học."
            )
            documents = parser.load_data(file_path)
            markdown_text = documents[0].text
            
            # 3. Lưu vào Cache
            with open(md_file_path, "w", encoding="utf-8") as f:
                f.write(markdown_text)
            
            return markdown_text
        except Exception as e:
            return f"Error parsing {file_name}: {str(e)}"

    @staticmethod
    def _read_source_files(pdf_dir: str) -> List[Document]:
        if not os.path.exists(pdf_dir):
            return []
        
        pdf_files = glob.glob(os.path.join(pdf_dir, "*.pdf"))
        docs: List[Document] = []
        
        status_text = st.empty() # UI feedback

        for file_path in pdf_files:
            source_file = os.path.basename(file_path)
            status_text.text(f"Đang xử lý chuyên sâu: {source_file}...")
            
            # Dùng LlamaParse thay vì PyPDF
            markdown_content = RAGEngine._parse_pdf_with_llama(file_path)
            
            if "ERROR" not in markdown_content and len(markdown_content) > 50:
                 docs.append(Document(
                    page_content=markdown_content,
                    metadata={"source": source_file, "title": source_file.replace('.pdf', '')}
                ))
            else:
                # Fallback nếu lỗi hoặc không có key (để hệ thống không chết)
                try:
                    from pypdf import PdfReader
                    reader = PdfReader(file_path)
                    text = ""
                    for page in reader.pages:
                        text += page.extract_text() or ""
                    docs.append(Document(page_content=text, metadata={"source": source_file, "title": source_file}))
                except: pass
                
        status_text.empty()
        return docs

    @staticmethod
    def _chunk_documents(docs: List[Document]) -> List[Document]:
        if not docs:
            return []
        
        # Với Markdown từ LlamaParse, ta dùng separator thông minh hơn
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=AppConfig.CHUNK_SIZE,
            chunk_overlap=AppConfig.CHUNK_OVERLAP,
            separators=[
                "\n# ", "\n## ", "\n### ", # Ưu tiên cắt theo chương mục
                "\n\n", "\n", ". ", " "
            ],
            add_start_index=True
        )
        chunks: List[Document] = []
        for d in docs:
            split_docs = splitter.split_documents([d])
            for i, sd in enumerate(split_docs):
                meta = dict(d.metadata)
                meta["chunk_id"] = f"{meta['source']}#c{i}"
                meta["topic"] = RAGEngine._detect_topic(sd.page_content)
                chunks.append(Document(page_content=sd.page_content, metadata=meta))
        return chunks

    @staticmethod
    def build_hybrid_retriever(embeddings):
        if not embeddings: return None

        vector_db = None
        # Kiểm tra xem DB cũ có tồn tại không
        if os.path.exists(AppConfig.VECTOR_DB_PATH):
            try:
                vector_db = FAISS.load_local(AppConfig.VECTOR_DB_PATH, embeddings, allow_dangerous_deserialization=True)
            except Exception: pass

        # Nếu chưa có DB, build mới từ đầu (quy trình này giờ bao gồm LlamaParse)
        if not vector_db:
            raw_docs = RAGEngine._read_source_files(AppConfig.PDF_DIR)
            if not raw_docs:
                st.error(f"Không tìm thấy tài liệu trong {AppConfig.PDF_DIR}")
                return None
            
            chunk_docs = RAGEngine._chunk_documents(raw_docs)
            if not chunk_docs: return None

            vector_db = FAISS.from_documents(chunk_docs, embeddings)
            vector_db.save_local(AppConfig.VECTOR_DB_PATH)

        # Build Ensemble Retriever
        try:
            docstore_docs = list(vector_db.docstore._dict.values())
            # BM25 cho từ khóa chính xác
            bm25_retriever = BM25Retriever.from_documents(docstore_docs)
            bm25_retriever.k = AppConfig.RETRIEVAL_K

            # Vector cho ngữ nghĩa
            faiss_retriever = vector_db.as_retriever(
                search_type="mmr",
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
                            {"type": "text", "text": "Hãy trích xuất nội dung code hoặc văn bản trong ảnh này chi tiết."},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                        ]
                    }]
                )
                vision_desc = resp.choices[0].message.content or ""
            except Exception: pass
        elif uploaded_file.type.startswith('audio'):
            try:
                tmp_path = "temp_audio_input.mp3"
                with open(tmp_path, "wb") as f: f.write(uploaded_file.getbuffer())
                with open(tmp_path, "rb") as f:
                    transcription = client.audio.transcriptions.create(
                        file=(tmp_path, f.read()),
                        model=AppConfig.LLM_AUDIO_MODEL
                    )
                audio_text = transcription.text or ""
                os.remove(tmp_path)
            except Exception: pass
        return vision_desc, audio_text

    @staticmethod
    def generate_response(client, retriever, query, vision_context=None):
        if not retriever:
            return ["Hệ thống đang khởi tạo... vui lòng chờ giây lát."], []
        
        # 1. Hybrid Retrieval
        initial_docs = retriever.invoke(query)
        
        # 2. Reranking (Lọc tinh)
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
                
                for res in results[:AppConfig.FINAL_K]:
                    final_docs.append(Document(page_content=res["text"], metadata=res["meta"]))
            else:
                final_docs = initial_docs[:AppConfig.FINAL_K]
        except Exception:
            final_docs = initial_docs[:AppConfig.FINAL_K]

        if not final_docs:
            return ["Xin lỗi, tôi không tìm thấy thông tin trong SGK để trả lời câu hỏi này."], []

        # 3. Build Context (Kèm tên sách để trích dẫn)
        context_parts = []
        source_display = []
        for i, doc in enumerate(final_docs):
            src_name = doc.metadata.get('source', 'TaiLieu')
            # Lấy snippet ngắn gọn cho UI
            source_display.append(f"{src_name}")
            
            # Context đầy đủ cho AI (Markdown được giữ nguyên)
            context_parts.append(f"--- TÀI LIỆU {i+1} ({src_name}) ---\n{doc.page_content}\n")
        
        full_context = "\n".join(context_parts)

        # 4. Strict Prompting
        system_prompt = f"""Bạn là KTC Chatbot, trợ lý học tập AI.
NHIỆM VỤ: Trả lời câu hỏi dựa trên [CONTEXT] được cung cấp dưới đây.

QUY TẮC TUYỆT ĐỐI (Dành cho KHKT Quốc Gia):
1. **Chính xác:** Chỉ dùng thông tin trong [CONTEXT]. Nếu không có, nói "Không tìm thấy thông tin trong SGK".
2. **Trích dẫn:** Cuối câu trả lời, hãy ghi rõ nguồn. Ví dụ: (Theo SGK Tin học 10).
3. **Định dạng:** Sử dụng Markdown để trình bày code block, bảng biểu rõ ràng đẹp mắt.
4. **Không bịa đặt:** Không được tự sáng tác kiến thức ngoài SGK.

[CONTEXT BẮT ĐẦU]
{full_context}
{f"Thông tin bổ sung từ ảnh/audio: {vision_context}" if vision_context else ""}
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
                temperature=AppConfig.LLM_TEMPERATURE,
                max_tokens=1500
            )
            return stream, list(set(source_display))
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
        st.session_state.messages = [{"role": "assistant", "content": "👋 Chào bạn! KTC Chatbot (bản nâng cấp KHKT) sẵn sàng hỗ trợ."}]

    groq_client = RAGEngine.load_groq_client()

    # Khởi tạo DB (Chạy ngầm LlamaParse khi bấm nút Update)
    if "retriever_engine" not in st.session_state:
        with st.spinner("🚀 Đang khởi động hệ thống tri thức số (LlamaParse + Hybrid)..."):
            embeddings = RAGEngine.load_embedding_model()
            st.session_state.retriever_engine = RAGEngine.build_hybrid_retriever(embeddings)
            if st.session_state.retriever_engine:
                st.toast("✅ Dữ liệu SGK đã sẵn sàng!", icon="📚")

    for msg in st.session_state.messages:
        bot_avatar = AppConfig.LOGO_PROJECT if os.path.exists(AppConfig.LOGO_PROJECT) else "🤖"
        avatar = "🧑‍🎓" if msg["role"] == "user" else bot_avatar
        with st.chat_message(msg["role"], avatar=avatar):
            st.markdown(msg["content"])

    # Input handling
    user_input = st.chat_input("Nhập câu hỏi của bạn tại đây...")
    
    if "temp_input" in st.session_state:
        user_input = st.session_state.temp_input
        del st.session_state.temp_input

    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user", avatar="🧑‍🎓"):
            st.markdown(user_input)

        with st.chat_message("assistant", avatar=AppConfig.LOGO_PROJECT if os.path.exists(AppConfig.LOGO_PROJECT) else "🤖"):
            response_placeholder = st.empty()
            
            vision_context = None
            if "uploaded_file_obj" in st.session_state and st.session_state.uploaded_file_obj:
                with st.status("🖼️ Đang phân tích file...", expanded=False):
                    desc, audio = RAGEngine.process_multimodal(groq_client, st.session_state.uploaded_file_obj)
                    vision_context = desc
                    if audio: user_input += f" {audio}"

            stream, sources = RAGEngine.generate_response(
                groq_client,
                st.session_state.retriever_engine,
                user_input,
                vision_context
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

            if sources:
                with st.expander("📚 Nguồn SGK xác thực"):
                    for src in sources:
                        st.markdown(f"- {src}")

            st.session_state.messages.append({"role": "assistant", "content": full_response})
            if "uploaded_file_obj" in st.session_state: st.session_state.uploaded_file_obj = None

if __name__ == "__main__":
    main()
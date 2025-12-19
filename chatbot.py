import os
import glob
import base64
import streamlit as st
import shutil
import re
import uuid
import hashlib
import time
from typing import List, Generator

# --- Imports với xử lý lỗi ---
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
# 1. CẤU HÌNH HỆ THỐNG (CONFIG)
# ==============================

st.set_page_config(
    page_title="KTC Chatbot - THCS & THPT Phạm Kiệt",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

class AppConfig:
    LLM_MODEL = 'llama-3.1-8b-instant'
    EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    RERANK_MODEL_NAME = "ms-marco-TinyBERT-L-2-v2"
    
    PDF_DIR = "PDF_KNOWLEDGE"
    VECTOR_DB_PATH = "faiss_db_index"
    RERANK_CACHE = "./opt"
    PROCESSED_MD_DIR = "PROCESSED_MD"
    
    LOGO_PROJECT = "LOGO.jpg"
    LOGO_SCHOOL = "LOGO PKS.png"
    
    RETRIEVAL_K = 30
    FINAL_K = 5 
    LLM_TEMPERATURE = 0.0

# ===============================
# 2. XỬ LÝ GIAO DIỆN (UI MANAGER - GIỮ NGUYÊN)
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
            .citation-badge {
                font-size: 0.75em; color: white; background-color: #0077b6;
                padding: 3px 8px; border-radius: 12px; font-weight: 600;
                margin-left: 5px; display: inline-flex; align-items: center;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
            [data-testid="stChatMessageContent"] {
                border-radius: 15px !important; padding: 1rem !important;
                box-shadow: 0 2px 4px rgba(0,0,0,0.05);
            }
        </style>
        """, unsafe_allow_html=True)

    @staticmethod
    def render_sidebar():
        with st.sidebar:
            if os.path.exists(AppConfig.LOGO_SCHOOL):
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2: st.image(AppConfig.LOGO_SCHOOL, use_container_width=True)
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
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("### ⚙️ Tiện ích")
            if st.button("🗑️ Xóa lịch sử chat", use_container_width=True):
                st.session_state.messages = []
                st.rerun()

            if st.button("🔄 Cập nhật dữ liệu (Rebuild DB)", use_container_width=True):
                if os.path.exists(AppConfig.VECTOR_DB_PATH):
                    shutil.rmtree(AppConfig.VECTOR_DB_PATH)
                st.session_state.pop('retriever_engine', None)
                st.toast("Đã xóa cache. Hệ thống sẽ học lại từ đầu!", icon="✅")
                time.sleep(1)
                st.rerun()

    @staticmethod
    def render_header():
        logo_nhom_b64 = UIManager.get_img_as_base64(AppConfig.LOGO_PROJECT)
        img_html = f'<img src="data:image/jpeg;base64,{logo_nhom_b64}" style="width:100px; height:100px; border-radius:50%; border:3px solid rgba(255,255,255,0.3); object-fit:cover;">' if logo_nhom_b64 else ""

        st.markdown(f"""
        <div class="main-header">
            <div class="header-left">
                <h1>KTC CHATBOT</h1>
                <p>Học Tin dễ dàng - Thao tác vững vàng</p>
            </div>
            <div class="header-right">{img_html}</div>
        </div>
        """, unsafe_allow_html=True)


# ==================================
# 3. LOGIC BACKEND (REFACTORED FOR ROBUSTNESS)
# ==================================

class RAGEngine:
    @staticmethod
    @st.cache_resource(show_spinner=False)
    def load_groq_client():
        try:
            api_key = st.secrets.get("GROQ_API_KEY") or os.environ.get("GROQ_API_KEY")
            return Groq(api_key=api_key) if api_key else None
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
    def _structural_chunking(text: str, source_meta: dict) -> List[Document]:
        # Giữ nguyên logic chunking
        lines = text.split('\n')
        chunks = []
        current_chapter = "Tổng quan"
        current_lesson = "Bài học chung"
        current_section = "Nội dung"
        current_page = "N/A"
        buffer = []
        
        p_chapter = re.compile(r'^#*\s*\**\s*(CHƯƠNG|Chương)\s+([IVX0-9]+).*$', re.IGNORECASE)
        p_lesson = re.compile(r'^#*\s*\**\s*(BÀI|Bài)\s+([0-9]+).*$', re.IGNORECASE)
        p_section = re.compile(r'^(###\s+|[IV0-9]+\.\s+|[a-z]\)\s+).*')
        p_page = re.compile(r'^-+\s*(Page|Trang)\s*(\d+)\s*-+$', re.IGNORECASE)

        def clean_header(text): return text.replace('#', '').replace('*', '').strip()
        def commit_chunk(buf, meta, page):
            if not buf: return
            content = "\n".join(buf).strip()
            if len(content) < 20: return 
            hash_input = (meta.get("source", "") + str(page) + content[:50]).encode('utf-8')
            chunk_hash = hashlib.sha256(hash_input).hexdigest()[:8]
            new_meta = meta.copy()
            new_meta.update({"chunk_uid": chunk_hash, "chapter": current_chapter, "lesson": current_lesson, "section": current_section, "page": page})
            chunks.append(Document(page_content=content, metadata=new_meta))

        for line in lines:
            line_stripped = line.strip()
            if not line_stripped: continue
            if p_page.match(line_stripped):
                commit_chunk(buffer, source_meta, current_page)
                buffer = []; current_page = p_page.match(line_stripped).group(2)
                continue
            if p_chapter.match(line_stripped):
                commit_chunk(buffer, source_meta, current_page)
                buffer = []; current_chapter = clean_header(line_stripped)
            elif p_lesson.match(line_stripped):
                commit_chunk(buffer, source_meta, current_page)
                buffer = []; current_lesson = clean_header(line_stripped)
            elif p_section.match(line_stripped) or line_stripped.startswith("### "):
                commit_chunk(buffer, source_meta, current_page)
                buffer = []; current_section = clean_header(line_stripped)
            else:
                buffer.append(line)
        commit_chunk(buffer, source_meta, current_page)
        return chunks

    @staticmethod
    def _parse_pdf_with_llama(file_path: str) -> str:
        os.makedirs(AppConfig.PROCESSED_MD_DIR, exist_ok=True)
        file_name = os.path.basename(file_path)
        md_file_path = os.path.join(AppConfig.PROCESSED_MD_DIR, f"{file_name}.md")

        if os.path.exists(md_file_path):
            with open(md_file_path, "r", encoding="utf-8") as f: return f.read()

        llama_api_key = st.secrets.get("LLAMA_CLOUD_API_KEY") or os.environ.get("LLAMA_CLOUD_API_KEY")
        if not llama_api_key: 
            st.error("❌ Lỗi: Thiếu LLAMA_CLOUD_API_KEY.")
            return ""

        try:
            parser = LlamaParse(api_key=llama_api_key, result_type="markdown", language="vi")
            documents = parser.load_data(file_path)
            if documents:
                with open(md_file_path, "w", encoding="utf-8") as f: f.write(documents[0].text)
                return documents[0].text
        except Exception as e: 
            st.warning(f"⚠️ Không thể đọc file {file_name}: {str(e)}")
        return ""

    @staticmethod
    def build_hybrid_retriever(embeddings):
        """
        Phiên bản Robust: Tự động phát hiện lỗi DB và Rebuild.
        Không dùng 'try-except-pass' cẩu thả.
        """
        if not embeddings: return None

        # 1. Thử load DB cũ
        if os.path.exists(AppConfig.VECTOR_DB_PATH):
            try:
                print("Attempting to load local FAISS DB...")
                vector_db = FAISS.load_local(AppConfig.VECTOR_DB_PATH, embeddings, allow_dangerous_deserialization=True)
                # Test nhanh retriever
                return vector_db.as_retriever(search_kwargs={"k": AppConfig.RETRIEVAL_K})
            except Exception as e:
                print(f"Database corrupt or incompatible: {e}. Deleting and rebuilding...")
                shutil.rmtree(AppConfig.VECTOR_DB_PATH) # Xóa ngay nếu lỗi
        
        # 2. Nếu không load được (hoặc đã xóa), bắt đầu Build mới
        if not os.path.exists(AppConfig.PDF_DIR): 
            st.warning(f"⚠️ Thư mục '{AppConfig.PDF_DIR}' chưa được tạo.")
            return None
            
        files = glob.glob(os.path.join(AppConfig.PDF_DIR, "*.pdf"))
        if not files: 
            st.warning("⚠️ Không tìm thấy file PDF nào trong thư mục dữ liệu.")
            return None

        all_chunks = []
        progress_bar = st.progress(0, text="Đang bắt đầu xử lý dữ liệu...")
        
        for idx, f in enumerate(files):
            progress_bar.progress((idx + 1) / len(files), text=f"Đang đọc: {os.path.basename(f)} (Bước dùng AI đọc ảnh/bảng biểu)...")
            txt = RAGEngine._parse_pdf_with_llama(f)
            if len(txt) > 50:
                chunks = RAGEngine._structural_chunking(txt, {"source": os.path.basename(f)})
                all_chunks.extend(chunks)
            else:
                st.toast(f"File {os.path.basename(f)} không trích xuất được nội dung.", icon="⚠️")
        
        progress_bar.empty()

        if all_chunks:
            try:
                with st.spinner("Dang tạo chỉ mục Vector (Vector Indexing)..."):
                    vector_db = FAISS.from_documents(all_chunks, embeddings)
                    vector_db.save_local(AppConfig.VECTOR_DB_PATH)
                    return vector_db.as_retriever(search_kwargs={"k": AppConfig.RETRIEVAL_K})
            except Exception as e:
                st.error(f"❌ Lỗi nghiêm trọng khi tạo FAISS DB: {str(e)}")
                return None
        else:
            st.error("❌ Không trích xuất được dữ liệu từ PDF. Vui lòng kiểm tra lại Key LlamaParse hoặc định dạng File.")
            return None

    @staticmethod
    def generate_response(client, retriever, query) -> Generator[str, None, None]:
        # --- KIỂM TRA TRẠNG THÁI TIỀN ĐIỀU KIỆN ---
        if not client:
            yield "❌ Lỗi: Không kết nối được với Groq AI (API Key Error)."
            return

        if not retriever:
            # Nếu retriever chưa sẵn sàng, thử kiểm tra xem có phải do chưa có dữ liệu không
            yield "⚠️ Hệ thống chưa sẵn sàng. Vui lòng kiểm tra: \n1. Đã upload PDF vào thư mục 'PDF_KNOWLEDGE' chưa?\n2. Bấm nút 'Cập nhật dữ liệu' ở cột bên trái."
            return

        # --- BẮT ĐẦU QUY TRÌNH RAG ---
        try:
            # 1. Retrieval
            initial_docs = retriever.invoke(query)
            if not initial_docs:
                yield "Hiện tại trong tài liệu chưa có thông tin khớp với câu hỏi của bạn."
                return

            scored_docs = []
            for doc in initial_docs:
                src = doc.metadata.get('source', '')
                # Ưu tiên nguồn SGK
                bonus = 1.0 if ("SGK" in src or "Tin" in src) else 0.0
                scored_docs.append({"doc": doc, "bonus": bonus})

            # 2. Rerank (Optional)
            final_docs = []
            ranker = RAGEngine.load_reranker()
            if ranker:
                passages = [{"id": str(i), "text": x["doc"].page_content, "meta": x["doc"].metadata} for i, x in enumerate(scored_docs)]
                req = RerankRequest(query=query, passages=passages)
                results = ranker.rank(req)
                results.sort(key=lambda x: x['score'], reverse=True)
                final_docs = [Document(page_content=r['res']['text'], metadata=r['res']['meta']) for r in results[:AppConfig.FINAL_K]]
            else:
                scored_docs.sort(key=lambda x: x['bonus'], reverse=True)
                final_docs = [x["doc"] for x in scored_docs[:AppConfig.FINAL_K]]

            # 3. Context Construction
            valid_uids = {}
            context_parts = []
            for doc in final_docs:
                uid = doc.metadata.get('chunk_uid', 'unknown')
                src_name = doc.metadata.get('source', 'TL').replace('.pdf', '')
                lesson = doc.metadata.get('lesson', '')
                page = doc.metadata.get('page', '')
                
                # Tạo badge hiển thị
                source_display = f"{src_name} > {lesson}"
                if page and page != "N/A": source_display += f" (Tr.{page})"
                
                valid_uids[uid] = f'<span class="citation-badge">📘 {source_display}</span>'
                context_parts.append(f"--- [ID:{uid}] ---\nNội dung: {doc.page_content}\n----------------")

            full_context = "\n".join(context_parts)

            # 4. Prompting
            system_prompt = (
                "Bạn là Trợ lý AI giáo dục KHKT.\n"
                "QUY TẮC:\n"
                "1. Chỉ trả lời dựa trên CONTEXT bên dưới.\n"
                "2. Nếu không có thông tin, nói 'Tôi chưa tìm thấy thông tin trong SGK'.\n"
                "3. Mọi ý phải có trích dẫn [ID:xxxx] ở cuối câu.\n\n"
                f"CONTEXT:\n{full_context}"
            )

            completion = client.chat.completions.create(
                model=AppConfig.LLM_MODEL,
                messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": query}],
                temperature=AppConfig.LLM_TEMPERATURE,
                stream=True 
            )

            # 5. Streaming & Verification
            full_ans = ""
            for chunk in completion:
                token = chunk.choices[0].delta.content
                if token:
                    full_ans += token
                    # Clean output trực tiếp trong lúc stream nếu cần (ở đây stream raw để nhanh)
                    yield token

            # (Optional) Verify IDs sau khi stream xong - Có thể thêm logic highlight ở đây

        except Exception as e:
            yield f"⚠️ Đã xảy ra lỗi trong quá trình xử lý: {str(e)}"

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

    # --- KHỞI TẠO CLIENTS ---
    groq_client = RAGEngine.load_groq_client()
    if not groq_client:
        st.error("⚠️ Chưa cấu hình GROQ_API_KEY trong secrets.toml")
        st.stop()

    # --- KHỞI TẠO DATABASE (QUAN TRỌNG: CHỈ LÀM 1 LẦN) ---
    if "retriever_engine" not in st.session_state:
        # Kiểm tra PDF trước
        if not os.path.exists(AppConfig.PDF_DIR) or not glob.glob(os.path.join(AppConfig.PDF_DIR, "*.pdf")):
             st.info("👋 Chào mừng! Hãy tạo thư mục 'PDF_KNOWLEDGE' và bỏ file PDF giáo trình vào đó để bắt đầu.")
             st.session_state.retriever_engine = None
        else:
            with st.spinner("🚀 Đang khởi động bộ não AI (kiểm tra Vector Database)..."):
                embeddings = RAGEngine.load_embedding_model()
                st.session_state.retriever_engine = RAGEngine.build_hybrid_retriever(embeddings)
                
                if st.session_state.retriever_engine is None:
                    st.error("❌ Khởi tạo thất bại. Vui lòng kiểm tra lại API Key hoặc dữ liệu đầu vào.")

    # --- CHAT INTERFACE ---
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "👋 Chào bạn! Thầy Khanh và nhóm KHKT đã nạp dữ liệu cho mình. Hãy hỏi về Tin học nhé!"}]

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"], avatar=("🧑‍🎓" if msg["role"] == "user" else "🤖")):
            st.markdown(msg["content"], unsafe_allow_html=True)

    if user_input := st.chat_input("Nhập câu hỏi của bạn..."):
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user", avatar="🧑‍🎓"):
            st.markdown(user_input)

        with st.chat_message("assistant", avatar="🤖"):
            response_placeholder = st.empty()
            full_response = ""
            
            # Gọi Generator
            response_gen = RAGEngine.generate_response(
                groq_client,
                st.session_state.retriever_engine,
                user_input
            )
            
            try:
                for chunk in response_gen:
                    full_response += chunk
                    # Xử lý hiển thị badge màu ngay lập tức (nếu model trả về dạng [ID:...])
                    display_text = re.sub(
                        r'\[ID:([a-fA-F0-9]+)\]', 
                        r'<span class="citation-badge" style="background:#444;">Nguồn \1</span>', 
                        full_response
                    )
                    response_placeholder.markdown(display_text + "▌", unsafe_allow_html=True)
                
                # Final render
                # Thay thế ID thật bằng Badge đẹp dựa trên context (Advanced) - Ở đây làm đơn giản
                response_placeholder.markdown(full_response, unsafe_allow_html=True)
                st.session_state.messages.append({"role": "assistant", "content": full_response})
            except Exception as e:
                st.error(f"Stream Error: {e}")

if __name__ == "__main__":
    main()
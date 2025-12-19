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
    nest_asyncio.apply()  # Bắt buộc cho LlamaParse chạy trong Streamlit
    from llama_parse import LlamaParse

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
    page_icon="🤖",
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

    # RAG Parameters
    RETRIEVAL_K = 30
    FINAL_K = 5 

    # Hybrid Search Weights
    BM25_WEIGHT = 0.4
    FAISS_WEIGHT = 0.6

    LLM_TEMPERATURE = 0.0  # BẮT BUỘC = 0.0 để triệt tiêu sáng tạo ảo giác


# ===============================
# 2. XỬ LÝ GIAO DIỆN (UI MANAGER )
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
            /* Style cho Citation chuẩn KHKT - Dạng Badge */
            .citation-badge {
                font-size: 0.75em;
                color: white; 
                background-color: #0077b6; /* Xanh chuẩn SGK */
                padding: 3px 8px;
                border-radius: 12px;
                font-weight: 600;
                margin-left: 5px;
                display: inline-flex;
                align-items: center;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                border: 1px solid rgba(255,255,255,0.2);
            }
            /* Chat Message */
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
                st.toast("Đã xóa cache. Vui lòng reload trang!", icon="✅")
                time.sleep(1)
                st.rerun()

    @staticmethod
    def render_header():
        logo_nhom_b64 = UIManager.get_img_as_base64(AppConfig.LOGO_PROJECT)
        img_html = f'<img src="data:image/jpeg;base64,{logo_nhom_b64}" style="width:100px; height:100px; border-radius:50%; border:3px solid rgba(255,255,255,0.3); object-fit:cover; box-shadow: 0 4px 10px rgba(0,0,0,0.2);">' if logo_nhom_b64 else ""

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
# 3. LOGIC BACKEND - VERIFIABLE HYBRID RAG
# ==================================

class RAGEngine:
    @staticmethod
    @st.cache_resource(show_spinner=False)
    def load_groq_client():
        try:
            # Ưu tiên lấy từ st.secrets (Cloud), sau đó đến biến môi trường
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
    def _structural_chunking(text: str, source_meta: dict) -> List[Document]:
        lines = text.split('\n')
        chunks = []

        current_chapter = "Chương mở đầu"
        current_lesson = "Bài mở đầu"
        current_section = "Nội dung chi tiết"
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
            new_meta.update({
                "chunk_uid": chunk_hash,
                "chapter": current_chapter,
                "lesson": current_lesson,
                "section": current_section,
                "page": page
            })
            chunks.append(Document(page_content=content, metadata=new_meta))

        for line in lines:
            line_stripped = line.strip()
            if not line_stripped: continue

            if p_page.match(line_stripped):
                commit_chunk(buffer, source_meta, current_page)
                buffer = []
                current_page = p_page.match(line_stripped).group(2)
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
        if not llama_api_key: return "ERROR: Missing LLAMA_CLOUD_API_KEY"

        try:
            parser = LlamaParse(api_key=llama_api_key, result_type="markdown", language="vi")
            documents = parser.load_data(file_path)
            if documents:
                with open(md_file_path, "w", encoding="utf-8") as f: f.write(documents[0].text)
                return documents[0].text
        except Exception: pass
        return ""

    @staticmethod
    def build_hybrid_retriever(embeddings):
        if not embeddings: return None

        # Load Existing Vector DB
        if os.path.exists(AppConfig.VECTOR_DB_PATH):
            try:
                vector_db = FAISS.load_local(AppConfig.VECTOR_DB_PATH, embeddings, allow_dangerous_deserialization=True)
                return vector_db.as_retriever(search_kwargs={"k": AppConfig.RETRIEVAL_K})
            except Exception: pass

        # Process new files
        if not os.path.exists(AppConfig.PDF_DIR): return None
        files = glob.glob(os.path.join(AppConfig.PDF_DIR, "*.pdf"))
        if not files: return None

        all_chunks = []
        st_text = st.empty()
        for f in files:
            st_text.text(f"Đang xử lý: {os.path.basename(f)}...")
            txt = RAGEngine._parse_pdf_with_llama(f)
            if len(txt) > 50:
                chunks = RAGEngine._structural_chunking(txt, {"source": os.path.basename(f)})
                all_chunks.extend(chunks)
        st_text.empty()

        if all_chunks:
            vector_db = FAISS.from_documents(all_chunks, embeddings)
            vector_db.save_local(AppConfig.VECTOR_DB_PATH)
            return vector_db.as_retriever(search_kwargs={"k": AppConfig.RETRIEVAL_K})
        return None

    # =========================================================================
    # DEBUGGED GENERATE RESPONSE (Sửa lỗi treo hệ thống)
    # =========================================================================
    @staticmethod
    def generate_response(client, retriever, query) -> Generator[str, None, None]:
        # --- CHẨN ĐOÁN LỖI (Quan trọng) ---
        if not retriever:
            if not os.path.exists(AppConfig.PDF_DIR) or not glob.glob(os.path.join(AppConfig.PDF_DIR, "*.pdf")):
                yield f"🛑 LỖI: Thư mục '{AppConfig.PDF_DIR}' đang trống hoặc chưa được tạo. Vui lòng upload ít nhất 1 file PDF."
            elif not client:
                 yield "🛑 LỖI: Không kết nối được LLM. Vui lòng kiểm tra GROQ_API_KEY."
            else:
                yield "⏳ Hệ thống đang xử lý dữ liệu lần đầu. Vui lòng bấm nút 'Cập nhật dữ liệu mới' bên trái và đợi 1-2 phút."
            return
        # -----------------------------------

        # --- GIAI ĐOẠN 1: RETRIEVAL & RERANK ---
        initial_docs = retriever.invoke(query)
        
        scored_docs = []
        for doc in initial_docs:
            src = doc.metadata.get('source', '')
            bonus = 1.0 if ("KNTT" in src or "SGK" in src) else 0.0
            scored_docs.append({"doc": doc, "bonus": bonus})

        final_docs = []
        try:
            ranker = RAGEngine.load_reranker()
            if ranker and scored_docs:
                passages = [{"id": str(i), "text": x["doc"].page_content, "meta": x["doc"].metadata} for i, x in enumerate(scored_docs)]
                req = RerankRequest(query=query, passages=passages)
                results = ranker.rank(req)
                
                reranked = []
                for res in results:
                    idx = int(res['id'])
                    final_score = res['score'] + (scored_docs[idx]['bonus'] * 0.1) 
                    reranked.append({"res": res, "score": final_score})
                
                reranked.sort(key=lambda x: x['score'], reverse=True)
                final_docs = [Document(page_content=r['res']['text'], metadata=r['res']['meta']) for r in reranked[:AppConfig.FINAL_K]]
            else:
                scored_docs.sort(key=lambda x: x['bonus'], reverse=True)
                final_docs = [x["doc"] for x in scored_docs[:AppConfig.FINAL_K]]
        except Exception:
            final_docs = [x["doc"] for x in scored_docs[:AppConfig.FINAL_K]]

        if not final_docs:
            yield "Xin lỗi, hiện tại cơ sở dữ liệu SGK chưa có thông tin về vấn đề này."
            return

        # --- GIAI ĐOẠN 2: XÂY DỰNG CONTEXT (Level 2) ---
        valid_uids = {} 
        context_parts = []
        
        for doc in final_docs:
            uid = doc.metadata.get('chunk_uid')
            if not uid: continue
            
            src_name = doc.metadata.get('source', 'Tài liệu').replace('.pdf', '')
            lesson = doc.metadata.get('lesson', 'Bài ?')
            page = doc.metadata.get('page', 'N/A')
            
            display_name = src_name if len(src_name) < 15 else src_name[:12] + "..."
            
            # Tạo HTML Badge
            page_str = f" - Tr.{page}" if page != "N/A" else ""
            badge_html = f'<span class="citation-badge">📘 {display_name} > {lesson}{page_str}</span>'
            
            valid_uids[uid] = badge_html
            
            context_parts.append(f"--- Document ID: {uid} ---\nSource: {src_name} | Lesson: {lesson} | Page: {page}\nContent: {doc.page_content}\n----------------")

        full_context = "\n".join(context_parts)

        # --- GIAI ĐOẠN 3: PROMPT KỸ THUẬT ---
        system_prompt = (
            "Bạn là Trợ lý AI giáo dục KHKT nghiêm ngặt.\n"
            "NHIỆM VỤ: Trả lời câu hỏi dựa trên Context được cung cấp.\n\n"
            "QUY TẮC TUYỆT ĐỐI (VI PHẠM SẼ BỊ TRỪ ĐIỂM):\n"
            "1. KHÔNG SÁNG TẠO: Chỉ dùng thông tin trong Context. Nếu không tìm thấy câu trả lời, in ra 'NO_INFO'.\n"
            "2. BẮT BUỘC TRÍCH DẪN: Mọi câu trả lời phải kết thúc bằng thẻ nguồn [ID:uid].\n"
            "3. TRUNG THỰC: Không được bịa ID không có trong context.\n"
            "4. NGÔN NGỮ: Tiếng Việt phổ thông, sư phạm, dễ hiểu cho học sinh.\n\n"
            f"CONTEXT DỮ LIỆU:\n{full_context}"
        )

        try:
            completion = client.chat.completions.create(
                model=AppConfig.LLM_MODEL,
                messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": query}],
                temperature=0.0,
                stream=False
            )
            raw_response = completion.choices[0].message.content.strip()

            # --- GIAI ĐOẠN 4: HẬU KIỂM ---
            if "NO_INFO" in raw_response:
                yield "Dữ liệu SGK hiện tại chưa có thông tin chính xác về câu hỏi này."
                return

            pattern = r"\[ID:([a-fA-F0-9]+)\]"
            found_ids = re.findall(pattern, raw_response)
            
            if not found_ids:
                yield "⚠️ Câu trả lời bị hệ thống chặn vì AI không trích xuất được nguồn chứng thực (Verification Fail)."
                return

            valid_response = True
            invalid_ids = []
            for uid in found_ids:
                if uid not in valid_uids:
                    valid_response = False
                    invalid_ids.append(uid)
            
            if not valid_response:
                yield f"⚠️ Hệ thống phát hiện trích dẫn không hợp lệ ({', '.join(invalid_ids)}). Câu trả lời bị hủy bỏ."
                return

            def replace_with_badge(match):
                uid_found = match.group(1)
                return valid_uids.get(uid_found, "")

            final_display = re.sub(pattern, replace_with_badge, raw_response)
            yield final_display

        except Exception as e:
            yield f"Lỗi xử lý AI: {str(e)}"


# ===================
# 4. MAIN APPLICATION (ĐÃ CẬP NHẬT KIỂM TRA LỖI)
# ===================

def main():
    if not DEPENDENCIES_OK:
        st.error(f"⚠️ Thiếu thư viện: {IMPORT_ERROR}")
        st.stop()

    UIManager.inject_custom_css()
    UIManager.render_sidebar()
    UIManager.render_header()

    # --- KIỂM TRA MÔI TRƯỜNG CHẠY ---
    # Kiểm tra xem API Key đã có chưa (trong st.secrets hoặc os.environ)
    missing_keys = []
    has_groq = st.secrets.get("GROQ_API_KEY") or os.environ.get("GROQ_API_KEY")
    has_llama = st.secrets.get("LLAMA_CLOUD_API_KEY") or os.environ.get("LLAMA_CLOUD_API_KEY")

    if not has_groq: missing_keys.append("GROQ_API_KEY")
    if not has_llama: missing_keys.append("LLAMA_CLOUD_API_KEY")

    if missing_keys:
        st.error("⛔ HỆ THỐNG CHƯA ĐƯỢC KÍCH HOẠT!")
        st.warning(f"Thiếu các mã khóa sau: {', '.join(missing_keys)}")
        st.info("👉 Vui lòng cấu hình 'secrets.toml' (nếu chạy Local) hoặc 'Streamlit Secrets' (nếu chạy Cloud).")
        st.stop()
    
    # Kiểm tra File PDF
    os.makedirs(AppConfig.PDF_DIR, exist_ok=True)
    pdf_files = glob.glob(os.path.join(AppConfig.PDF_DIR, "*.pdf"))
    if not pdf_files:
        st.error(f"⚠️ Thư mục '{AppConfig.PDF_DIR}' đang trống!")
        st.info("👉 Vui lòng upload ít nhất 1 file giáo trình PDF vào thư mục này để AI có dữ liệu.")
        st.stop()

    # --- KHỞI ĐỘNG HỆ THỐNG ---
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "👋 Chào bạn! Mình là KTC Chatbot. Hãy hỏi mình về nội dung SGK Tin học nhé!"}]

    groq_client = RAGEngine.load_groq_client()

    # Khởi tạo Retriever
    if "retriever_engine" not in st.session_state:
        with st.spinner("🚀 Đang khởi động hệ thống tri thức số... (Lần đầu sẽ mất khoảng 1-2 phút)"):
            embeddings = RAGEngine.load_embedding_model()
            st.session_state.retriever_engine = RAGEngine.build_hybrid_retriever(embeddings)
            
            # Kiểm tra lại lần nữa nếu build thất bại
            if not st.session_state.retriever_engine:
                 st.error("❌ Lỗi khởi tạo: Không thể đọc tài liệu PDF. Vui lòng kiểm tra lại Key LLAMA_CLOUD_API_KEY.")
                 st.stop()
    
    # Hiển thị Chat
    bot_avatar = AppConfig.LOGO_PROJECT if os.path.exists(AppConfig.LOGO_PROJECT) else "🤖"
    for msg in st.session_state.messages:
        role = msg["role"]
        avatar = "🧑‍🎓" if role == "user" else bot_avatar
        with st.chat_message(role, avatar=avatar):
            st.markdown(msg["content"], unsafe_allow_html=True)

    # Xử lý Input
    if user_input := st.chat_input("Nhập câu hỏi..."):
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user", avatar="🧑‍🎓"):
            st.markdown(user_input)

        with st.chat_message("assistant", avatar=bot_avatar):
            response_placeholder = st.empty()
            full_response = ""
            
            response_gen = RAGEngine.generate_response(
                groq_client,
                st.session_state.retriever_engine,
                user_input
            )
            
            for chunk in response_gen:
                full_response += chunk
                response_placeholder.markdown(full_response + "▌", unsafe_allow_html=True)
            
            response_placeholder.markdown(full_response, unsafe_allow_html=True)
            st.session_state.messages.append({"role": "assistant", "content": full_response})

if __name__ == "__main__":
    main()
import os
import glob
import base64
import streamlit as st
import shutil
import pickle
import re
import uuid
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Generator

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
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

class AppConfig:
    # Model Config
    LLM_MODEL = 'llama-3.1-8b-instant' # Tốc độ cao, phù hợp demo
    
    # Embedding & Rerank
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
    RETRIEVAL_K = 30       # Lấy rộng để lọc
    FINAL_K = 5            # Lấy top 5 cái tốt nhất để đưa vào prompt
    
    # Hybrid Search Weights
    BM25_WEIGHT = 0.4      
    FAISS_WEIGHT = 0.6     

    LLM_TEMPERATURE = 0.1 # Thấp để trả lời chính xác, ít bịa

# ===============================
# 2. XỬ LÝ GIAO DIỆN (UI MANAGER) 
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
            /* Giao diện Sidebar */
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
            
            /* Header chính */
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
            
            /* Chat message bubbles */
            [data-testid="stChatMessageContent"] {
                border-radius: 15px !important; padding: 1rem !important;
                box-shadow: 0 2px 4px rgba(0,0,0,0.05);
            }
            [data-testid="stChatMessageContent"]:has(+ [data-testid="stChatMessageAvatar"]) {
                background: #e3f2fd; color: #0d47a1; /* User messsage */
            }
            [data-testid="stChatMessageContent"]:not(:has(+ [data-testid="stChatMessageAvatar"])) {
                background: white; border: 1px solid #e9ecef; /* Bot message */
                border-left: 5px solid #00b4d8;
            }
            
            /* Style cho Source Citation [1], [2] */
            .citation-badge {
                font-size: 0.75em; vertical-align: super;
                color: #d63384; font-weight: bold; cursor: help;
                margin-left: 2px;
            }
            
            /* Footer reference section */
            .ref-section {
                font-size: 0.9rem; background: #f8f9fa; padding: 10px;
                border-radius: 8px; margin-top: 10px; border: 1px dashed #ced4da;
            }
        </style>
        """, unsafe_allow_html=True)

    @staticmethod
    def render_sidebar():
        with st.sidebar:
            if os.path.exists(AppConfig.LOGO_SCHOOL):
                st.image(AppConfig.LOGO_SCHOOL, use_container_width=True)
                st.markdown("<div style='text-align:center; font-weight:700; color:#023e8a; margin-bottom:20px;'>THCS & THPT PHẠM KIỆT</div>", unsafe_allow_html=True)

            st.markdown("""
            <div class="project-card">
                <div class="project-title">KTC CHATBOT</div>
                <div style="font-size: 0.8rem; text-align: center; color: #666; font-style: italic;">Sản phẩm KHKT 2025-2026</div>
                <hr style="margin: 10px 0; border-top: 1px dashed #dee2e6;">
                <div style="font-size: 0.9rem;">
                    <b>👨‍💻 Tác giả:</b> Bùi Tá Tùng & Cao Sỹ Bảo Chung<br>
                    <b>👨‍🏫 GVHD:</b> Thầy Nguyễn Thế Khanh
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            if st.button("🗑️ Xóa lịch sử chat", use_container_width=True):
                st.session_state.messages = []
                st.rerun()

            if st.button("🔄 Nạp lại dữ liệu", use_container_width=True):
                st.session_state.pop('retriever_engine', None)
                st.rerun()

    @staticmethod
    def render_header():
        logo_nhom_b64 = UIManager.get_img_as_base64(AppConfig.LOGO_PROJECT)
        img_html = f'<img src="data:image/jpeg;base64,{logo_nhom_b64}" style="width:100px; height:100px; border-radius:50%; border:3px solid white;">' if logo_nhom_b64 else ""

        st.markdown(f"""
        <div class="main-header">
            <div class="header-left">
                <h1>KTC CHATBOT</h1>
                <p>Trợ lý học tập thông minh & Minh bạch nguồn tin</p>
            </div>
            <div class="header-right">{img_html}</div>
        </div>
        """, unsafe_allow_html=True)

# ==================================
# 3. LOGIC BACKEND (RAG ENGINE)
# ==================================

class RAGEngine:
    @staticmethod
    @st.cache_resource(show_spinner=False)
    def load_groq_client():
        try:
            api_key = st.secrets.get("GROQ_API_KEY") or os.environ.get("GROQ_API_KEY")
            return Groq(api_key=api_key) if api_key else None
        except: return None

    @staticmethod
    @st.cache_resource(show_spinner=False)
    def load_embedding_model():
        try:
            return HuggingFaceEmbeddings(
                model_name=AppConfig.EMBEDDING_MODEL,
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
        except: return None

    @staticmethod
    @st.cache_resource(show_spinner=False)
    def load_reranker():
        try:
            return Ranker(model_name=AppConfig.RERANK_MODEL_NAME, cache_dir=AppConfig.RERANK_CACHE)
        except: return None

    # --- Xử lý dữ liệu đầu vào (Giữ nguyên logic Chunking tốt của thầy) ---
    @staticmethod
    def _structural_chunking(text: str, source_meta: dict) -> List[Document]:
        # Logic chia nhỏ văn bản theo chương/bài
        lines = text.split('\n')
        chunks = []
        current_chapter = "Chương mở đầu"
        current_lesson = "Bài mở đầu"
        buffer = []

        # Regex đơn giản hóa
        p_chapter = re.compile(r'^(CHƯƠNG|Chương)\s+[IVX0-9]+', re.IGNORECASE)
        p_lesson = re.compile(r'^(BÀI|Bài)\s+[0-9]+', re.IGNORECASE)

        def commit_chunk(buf, meta, chap, lesson):
            if not buf: return
            content = "\n".join(buf).strip()
            if len(content) < 50: return
            new_meta = meta.copy()
            new_meta.update({"chapter": chap, "lesson": lesson})
            # Lưu content kèm ngữ cảnh để search tốt hơn
            full_content = f"Context: {chap} > {lesson}\nContent: {content}"
            chunks.append(Document(page_content=full_content, metadata=new_meta))

        for line in lines:
            line_s = line.strip()
            if not line_s: continue
            
            if p_chapter.match(line_s) or line_s.startswith("# "):
                commit_chunk(buffer, source_meta, current_chapter, current_lesson)
                buffer = []
                current_chapter = line_s.replace('#', '').strip()
            elif p_lesson.match(line_s) or line_s.startswith("## "):
                commit_chunk(buffer, source_meta, current_chapter, current_lesson)
                buffer = []
                current_lesson = line_s.replace('#', '').strip()
            else:
                buffer.append(line)
        
        commit_chunk(buffer, source_meta, current_chapter, current_lesson)
        return chunks

    @staticmethod
    def _read_and_process_files() -> List[Document]:
        if not os.path.exists(AppConfig.PDF_DIR): return []
        files = glob.glob(os.path.join(AppConfig.PDF_DIR, "*.pdf"))
        all_chunks = []
        
        status = st.status("📦 Đang số hóa tài liệu...", expanded=True)
        
        for file_path in files:
            fname = os.path.basename(file_path)
            status.write(f"Đang đọc: {fname}...")
            
            # 1. Parse bằng LlamaParse (hoặc fallback nếu lỗi)
            md_path = os.path.join(AppConfig.PROCESSED_MD_DIR, f"{fname}.md")
            os.makedirs(AppConfig.PROCESSED_MD_DIR, exist_ok=True)
            
            content = ""
            if os.path.exists(md_path):
                with open(md_path, "r", encoding="utf-8") as f: content = f.read()
            else:
                try:
                    parser = LlamaParse(api_key=st.secrets.get("LLAMA_CLOUD_API_KEY"), result_type="markdown", language="vi")
                    docs = parser.load_data(file_path)
                    content = docs[0].text
                    with open(md_path, "w", encoding="utf-8") as f: f.write(content)
                except:
                    status.write(f"⚠️ Lỗi LlamaParse với {fname}, bỏ qua.")
                    continue

            # 2. Chunking
            chunks = RAGEngine._structural_chunking(content, {"source": fname})
            all_chunks.extend(chunks)
            
        status.update(label="✅ Đã xử lý xong dữ liệu!", state="complete", expanded=False)
        return all_chunks

    @staticmethod
    def build_hybrid_retriever(embeddings):
        if not embeddings: return None
        
        # Load hoặc Build Vector DB
        if os.path.exists(AppConfig.VECTOR_DB_PATH):
            try:
                vector_db = FAISS.load_local(AppConfig.VECTOR_DB_PATH, embeddings, allow_dangerous_deserialization=True)
            except: vector_db = None
        else: vector_db = None

        if not vector_db:
            docs = RAGEngine._read_and_process_files()
            if not docs: return None
            vector_db = FAISS.from_documents(docs, embeddings)
            vector_db.save_local(AppConfig.VECTOR_DB_PATH)

        # Tạo Hybrid Retriever
        bm25 = BM25Retriever.from_documents(list(vector_db.docstore._dict.values()))
        bm25.k = AppConfig.RETRIEVAL_K
        faiss_retriever = vector_db.as_retriever(search_kwargs={"k": AppConfig.RETRIEVAL_K})
        
        return EnsembleRetriever(
            retrievers=[bm25, faiss_retriever],
            weights=[AppConfig.BM25_WEIGHT, AppConfig.FAISS_WEIGHT]
        )

    # --- HÀM CORE: TẠO CÂU TRẢ LỜI CHUẨN KHOA HỌC ---
    @staticmethod
    def generate_response(client, retriever, query) -> Generator[str, None, None]:
        if not retriever:
            yield "Hệ thống đang khởi tạo..."
            return

        # 1. Retrieve & Rerank
        raw_docs = retriever.invoke(query)
        final_docs = []
        
        try: # Rerank nếu có thể
            ranker = RAGEngine.load_reranker()
            passages = [{"id": str(i), "text": d.page_content, "meta": d.metadata} for i, d in enumerate(raw_docs)]
            ranked = ranker.rank(RerankRequest(query=query, passages=passages))
            final_docs = [Document(page_content=r["text"], metadata=r["meta"]) for r in ranked[:AppConfig.FINAL_K]]
        except:
            final_docs = raw_docs[:AppConfig.FINAL_K]

        if not final_docs:
            yield "Không tìm thấy thông tin trong tài liệu."
            return

        # 2. Chuẩn bị Context với Index [1], [2]
        context_str = ""
        references = [] # Để hiển thị ở Footer
        
        for i, doc in enumerate(final_docs, 1):
            src = doc.metadata.get('source', 'TaiLieu').replace('.pdf', '')
            chap = doc.metadata.get('chapter', 'Geral')
            lesson = doc.metadata.get('lesson', '')
            
            # Chuỗi context đưa vào LLM
            context_str += f"Tài liệu [{i}]:\nNội dung: {doc.page_content}\nNguồn: {src} > {chap} > {lesson}\n\n"
            
            # Lưu metadata để hiển thị UI
            references.append(f"**[{i}] {src}**: {chap} - {lesson}")

        # 3. Prompt chuẩn ViSEF
        system_prompt = f"""Bạn là KTC Chatbot, trợ lý học tập môn Tin học.
NHIỆM VỤ: Trả lời câu hỏi dựa trên các [Tài liệu] được cung cấp bên dưới.

YÊU CẦU NGHIÊM NGẶT:
1. Chỉ sử dụng thông tin trong [Tài liệu]. Không bịa đặt.
2. TRÍCH DẪN: Mọi ý trả lời phải gắn với số thứ tự tài liệu. 
   - Ví dụ: Python là ngôn ngữ thông dịch [1]. Nó dễ học [2].
3. Trình bày Markdown đẹp, rõ ràng.
4. Nếu không có thông tin, hãy nói "Chưa có dữ liệu về vấn đề này trong SGK".

[DANH SÁCH TÀI LIỆU]:
{context_str}
"""
        
        # 4. Gọi LLM Stream
        try:
            stream = client.chat.completions.create(
                model=AppConfig.LLM_MODEL,
                messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": query}],
                stream=True, temperature=AppConfig.LLM_TEMPERATURE, max_tokens=1500
            )
            
            full_ans = ""
            for chunk in stream:
                if chunk.choices[0].delta.content:
                    text = chunk.choices[0].delta.content
                    full_ans += text
                    yield text
            
            # 5. Yield phần Danh mục tham khảo (Footer)
            yield "\n\n" + "---" + "\n"
            yield "### 📚 Tài liệu tham khảo & Minh chứng:\n"
            for ref in references:
                yield f"- {ref}\n"

        except Exception as e:
            yield f"Lỗi: {str(e)}"

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

    # Khởi tạo Session
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "Chào bạn! Tôi có thể giúp gì về môn Tin học?"}]

    # Khởi tạo RAG Engine
    if "retriever_engine" not in st.session_state:
        emb = RAGEngine.load_embedding_model()
        st.session_state.retriever_engine = RAGEngine.build_hybrid_retriever(emb)

    # Render tin nhắn cũ
    for msg in st.session_state.messages:
        avatar = "🧑‍🎓" if msg["role"] == "user" else "🤖"
        with st.chat_message(msg["role"], avatar=avatar):
            st.markdown(msg["content"], unsafe_allow_html=True)

    # Xử lý input mới
    if prompt := st.chat_input("Nhập câu hỏi..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user", avatar="🧑‍🎓"):
            st.markdown(prompt)

        with st.chat_message("assistant", avatar="🤖"):
            response_container = st.empty()
            full_response = ""
            client = RAGEngine.load_groq_client()
            
            # Chạy Generator
            generator = RAGEngine.generate_response(client, st.session_state.retriever_engine, prompt)
            
            for chunk in generator:
                full_response += chunk
                response_container.markdown(full_response + "▌")
            
            response_container.markdown(full_response) # Final render
            
            # Lưu lịch sử
            st.session_state.messages.append({"role": "assistant", "content": full_response})
            
            # --- TÍNH NĂNG MỚI: SOI MINH CHỨNG (Dành cho Giám khảo KHKT) ---
            # Lấy lại docs để hiển thị trong Expander (tuy hơi dư thừa request nhưng an toàn cho logic tách biệt)
            if st.session_state.retriever_engine:
                with st.expander("🔍 [Dành cho Giám khảo] Xem đoạn trích văn bản gốc (Raw Text)"):
                    raw_docs = st.session_state.retriever_engine.invoke(prompt)
                    for i, d in enumerate(raw_docs[:3]): # Chỉ hiện 3 cái đầu
                        st.caption(f"**Nguồn {i+1}:** {d.metadata.get('source')} (Score: Tự động)")
                        st.code(d.page_content, language="text")

if __name__ == "__main__":
    main()
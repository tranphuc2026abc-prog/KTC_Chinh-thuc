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
    nest_asyncio.apply()
    from llama_parse import LlamaParse 
    
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

# ==============================
# 1. CẤU HÌNH HỆ THỐNG
# ==============================

st.set_page_config(page_title="KTC Chatbot", page_icon="🤖", layout="wide")

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
    BM25_WEIGHT = 0.4      
    FAISS_WEIGHT = 0.6     
    LLM_TEMPERATURE = 0.0

# ===============================
# 2. XỬ LÝ GIAO DIỆN
# ===============================

class UIManager:
    @staticmethod
    def get_img_as_base64(file_path):
        if not os.path.exists(file_path): return ""
        with open(file_path, "rb") as f: data = f.read()
        return base64.b64encode(data).decode()

    @staticmethod
    def inject_custom_css():
        st.markdown("""
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;800&display=swap');
            html, body, .stMarkdown { font-family: 'Inter', sans-serif !important; }
            
            /* CSS Citation Gọn gàng - Không hiện code rác */
            .citation-footer {
                margin-top: 15px; padding: 10px 15px;
                background-color: #f1f3f5; border-radius: 8px;
                border-left: 5px solid #e03131; font-size: 0.9rem;
            }
            .citation-header {
                font-weight: 800; color: #e03131; text-transform: uppercase;
                margin-bottom: 8px; font-size: 0.85rem; letter-spacing: 0.5px;
            }
            .citation-group { margin-bottom: 4px; color: #495057; }
            .source-name { font-weight: 700; color: #1864ab; }
            .source-loc { font-style: italic; color: #495057; }
            
            #MainMenu {visibility: hidden;} footer {visibility: hidden;}
        </style>
        """, unsafe_allow_html=True)

    @staticmethod
    def render_sidebar():
        with st.sidebar:
            if os.path.exists(AppConfig.LOGO_SCHOOL):
                st.image(AppConfig.LOGO_SCHOOL, use_container_width=True)
            st.markdown("<h3 style='text-align: center; color: #003049;'>KTC CHATBOT</h3>", unsafe_allow_html=True)
            st.info("Trợ lý ảo hỗ trợ dạy và học môn Tin học - Sản phẩm KHKT.")
            if st.button("🗑️ Xóa hội thoại", use_container_width=True):
                st.session_state.messages = []
                st.rerun()

    @staticmethod
    def render_header():
        # Header đơn giản để tập trung vào chat
        pass

# ==================================
# 3. LOGIC BACKEND
# ==================================

class RAGEngine:
    @staticmethod
    @st.cache_resource(show_spinner=False)
    def load_groq_client():
        api_key = st.secrets.get("GROQ_API_KEY") or os.environ.get("GROQ_API_KEY")
        return Groq(api_key=api_key) if api_key else None

    @staticmethod
    @st.cache_resource(show_spinner=False)
    def load_embedding_model():
        return HuggingFaceEmbeddings(model_name=AppConfig.EMBEDDING_MODEL, model_kwargs={'device': 'cpu'})

    @staticmethod
    @st.cache_resource(show_spinner=False)
    def load_reranker():
        try: return Ranker(model_name=AppConfig.RERANK_MODEL_NAME, cache_dir=AppConfig.RERANK_CACHE)
        except: return None

    @staticmethod
    def _structural_chunking(text: str, source_meta: dict) -> List[Document]:
        lines = text.split('\n')
        chunks = []
        current_chapter = "Chương mở đầu"
        current_lesson = "Bài mở đầu"
        buffer = []

        p_chapter = re.compile(r'^#*\s*\**\s*(CHƯƠNG|Chương)\s+([IVX0-9]+).*$', re.IGNORECASE)
        p_lesson = re.compile(r'^#*\s*\**\s*(BÀI|Bài)\s+([0-9]+).*$', re.IGNORECASE)

        def commit_chunk(buf, meta):
            if not buf: return
            content = "\n".join(buf).strip()
            if len(content) < 50: return 
            
            new_meta = meta.copy()
            new_meta.update({
                "chunk_uid": str(uuid.uuid4())[:8],
                "chapter": current_chapter,
                "lesson": current_lesson,
                "context_str": f"{current_chapter} > {current_lesson}" 
            })
            chunks.append(Document(page_content=f"Loc: {new_meta['context_str']}\nData: {content}", metadata=new_meta))

        for line in lines:
            line_stripped = line.strip()
            if p_chapter.match(line_stripped):
                commit_chunk(buffer, source_meta)
                buffer = []
                current_chapter = line_stripped.replace('#', '').strip()
                current_lesson = "Tổng quan"
            elif p_lesson.match(line_stripped):
                commit_chunk(buffer, source_meta)
                buffer = []
                current_lesson = line_stripped.replace('#', '').strip()
            else:
                buffer.append(line)
        commit_chunk(buffer, source_meta)
        return chunks

    @staticmethod
    def _read_and_process_files(pdf_dir: str) -> List[Document]:
        if not os.path.exists(pdf_dir): return []
        files = glob.glob(os.path.join(pdf_dir, "*.pdf"))
        all_chunks = []
        status = st.empty()
        
        # Kiểm tra nếu chưa có processed folder thì dùng LlamaParse, nếu có rồi thì load cache cho nhanh
        # Ở đây giả lập logic đơn giản để code gọn
        for f in files:
            status.text(f"Đang xử lý: {os.path.basename(f)}...")
            try:
                # Fallback simple text loader nếu không có LlamaParse key
                from langchain_community.document_loaders import PyPDFLoader
                loader = PyPDFLoader(f)
                pages = loader.load()
                full_text = "\n".join([p.page_content for p in pages])
                if len(full_text) > 100:
                    all_chunks.extend(RAGEngine._structural_chunking(full_text, {"source": os.path.basename(f)}))
            except: pass
        status.empty()
        return all_chunks

    @staticmethod
    def build_hybrid_retriever(embeddings):
        if not embeddings: return None
        if os.path.exists(AppConfig.VECTOR_DB_PATH):
            try: vector_db = FAISS.load_local(AppConfig.VECTOR_DB_PATH, embeddings, allow_dangerous_deserialization=True)
            except: vector_db = None
        else:
            docs = RAGEngine._read_and_process_files(AppConfig.PDF_DIR)
            if not docs: return None
            vector_db = FAISS.from_documents(docs, embeddings)
            vector_db.save_local(AppConfig.VECTOR_DB_PATH)
        
        if not vector_db: return None
        return vector_db.as_retriever(search_kwargs={"k": AppConfig.RETRIEVAL_K})

    @staticmethod
    def generate_response(client, retriever, query) -> Generator[str, None, None]:
        if not retriever:
            yield "Hệ thống đang khởi tạo..."
            return
        
        docs = retriever.invoke(query)
        final_docs = []
        try:
            ranker = RAGEngine.load_reranker()
            if ranker and docs:
                passages = [{"id": str(i), "text": d.page_content, "meta": d.metadata} for i, d in enumerate(docs)]
                results = ranker.rank(RerankRequest(query=query, passages=passages))
                final_docs = [Document(page_content=r["text"], metadata=r["meta"]) for r in results[:AppConfig.FINAL_K]]
            else: final_docs = docs[:AppConfig.FINAL_K]
        except: final_docs = docs[:AppConfig.FINAL_K]

        if not final_docs:
            yield "Không tìm thấy thông tin trong tài liệu."
            return

        context_text = "\n".join([d.page_content for d in final_docs])
        system_prompt = f"Bạn là chatbot giáo dục. Trả lời dựa trên context sau:\n{context_text}\nKhông tự bịa nguồn."

        try:
            stream = client.chat.completions.create(
                model=AppConfig.LLM_MODEL,
                messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": query}],
                stream=True, temperature=0.1, max_tokens=1500
            )
            
            full_response = ""
            for chunk in stream:
                content = chunk.choices[0].delta.content
                if content:
                    full_response += content
                    yield content

            # --- PHẦN XỬ LÝ CITATION (ĐÃ FIX LỖI HIỂN THỊ VÀ LỌC RÁC) ---
            
            source_map = {}
            has_specific_source = False # Cờ kiểm tra xem có bài học cụ thể nào không

            for doc in final_docs:
                src_raw = doc.metadata.get('source', 'Tài liệu').replace('.pdf', '').replace('_', ' ').title()
                chapter = doc.metadata.get('chapter', '').strip()
                lesson = doc.metadata.get('lesson', '').strip()
                
                # Logic gán nhãn
                is_intro = (chapter in ["Chương mở đầu", ""]) and (lesson in ["Bài mở đầu", "Tổng quan", ""])
                
                if is_intro:
                    label = "Nội dung liên quan"
                else:
                    label = f"{chapter} - {lesson}" if chapter and lesson != "Tổng quan" else (chapter or lesson)
                    has_specific_source = True # Đánh dấu là đã tìm thấy nguồn xịn

                if src_raw not in source_map: source_map[src_raw] = set()
                source_map[src_raw].add(label)

            # --- LOGIC LỌC (FILTERING) ---
            # Nếu đã có nguồn cụ thể (has_specific_source = True), ta xóa hết các dòng "Nội dung liên quan" đi cho đỡ rác.
            if has_specific_source:
                for src in source_map:
                    if "Nội dung liên quan" in source_map[src]:
                        source_map[src].discard("Nội dung liên quan") # Xóa phần tử rác
            
            # Xóa key rỗng (nếu sau khi xóa rác mà sách đó không còn bài nào)
            source_map = {k: v for k, v in source_map.items() if v}

            # Tạo HTML (Viết liền 1 dòng để tránh lỗi hiển thị code block)
            if source_map:
                html_parts = []
                html_parts.append("<div class='citation-footer'><div class='citation-header'>📚 CĂN CỨ TÀI LIỆU:</div>")
                
                for source, labels in source_map.items():
                    label_str = "; ".join(sorted(list(labels)))
                    # F-string không xuống dòng để tránh bug Streamlit
                    html_parts.append(f"<div class='citation-group'><span class='source-name'>📘 {source}:</span> <span class='source-loc'>{label_str}</span></div>")
                
                html_parts.append("</div>")
                yield "".join(html_parts) # Yield chuỗi HTML liền mạch

        except Exception as e:
            yield f"\n[Lỗi: {str(e)}]"

# ===================
# 4. MAIN RUN
# ===================

def main():
    if not DEPENDENCIES_OK:
        st.error(f"⚠️ {IMPORT_ERROR}")
        st.stop()

    UIManager.inject_custom_css()
    UIManager.render_sidebar()

    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "👋 Chào bạn! Mình là KTC Chatbot."}]

    if "retriever_engine" not in st.session_state:
        with st.spinner("🚀 Đang tải dữ liệu..."):
            embeddings = RAGEngine.load_embedding_model()
            st.session_state.retriever_engine = RAGEngine.build_hybrid_retriever(embeddings)

    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).markdown(msg["content"], unsafe_allow_html=True)

    if prompt := st.chat_input("Nhập câu hỏi..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").markdown(prompt)

        with st.chat_message("assistant"):
            res_box = st.empty()
            full_res = ""
            client = RAGEngine.load_groq_client()
            
            gen = RAGEngine.generate_response(client, st.session_state.retriever_engine, prompt)
            for chunk in gen:
                full_res += chunk
                res_box.markdown(full_res + "▌", unsafe_allow_html=True)
            
            res_box.markdown(full_res, unsafe_allow_html=True)
            st.session_state.messages.append({"role": "assistant", "content": full_res})

if __name__ == "__main__":
    main()
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

    # RAG Parameters
    RETRIEVAL_K = 30       
    FINAL_K = 6  # Tăng nhẹ để đảm bảo đủ ngữ cảnh gộp nguồn          
    
    # Hybrid Search Weights
    BM25_WEIGHT = 0.4      
    FAISS_WEIGHT = 0.6     

    LLM_TEMPERATURE = 0.0 # Deterministic output for Science

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
            /* Style cho citation gọn gàng hơn */
            .citation-footer {
                margin-top: 20px;
                padding: 12px;
                background-color: #f8f9fa;
                border-radius: 8px;
                border-left: 4px solid #d63384;
                font-size: 0.85rem;
                color: #495057;
            }
            .citation-header {
                font-weight: 800;
                color: #d63384;
                margin-bottom: 8px;
                display: flex;
                align-items: center;
                gap: 6px;
                text-transform: uppercase;
                letter-spacing: 0.5px;
            }
            .citation-group {
                margin-bottom: 6px;
                line-height: 1.5;
            }
            .source-name {
                font-weight: 700;
                color: #0077b6;
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
            <div style="background: white; padding: 15px; border-radius: 12px; box-shadow: 0 2px 8px rgba(0,0,0,0.05); margin-bottom: 20px; border: 1px solid #dee2e6;">
                <div style="color: #0077b6; font-weight: 800; font-size: 1.1rem; margin-bottom: 5px; text-align: center; text-transform: uppercase;">KTC CHATBOT</div>
                <div style="font-size: 0.8rem; color: #6c757d; text-align: center; margin-bottom: 15px; font-style: italic;">Sản phẩm dự thi KHKT cấp Tỉnh</div>
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
            if st.button("🔄 Cập nhật dữ liệu", use_container_width=True):
                if os.path.exists(AppConfig.VECTOR_DB_PATH):
                    shutil.rmtree(AppConfig.VECTOR_DB_PATH)
                st.session_state.pop('retriever_engine', None)
                st.rerun()

    @staticmethod
    def render_header():
        logo_nhom_b64 = UIManager.get_img_as_base64(AppConfig.LOGO_PROJECT)
        img_html = f'<img src="data:image/jpeg;base64,{logo_nhom_b64}" alt="Logo" style="border-radius: 50%; border: 3px solid rgba(255,255,255,0.3); width: 100px; height: 100px; object-fit: cover;">' if logo_nhom_b64 else ""

        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #023e8a 0%, #0077b6 100%); padding: 1.5rem 2rem; border-radius: 15px; color: white; margin-bottom: 2rem; box-shadow: 0 8px 20px rgba(0, 119, 182, 0.3); display: flex; align-items: center; justify-content: space-between;">
            <div>
                <h1 style="color: #caf0f8 !important; font-weight: 900; margin: 0; font-size: 2.2rem;">KTC CHATBOT</h1>
                <p style="color: #e0fbfc; margin: 5px 0 0 0; font-size: 1rem; opacity: 0.9;">Học Tin dễ dàng - Thao tác vững vàng</p>
            </div>
            <div>{img_html}</div>
        </div>
        """, unsafe_allow_html=True)

# ==================================
# 3. LOGIC BACKEND - SMART RAG
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

    @staticmethod
    def _structural_chunking(text: str, source_meta: dict) -> List[Document]:
        lines = text.split('\n')
        chunks = []
        current_chapter = "Chương mở đầu"
        current_lesson = "Bài mở đầu"
        current_section = "Nội dung"
        buffer = []

        # Regex tối ưu hơn cho Tiếng Việt
        p_chapter = re.compile(r'^#*\s*\**\s*(CHƯƠNG|Chương)\s+([IVX0-9]+).*$', re.IGNORECASE)
        p_lesson = re.compile(r'^#*\s*\**\s*(BÀI|Bài)\s+([0-9]+).*$', re.IGNORECASE)
        p_section = re.compile(r'^(###\s+|[IV0-9]+\.\s+|[a-z]\)\s+).*')

        def clean_header(text):
            return text.replace('#', '').replace('*', '').strip()

        def commit_chunk(buf, meta):
            if not buf: return
            content = "\n".join(buf).strip()
            if len(content) < 50: return 
            
            new_meta = meta.copy()
            new_meta.update({
                "chunk_uid": str(uuid.uuid4())[:8],
                "chapter": current_chapter,
                "lesson": current_lesson,
                "section": current_section,
                "context_str": f"{current_chapter} > {current_lesson}" 
            })
            full_content = f"Location: {new_meta['context_str']}\nContent: {content}"
            chunks.append(Document(page_content=full_content, metadata=new_meta))

        for line in lines:
            line_stripped = line.strip()
            if not line_stripped: continue
            
            if p_chapter.match(line_stripped):
                commit_chunk(buffer, source_meta)
                buffer = []
                current_chapter = clean_header(line_stripped)
                current_lesson = "Tổng quan" # Reset lesson khi qua chương mới
            elif p_lesson.match(line_stripped):
                commit_chunk(buffer, source_meta)
                buffer = []
                current_lesson = clean_header(line_stripped)
            elif p_section.match(line_stripped) or line_stripped.startswith("### "):
                commit_chunk(buffer, source_meta)
                buffer = []
                current_section = clean_header(line_stripped)
            elif line_stripped.startswith("# "): 
                commit_chunk(buffer, source_meta)
                buffer = []
                current_chapter = clean_header(line_stripped)
                current_lesson = "Tổng quan"
            elif line_stripped.startswith("## "): 
                commit_chunk(buffer, source_meta)
                buffer = []
                current_lesson = clean_header(line_stripped)
            else:
                buffer.append(line)
        
        commit_chunk(buffer, source_meta)
        return chunks

    @staticmethod
    def _parse_pdf_with_llama(file_path: str) -> str:
        os.makedirs(AppConfig.PROCESSED_MD_DIR, exist_ok=True)
        file_name = os.path.basename(file_path)
        md_file_path = os.path.join(AppConfig.PROCESSED_MD_DIR, f"{file_name}.md")
        
        if os.path.exists(md_file_path):
            with open(md_file_path, "r", encoding="utf-8") as f: return f.read()
        
        llama_api_key = st.secrets.get("LLAMA_CLOUD_API_KEY")
        if not llama_api_key: return "ERROR_KEY"

        try:
            parser = LlamaParse(
                api_key=llama_api_key, result_type="markdown", language="vi", verbose=True,
                parsing_instruction="Giữ nguyên cấu trúc Chương/Bài (#, ##)."
            )
            docs = parser.load_data(file_path)
            md_text = docs[0].text
            with open(md_file_path, "w", encoding="utf-8") as f: f.write(md_text)
            return md_text
        except: return ""

    @staticmethod
    def _read_and_process_files(pdf_dir: str) -> List[Document]:
        if not os.path.exists(pdf_dir): return []
        files = glob.glob(os.path.join(pdf_dir, "*.pdf"))
        all_chunks = []
        status = st.empty()
        for f in files:
            name = os.path.basename(f)
            status.text(f"Đang xử lý: {name}...")
            content = RAGEngine._parse_pdf_with_llama(f)
            if len(content) > 50:
                all_chunks.extend(RAGEngine._structural_chunking(content, {"source": name}))
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
        
        # Hybrid Search
        docstore_docs = list(vector_db.docstore._dict.values())
        bm25 = BM25Retriever.from_documents(docstore_docs)
        bm25.k = AppConfig.RETRIEVAL_K
        faiss_ret = vector_db.as_retriever(search_type="mmr", search_kwargs={"k": AppConfig.RETRIEVAL_K})
        return EnsembleRetriever(retrievers=[bm25, faiss_ret], weights=[AppConfig.BM25_WEIGHT, AppConfig.FAISS_WEIGHT])

    @staticmethod
    def generate_response(client, retriever, query) -> Generator[str, None, None]:
        if not retriever:
            yield "Hệ thống đang khởi tạo..."
            return
        
        # 1. Retrieval & Rerank
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

        # 2. Build Context
        context_text = "\n".join([f"--- SOURCE ---\n{d.page_content}" for d in final_docs])

        # 3. Prompt
        system_prompt = f"""Bạn là KTC Chatbot. Trả lời dựa trên [CONTEXT].
Nguyên tắc:
1. Thông tin phải từ [CONTEXT].
2. Nếu không có, nói không biết.
3. Không tự bịa nguồn [ID:...].
4. Trình bày rõ ràng.

[CONTEXT]
{context_text}"""

        try:
            stream = client.chat.completions.create(
                model=AppConfig.LLM_MODEL,
                messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": query}],
                stream=True, temperature=AppConfig.LLM_TEMPERATURE, max_tokens=1500
            )
            
            full_response = ""
            for chunk in stream:
                content = chunk.choices[0].delta.content
                if content:
                    full_response += content
                    yield content

            # --- 4. SMART CITATION GROUPING (KHỬ RỐI) ---
            # Mục tiêu: Gom các Bài thuộc cùng 1 Sách lại với nhau
            
            source_map = {} # { "Tên sách": set("Bài 1", "Bài 2") }

            for doc in final_docs:
                # Làm sạch tên sách
                src_raw = doc.metadata.get('source', 'Tài liệu tham khảo')
                src_clean = src_raw.replace('.pdf', '').replace('_', ' ').title()
                
                # Lấy vị trí (Chương/Bài)
                chapter = doc.metadata.get('chapter', '').strip()
                lesson = doc.metadata.get('lesson', '').strip()
                
                # Logic xác định nhãn vị trí (Label)
                is_default_chap = chapter in ["Chương mở đầu", ""]
                is_default_less = lesson in ["Bài mở đầu", "Tổng quan", "Tổng quan chương", ""]
                
                label = ""
                if not is_default_less:
                    # Ưu tiên hiển thị bài học cụ thể
                    label = f"{lesson}" 
                    # Nếu tên bài ngắn quá (ví dụ "Bài 1"), có thể ghép thêm chương cho rõ
                    if len(lesson) < 10 and not is_default_chap:
                        label = f"{chapter} - {lesson}"
                elif not is_default_chap:
                    label = chapter
                else:
                    label = "Nội dung liên quan"

                if src_clean not in source_map:
                    source_map[src_clean] = set()
                source_map[src_clean].add(label)

            # Tạo HTML Footer Gom nhóm
            if source_map:
                html = "\n\n<div class='citation-footer'><div class='citation-header'>📚 CĂN CỨ TÀI LIỆU:</div>"
                
                # Duyệt qua từng sách
                for source, labels in source_map.items():
                    # Sắp xếp các label (Bài 1, Bài 2...)
                    sorted_labels = sorted(list(labels))
                    label_str = "; ".join(sorted_labels)
                    
                    # Dòng hiển thị: Tên sách (đậm) -> Các bài (thường)
                    html += f"""
                    <div class='citation-group'>
                        <span class='source-name'>📘 {source}:</span> 
                        <span class='source-loc'>{label_str}</span>
                    </div>
                    """
                html += "</div>"
                yield html

        except Exception as e:
            yield f"\n[Lỗi hệ thống: {str(e)}]"

# ===================
# 4. MAIN RUN
# ===================

def main():
    if not DEPENDENCIES_OK:
        st.error(f"⚠️ {IMPORT_ERROR}")
        st.stop()

    UIManager.inject_custom_css()
    UIManager.render_sidebar()
    UIManager.render_header()

    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "👋 KTC Chatbot sẵn sàng hỗ trợ!"}]

    if "retriever_engine" not in st.session_state:
        with st.spinner("🚀 Khởi động hệ thống tri thức số..."):
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
            
            # Streaming response
            gen = RAGEngine.generate_response(client, st.session_state.retriever_engine, prompt)
            for chunk in gen:
                full_res += chunk
                res_box.markdown(full_res + "▌", unsafe_allow_html=True)
            
            res_box.markdown(full_res, unsafe_allow_html=True)
            st.session_state.messages.append({"role": "assistant", "content": full_res})

if __name__ == "__main__":
    main()
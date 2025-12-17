import os
import glob
import base64
import streamlit as st
import shutil
import pickle
import re
import uuid
import hashlib
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
    FINAL_K = 7            # Tăng nhẹ để đảm bảo đủ ngữ cảnh
    
    # Hybrid Search Weights
    BM25_WEIGHT = 0.4      
    FAISS_WEIGHT = 0.6     

    LLM_TEMPERATURE = 0.0 # BẮT BUỘC = 0.0 để triệt tiêu sáng tạo ảo giác

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
            /* Style cho Citation chuẩn KHKT */
            .citation-source {
                font-size: 0.75em;
                color: #ffffff; 
                background-color: #d63384; /* Màu hồng đậm */
                padding: 2px 8px;
                border-radius: 10px;
                font-weight: 600;
                margin-left: 4px;
                display: inline-block;
                vertical-align: middle;
                cursor: help;
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

# ==================================
# 3. LOGIC BACKEND - VERIFIABLE HYBRID RAG
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
    def _structural_chunking(text: str, source_meta: dict) -> List[Document]:
        lines = text.split('\n')
        chunks = []
        
        current_chapter = "Chương mở đầu"
        current_lesson = "Bài mở đầu"
        current_section = "Nội dung"
        
        buffer = []

        # --- REGEX PATTERNS CHO SGK VIỆT NAM ---
        p_chapter = re.compile(r'^#*\s*\**\s*(CHƯƠNG|Chương)\s+([IVX0-9]+).*$', re.IGNORECASE)
        p_lesson = re.compile(r'^#*\s*\**\s*(BÀI|Bài)\s+([0-9]+).*$', re.IGNORECASE)
        p_section = re.compile(r'^(###\s+|[IV0-9]+\.\s+|[a-z]\)\s+).*')

        def clean_header(text):
            return text.replace('#', '').replace('*', '').strip()

        def commit_chunk(buf, meta):
            if not buf: return
            content = "\n".join(buf).strip()
            if len(content) < 50: return 
            
            # DETERMINISTIC CHUNK UID: SHA256 trên nội dung + nguồn + chapter + lesson + section
            hash_input = (meta.get("source","") + "|" + current_chapter + "|" + current_lesson + "|" + current_section + "|" + content).encode('utf-8')
            chunk_hash = hashlib.sha256(hash_input).hexdigest()[:12]  # 12 hex chars đủ phân biệt
                        
            new_meta = meta.copy()
            new_meta.update({
                "chunk_uid": chunk_hash,
                "chapter": current_chapter,
                "lesson": current_lesson,
                "section": current_section,
                "context_str": f"{current_chapter} > {current_lesson} > {current_section}" 
            })
            
            chunks.append(Document(page_content=content, metadata=new_meta))

        for line in lines:
            line_stripped = line.strip()
            if not line_stripped: continue
            
            if p_chapter.match(line_stripped):
                commit_chunk(buffer, source_meta)
                buffer = []
                current_chapter = clean_header(line_stripped)
                current_lesson = "Tổng quan chương"
                current_section = "Giới thiệu"
            
            elif p_lesson.match(line_stripped):
                commit_chunk(buffer, source_meta)
                buffer = []
                current_lesson = clean_header(line_stripped)
                current_section = "Tổng quan bài"
                
            elif p_section.match(line_stripped) or line_stripped.startswith("### "):
                commit_chunk(buffer, source_meta)
                buffer = []
                current_section = clean_header(line_stripped)
                
            elif line_stripped.startswith("# "): 
                commit_chunk(buffer, source_meta)
                buffer = []
                current_chapter = clean_header(line_stripped)
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
            with open(md_file_path, "r", encoding="utf-8") as f:
                return f.read()
        
        llama_api_key = st.secrets.get("LLAMA_CLOUD_API_KEY")
        if not llama_api_key:
            return "ERROR: Missing LLAMA_CLOUD_API_KEY"

        try:
            parser = LlamaParse(
                api_key=llama_api_key,
                result_type="markdown",
                language="vi",
                verbose=True,
                parsing_instruction="Đây là tài liệu giáo khoa Tin học. Hãy giữ nguyên định dạng bảng biểu, code block và cấu trúc chương mục (#, ##, ###)."
            )
            documents = parser.load_data(file_path)
            markdown_text = documents[0].text
            
            with open(md_file_path, "w", encoding="utf-8") as f:
                f.write(markdown_text)
            
            return markdown_text
        except Exception as e:
            return f"Error parsing {file_name}: {str(e)}"

    @staticmethod
    def _read_and_process_files(pdf_dir: str) -> List[Document]:
        if not os.path.exists(pdf_dir):
            return []
        
        pdf_files = glob.glob(os.path.join(pdf_dir, "*.pdf"))
        all_chunks: List[Document] = []
        status_text = st.empty()

        for file_path in pdf_files:
            source_file = os.path.basename(file_path)
            status_text.text(f"Đang xử lý cấu trúc tri thức: {source_file}...")
            
            markdown_content = RAGEngine._parse_pdf_with_llama(file_path)
            
            if "ERROR" not in markdown_content and len(markdown_content) > 50:
                 meta = {
                     "source": source_file, 
                 }
                 file_chunks = RAGEngine._structural_chunking(markdown_content, meta)
                 all_chunks.extend(file_chunks)
            else:
                pass 
                
        status_text.empty()
        return all_chunks

    @staticmethod
    def build_hybrid_retriever(embeddings):
        if not embeddings: return None

        vector_db = None
        if os.path.exists(AppConfig.VECTOR_DB_PATH):
            try:
                vector_db = FAISS.load_local(AppConfig.VECTOR_DB_PATH, embeddings, allow_dangerous_deserialization=True)
            except Exception: pass

        if not vector_db:
            chunk_docs = RAGEngine._read_and_process_files(AppConfig.PDF_DIR)
            if not chunk_docs:
                st.error(f"Không tìm thấy tài liệu hoặc lỗi xử lý trong {AppConfig.PDF_DIR}")
                return None
            
            vector_db = FAISS.from_documents(chunk_docs, embeddings)
            vector_db.save_local(AppConfig.VECTOR_DB_PATH)

        try:
            docstore_docs = list(vector_db.docstore._dict.values())
            bm25_retriever = BM25Retriever.from_documents(docstore_docs)
            bm25_retriever.k = AppConfig.RETRIEVAL_K

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
    def _sanitize_output(text: str) -> str:
        cjk_pattern = re.compile(r'[\u4e00-\u9fff\u3400-\u4dbf\u3040-\u309f\u30a0-\u30ff\uac00-\ud7af]+')
        text = cjk_pattern.sub("", text) 
        return text

    # =========================================================================
    # STRICT RAG GENERATION LOGIC - PHIÊN BẢN KIỂM CHỨNG KHKT
    # =========================================================================
    @staticmethod
    def generate_response(client, retriever, query) -> Generator[str, None, None]:
        if not retriever:
            yield "Hệ thống đang khởi tạo... vui lòng chờ giây lát."
            return
        
        # --- 1. STRICT RETRIEVAL & RERANK ---
        initial_docs = retriever.invoke(query)
        
        # Định tuyến ưu tiên: SGK > SGV > Tài liệu khác
        scored_docs = []
        for doc in initial_docs:
            src = doc.metadata.get('source', '')
            bonus = 0.0
            if "KNTT" in src: bonus = 2.0  # Ưu tiên cực cao cho SGK KNTT
            elif "_GV_" in src: bonus = 0.5
            elif "ON THI" in src: bonus = 0.2
            
            scored_docs.append({"doc": doc, "bonus": bonus})
            
        final_docs = []
        try:
            ranker = RAGEngine.load_reranker()
            if ranker and scored_docs:
                passages = [
                    {"id": str(i), "text": item["doc"].page_content, "meta": item["doc"].metadata} 
                    for i, item in enumerate(scored_docs)
                ]
                rerank_req = RerankRequest(query=query, passages=passages)
                results = ranker.rank(rerank_req)
                
                final_results_with_score = []
                for res in results:
                    original_idx = int(res['id'])
                    bonus = scored_docs[original_idx]['bonus']
                    ai_score = res['score']
                    # Công thức trọng số: AI Score + Bonus SGK
                    total_score = ai_score + (bonus * 0.3) 
                    final_results_with_score.append((res, total_score))
                
                final_results_with_score.sort(key=lambda x: x[1], reverse=True)
                top_k = final_results_with_score[:AppConfig.FINAL_K]
                final_docs = [Document(page_content=r[0]['text'], metadata=r[0]['meta']) for r in top_k]
            else:
                scored_docs.sort(key=lambda x: x['bonus'], reverse=True)
                final_docs = [item["doc"] for item in scored_docs][:AppConfig.FINAL_K]
        except Exception:
            final_docs = [item["doc"] for item in scored_docs][:AppConfig.FINAL_K]

        if not final_docs:
            yield "Không tìm thấy thông tin phù hợp trong SGK hiện có."
            return

        # --- 2. XÂY DỰNG REGISTRY NGUỒN (KIỂM SOÁT METADATA) ---
        valid_uids = {} # map uid -> display string "(Nguồn: <Tên SGK> – <Chương> – <Bài>)"
        context_parts = []
        
        for doc in final_docs:
            uid = doc.metadata.get('chunk_uid')
            if not uid: continue
            
            src_raw = doc.metadata.get('source', '')
            clean_name = src_raw.replace('.pdf', '').replace('_', ' ').strip()
            chapter = doc.metadata.get('chapter', '').strip() or "Chương"
            lesson = doc.metadata.get('lesson', '').strip() or doc.metadata.get('section','').strip() or "Bài"
            
            # Lưu vào registry để validate sau này
            display_label = f"(Nguồn: {clean_name} – {chapter} – {lesson})"
            valid_uids[uid] = display_label
            
            # Context format chặt chẽ để LLM nhìn thấy ID
            context_parts.append(
                f"<chunk id='{uid}'>\n{doc.page_content}\n</chunk>"
            )

        full_context = "\n".join(context_parts)

        # --- 3. PROMPT KỸ THUẬT NGHIÊM NGẶT (STRICT PROMPT) ---
        system_prompt = f"""Bạn là Trợ lý AI giáo dục chuẩn KHKT. Nhiệm vụ: Trả lời câu hỏi CHỈ DỰA TRÊN các chunk dữ liệu được cung cấp.

QUY TẮC BẮT BUỘC (TUÂN THỦ 100%):
1. KHÔNG BỊA ĐẶT: Nếu thông tin không có trong context, trả lời "NO_INFO".
2. TRÍCH DẪN NGAY LẬP TỨC: Mỗi câu hoặc mỗi ý kiến kiến thức phải kết thúc bằng đúng một ID chunk.
   - **Chỉ** dùng định dạng: [ID:<chunk_uid>]
   - **Không** dùng tên sách, ngoặc khác, hoặc bất kỳ dạng citation nào khác.
3. KHÔNG GỘP ID: Mỗi ý chỉ được phép một ID.
4. ƯU TIÊN SGK: Nếu có mâu thuẫn, lấy thông tin từ chunk có nguồn SGK.

DỮ LIỆU CONTEXT:
{full_context}
"""
        
        try:
            completion = client.chat.completions.create(
                model=AppConfig.LLM_MODEL,
                messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": query}],
                temperature=0.0, # Deterministic - Không sáng tạo
                stream=False
            )
            raw_response = completion.choices[0].message.content.strip()
            
            if "NO_INFO" in raw_response:
                yield "Không tìm thấy thông tin phù hợp trong SGK hiện có."
                return

            # --- 4. HẬU KIỂM NGHIÊM NGẶT (AUDIT LAYER) ---
            processed_response = raw_response

            # Regex tìm tất cả [ID:xxxxxxxx]
            pattern = r'

\[ID:([a-fA-F0-9]+)\]

'
            matches = list(re.finditer(pattern, processed_response))
            cited_ids = [m.group(1) for m in matches]

            # 4.a Nếu có ID không tồn tại trong registry -> HỦY TOÀN BỘ
            for cid in cited_ids:
                if cid not in valid_uids:
                    yield "Không tìm thấy thông tin phù hợp trong SGK hiện có."
                    return

            # 4.b Kiểm tra mỗi câu/ý phải có đúng 1 citation
            # Tách câu bằng dấu chấm hỏi, chấm than, chấm, xuống dòng
            sentence_separators = re.compile(r'(?<=[\.\?\!\n])\s+')
            sentences = sentence_separators.split(processed_response)
            # Nếu không tách được, fallback tách theo newline
            if len(sentences) == 1:
                sentences = processed_response.split('\n')

            for sent in sentences:
                # Bỏ khoảng trắng
                s = sent.strip()
                # Nếu câu rỗng hoặc chỉ chứa ký tự không chữ thì bỏ qua
                if not re.search(r'[A-Za-zÀ-ỹ0-9]', s):
                    continue
                # Đếm số citation trong câu
                count_ids = len(re.findall(pattern, s))
                if count_ids != 1:
                    # Nếu câu có 0 hoặc >1 citation -> HỦY TOÀN BỘ
                    yield "Không tìm thấy thông tin phù hợp trong SGK hiện có."
                    return

            # 4.c Nếu mọi kiểm tra OK -> thay thế mỗi [ID:...] bằng display_label (không lộ chunk_uid)
            # Duyệt từ trái sang phải và thay thế
            def replace_id(match):
                uid_found = match.group(1)
                return valid_uids.get(uid_found, "")

            final_display = re.sub(pattern, replace_id, processed_response)

            # Trả về kết quả đã map (không thêm cảnh báo hay text phụ)
            yield final_display

        except Exception as e:
            # Không tiết lộ lỗi nội bộ, trả về chuỗi hủy theo quy tắc nếu là lỗi liên quan kiểm chứng
            yield "Không tìm thấy thông tin phù hợp trong SGK hiện có."

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
        st.session_state.messages = [{"role": "assistant", "content": "👋 Chào bạn! KTC Chatbot sẵn sàng hỗ trợ tra cứu kiến thức SGK Tin học."}]

    groq_client = RAGEngine.load_groq_client()

    if "retriever_engine" not in st.session_state:
        with st.spinner("🚀 Đang khởi động hệ thống tri thức số (LlamaParse + Semantic Chunking)..."):
            embeddings = RAGEngine.load_embedding_model()
            st.session_state.retriever_engine = RAGEngine.build_hybrid_retriever(embeddings)
            if st.session_state.retriever_engine:
                st.toast("✅ Dữ liệu SGK đã sẵn sàng!", icon="📚")

    for msg in st.session_state.messages:
        bot_avatar = AppConfig.LOGO_PROJECT if os.path.exists(AppConfig.LOGO_PROJECT) else "🤖"
        avatar = "🧑‍🎓" if msg["role"] == "user" else bot_avatar
        with st.chat_message(msg["role"], avatar=avatar):
            st.markdown(msg["content"], unsafe_allow_html=True) 

    user_input = st.chat_input("Nhập câu hỏi học tập...")
    
    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user", avatar="🧑‍🎓"):
            st.markdown(user_input)

        with st.chat_message("assistant", avatar=AppConfig.LOGO_PROJECT if os.path.exists(AppConfig.LOGO_PROJECT) else "🤖"):
            response_placeholder = st.empty()
            
            response_gen = RAGEngine.generate_response(
                groq_client,
                st.session_state.retriever_engine,
                user_input
            )

            full_response = ""
            for chunk in response_gen:
                full_response += chunk
                response_placeholder.markdown(full_response + "▌", unsafe_allow_html=True)
            
            response_placeholder.markdown(full_response, unsafe_allow_html=True)

            st.session_state.messages.append({"role": "assistant", "content": full_response})

if __name__ == "__main__":
    main()

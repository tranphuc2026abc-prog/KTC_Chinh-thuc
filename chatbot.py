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
    FINAL_K = 5            
    
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
                font-size: 0.8em;
                color: #ffffff; 
                background-color: #d63384; /* Màu hồng đậm nổi bật */
                padding: 3px 8px;
                border-radius: 12px;
                font-weight: 600;
                margin-left: 5px;
                display: inline-block;
                box-shadow: 0 1px 2px rgba(0,0,0,0.1);
                border: none;
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
# 3. LOGIC BACKEND - ADAPTIVE KNOWLEDGE ROUTING (UPDATED)
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
            
            chunk_uid = str(uuid.uuid4())[:8] # Generate 8-char UID
            
            new_meta = meta.copy()
            new_meta.update({
                "chunk_uid": chunk_uid,
                "chapter": current_chapter,
                "lesson": current_lesson,
                "section": current_section,
                "context_str": f"{current_chapter} > {current_lesson} > {current_section}" 
            })
            
            # Lưu ý: Không nhét metadata vào page_content để tránh nhiễu ngữ nghĩa
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

    # --- ĐÂY LÀ PHẦN CỐT LÕI ĐƯỢC NÂNG CẤP (ADAPTIVE GENERATOR) ---
    @staticmethod
    def generate_response(client, retriever, query) -> Generator[str, None, None]:
        if not retriever:
            yield "Hệ thống đang khởi tạo... vui lòng chờ giây lát."
            return
        
        # --- PHASE 1: TRUY XUẤT (Retrieval) ---
        initial_docs = retriever.invoke(query)
        
        # --- PHASE 2: TÍNH ĐIỂM TRỌNG SỐ (ADAPTIVE WEIGHTING) ---
        # Tự động gán điểm thưởng dựa trên tên file
        scored_docs = []
        for doc in initial_docs:
            src = doc.metadata.get('source', '')
            
            # --- LOGIC ĐỊNH TUYẾN TRI THỨC ---
            priority_score = 0.0
            
            # Ưu tiên 1: Sách Giáo Khoa (KNTT)
            if "KNTT" in src:
                priority_score = 1.0  
                
            # Ưu tiên 2: Sách Giáo Viên (GV)
            elif "_GV_" in src:
                priority_score = 0.5  
            
            # Ưu tiên 3: Ôn tập (ON THI)
            elif "ON THI" in src:
                priority_score = 0.3
                
            # Các tài liệu tham khảo khác (PythonCoban, etc.) -> Không cộng điểm
            else:
                priority_score = 0.0  
            
            scored_docs.append({
                "doc": doc,
                "bonus": priority_score
            })
            
        # --- PHASE 3: RERANKING (AI + Rule Base) ---
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
                
                # Lai ghép điểm số: AI Score + Priority Bonus
                reranked_results = []
                for res in results:
                    original_idx = int(res['id'])
                    bonus = scored_docs[original_idx]['bonus']
                    ai_score = res['score'] # AI đánh giá độ khớp câu hỏi
                    
                    # Công thức "Bẻ lái" AI về phía SGK
                    # Nếu điểm AI ~ 0.8, Bonus SGK = 1.0 -> Final = 1.2
                    # Nếu điểm AI ~ 0.9 (PythonCoban), Bonus = 0 -> Final = 0.9
                    # => SGK thắng
                    final_score = ai_score + (bonus * 0.4) 
                    
                    reranked_results.append({
                        "result": res,
                        "final_score": final_score
                    })
                
                # Sắp xếp lại theo điểm tổng
                reranked_results.sort(key=lambda x: x['final_score'], reverse=True)
                
                # Lấy Top K (Đã ưu tiên SGK)
                top_k = reranked_results[:AppConfig.FINAL_K]
                final_docs = [Document(page_content=r['result']['text'], metadata=r['result']['meta']) for r in top_k]
            else:
                # Fallback: Chỉ sắp xếp theo Bonus nếu Reranker lỗi
                scored_docs.sort(key=lambda x: x['bonus'], reverse=True)
                final_docs = [item["doc"] for item in scored_docs][:AppConfig.FINAL_K]
        except Exception:
            final_docs = [item["doc"] for item in scored_docs][:AppConfig.FINAL_K]

        if not final_docs:
            yield "Không tìm thấy thông tin phù hợp trong tài liệu hiện có."
            return

        # --- PHASE 4: XÂY DỰNG CONTEXT & PROMPT ---
        citation_registry = {} 
        context_parts = []

        for doc in final_docs:
            uid = doc.metadata.get('chunk_uid')
            if not uid: continue
            
            # Xử lý hiển thị tên nguồn (Cleanup name)
            src_raw = doc.metadata.get('source', '')
            clean_name = src_raw.replace('.pdf', '').replace('_', ' ')
            
            # Gán nhãn hiển thị (UI Badge) - Tự động hóa
            badge = "📄"
            if "KNTT" in src_raw: 
                badge = "📘 SGK"  # Icon sách xanh cho SGK
            elif "GV" in src_raw: 
                badge = "📙 SGV"  # Icon sách cam cho SGV
            elif "Python" in src_raw: 
                badge = "🐍 Code" # Icon rắn cho Python
            
            # Lấy thông tin bài học
            lesson = doc.metadata.get('lesson', '').replace('Bài', 'B.').strip()
            citation_registry[uid] = f"{badge} {clean_name} > {lesson}"
            
            context_parts.append(
                f"--- BEGIN CHUNK ---\nID: {uid}\nCONTENT: {doc.page_content}\n--- END CHUNK ---"
            )

        full_context = "\n".join(context_parts)

        # Prompt đặc tả cho Gemini - Ép tuân thủ ưu tiên
        system_prompt = f"""Bạn là KTC Chatbot - Trợ lý học tập chuẩn SGK.
NHIỆM VỤ: Trả lời câu hỏi dựa trên [CONTEXT].

QUY TẮC BẮT BUỘC (QUAN TRỌNG):
1. ƯU TIÊN SỐ 1: Thông tin từ các chunk có tên chứa 'KNTT' hoặc 'SGK'.
2. CHỈ sử dụng thông tin từ tài liệu khác (như PythonCoban) nến SGK không đề cập.
3. Nếu [CONTEXT] có mâu thuẫn giữa SGK và tài liệu khác, PHẢI CHỌN SGK.
4. Cuối câu trả lời, BẮT BUỘC ghi mã tham chiếu duy nhất: [FINAL_REF:xxxxxxxx].
5. Chọn ID của chunk có chứa thông tin chính xác nhất (ưu tiên SGK) để làm tham chiếu.

[CONTEXT]
{full_context}
"""
        
        try:
            completion = client.chat.completions.create(
                model=AppConfig.LLM_MODEL,
                messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": query}],
                temperature=AppConfig.LLM_TEMPERATURE,
                stream=False
            )
            raw_response = completion.choices[0].message.content.strip()
            
            # Hậu xử lý (Hiển thị nguồn)
            cleaned_response = RAGEngine._sanitize_output(raw_response)
            pattern_final_ref = r'\[FINAL_REF:([a-zA-Z0-9]{8})\]'
            match = re.search(pattern_final_ref, cleaned_response)
            
            if match and match.group(1) in citation_registry:
                uid = match.group(1)
                content = re.sub(pattern_final_ref, '', cleaned_response).strip()
                # Render HTML nguồn đẹp
                source_html = f"<div style='margin-top:10px; text-align:right;'><span class='citation-source'>{citation_registry[uid]}</span></div>"
                yield content + source_html
            else:
                # Fallback mềm: Vẫn hiện nội dung nhưng cảnh báo nhẹ
                clean_text = re.sub(pattern_final_ref, '', cleaned_response).strip()
                if len(clean_text) > 10:
                    yield clean_text + "\n\n*(Nguồn: Tổng hợp từ dữ liệu học tập)*"
                else:
                    yield "Xin lỗi, tôi chưa tìm thấy thông tin chính xác trong SGK."

        except Exception as e:
            yield f"Lỗi hệ thống: {str(e)}"

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
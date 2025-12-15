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

    # RAG Parameters (Updated for Semantic Chunking)
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
# 3. LOGIC BACKEND - VERIFIABLE HYBRID RAG (AUDIT GRADE)
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
    def _detect_grade(filename: str) -> str:
        filename = filename.lower()
        if "10" in filename: return "10"
        if "11" in filename: return "11"
        if "12" in filename: return "12"
        return "general"

    @staticmethod
    def _structural_chunking(text: str, source_meta: dict) -> List[Document]:
        lines = text.split('\n')
        chunks = []
        
        current_chapter = "General"
        current_lesson = "General"
        current_section = "General"
        
        buffer = []
        
        def commit_chunk(buf, meta):
            if not buf: return
            content = "\n".join(buf).strip()
            if len(content) < 50: return 
            
            chunk_uid = str(uuid.uuid4())[:8]
            
            new_meta = meta.copy()
            new_meta.update({
                "chunk_uid": chunk_uid,
                "chapter": current_chapter,
                "lesson": current_lesson,
                "section": current_section,
                "context_str": f"{current_chapter} > {current_lesson} > {current_section}" 
            })
            
            # Context explicitly embedded for retrieval match
            full_content = f"Context: {new_meta['context_str']}\nContent: {content}"
            chunks.append(Document(page_content=full_content, metadata=new_meta))

        for line in lines:
            line = line.strip()
            if not line: continue
            
            if line.startswith("# "): 
                commit_chunk(buffer, source_meta)
                buffer = []
                current_chapter = line.replace("# ", "").strip()
                current_lesson = "Intro"
                current_section = "Intro"
            elif line.startswith("## "): 
                commit_chunk(buffer, source_meta)
                buffer = []
                current_lesson = line.replace("## ", "").strip()
                current_section = "Intro"
            elif line.startswith("### "): 
                commit_chunk(buffer, source_meta)
                buffer = []
                current_section = line.replace("### ", "").strip()
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
            return "ERROR: Missing LLAMA_CLOUD_API_KEY in secrets"

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
                     "grade": RAGEngine._detect_grade(source_file)
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
    def generate_response(client, retriever, query) -> Generator[str, None, None]:
        """
        Quy trình Verifiable Hybrid RAG Chuẩn KHKT Quốc Gia
        Mô hình: Evidence-Bound Generation (Ràng buộc minh chứng)
        """
        if not retriever:
            yield "Hệ thống đang khởi tạo... vui lòng chờ giây lát."
            return
        
        # --- TẦNG 1: RETRIEVAL & RERANK ---
        initial_docs = retriever.invoke(query)
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
            yield "Không tìm thấy thông tin phù hợp trong SGK hiện có."
            return

        # --- TẦNG 2: EVIDENCE REGISTRY (Tạo sổ cái minh chứng) ---
        valid_uids = set()
        uid_to_citation_text = {}
        context_parts = []

        for doc in final_docs:
            uid = doc.metadata.get('chunk_uid')
            if not uid: continue
            
            src_raw = doc.metadata.get('source', '')
            src_clean = src_raw.replace('.pdf', '').replace('_', ' ')
            chapter = doc.metadata.get('chapter', 'Chương không xác định')
            lesson = doc.metadata.get('lesson', 'Bài không xác định')
            
            # Format hiển thị chuẩn KHKT: (Nguồn: Sách - Chương - Bài)
            citation_display = f"(Nguồn: {src_clean} – {chapter} – {lesson})"
            
            valid_uids.add(uid)
            uid_to_citation_text[uid] = citation_display
            
            # Context chặt chẽ, buộc LLM nhìn thấy ID ngay cạnh nội dung
            context_parts.append(
                f"--- BEGIN CHUNK ---\nID: {uid}\n{doc.page_content}\n--- END CHUNK ---"
            )

        full_context = "\n".join(context_parts)

        # --- TẦNG 3: ATOMIC FACT PROMPTING (Ngăn chặn Source Collapse) ---
        # Prompt này ép LLM xé nhỏ câu trả lời để mỗi ý phải gắn đúng 1 ID.
        system_prompt = f"""Bạn là KTC Chatbot - Động cơ kiểm chứng tri thức giáo dục.
Nhiệm vụ: Trả lời câu hỏi dựa trên [CONTEXT] dưới dạng CÁC MỆNH ĐỀ ĐƠN (Atomic Claims).

QUY TẮC BẮT BUỘC (Strict Execution Rules):
1. KHÔNG được trả lời kiểu văn xuôi liên tục. Hãy tách ý rõ ràng.
2. MỖI mệnh đề/câu thông tin phải kết thúc ngay lập tức bằng ID nguồn.
   Định dạng bắt buộc: Nội dung thông tin... [ID: <chunk_id>]
3. TUYỆT ĐỐI KHÔNG gộp nhiều ý khác nguồn vào một câu. Tách chúng ra.
4. KHÔNG tự bịa ID. Chỉ dùng ID có trong "BEGIN CHUNK" của [CONTEXT].
5. KHÔNG tự viết tên sách/bài học vào câu trả lời. Hệ thống sẽ tự điền dựa trên ID.
6. Nếu không có thông tin trong Context, ghi: "NO_INFO".

[CONTEXT]
{full_context}
"""
        
        try:
            completion = client.chat.completions.create(
                model=AppConfig.LLM_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": query}
                ],
                stream=False,
                temperature=AppConfig.LLM_TEMPERATURE,
                max_tokens=1500
            )
            raw_response = completion.choices[0].message.content

            # --- TẦNG 4: STRICT VALIDATION & MAPPING (Tránh Citation Drift) ---
            
            if "NO_INFO" in raw_response or not raw_response.strip():
                yield "Không tìm thấy thông tin phù hợp trong SGK hiện có."
                return

            # Regex tìm ID
            pattern_id = r'\[ID:\s*([a-zA-Z0-9]{8})\]'
            
            # Validation: Nếu câu trả lời có ID "ma" (không có trong context) -> Cảnh báo
            found_ids = re.findall(pattern_id, raw_response)
            valid_response = True
            for fid in found_ids:
                if fid not in valid_uids:
                    # Phát hiện Hallucinated Citation -> Hủy phản hồi để đảm bảo tính khoa học
                    valid_response = False
                    break
            
            if not valid_response:
                yield "Hệ thống phát hiện lỗi trích dẫn không xác thực. Vui lòng thử lại."
                return

            # Mapping Presentation: Biến ID khô khan thành Nguồn đọc được
            # Sử dụng hàm callback để thay thế từng ID tìm thấy
            def citation_mapper(match):
                uid = match.group(1)
                if uid in uid_to_citation_text:
                    # In đậm nguồn để người dùng dễ tra cứu
                    return f"\n> **{uid_to_citation_text[uid]}**" 
                return "" # Xóa ID lỗi nếu lọt qua validation

            # Thực hiện thay thế
            final_response = re.sub(pattern_id, citation_mapper, raw_response)
            
            # Làm sạch định dạng dư thừa (nếu có)
            final_response = final_response.replace("--- BEGIN CHUNK ---", "").replace("--- END CHUNK ---", "")
            
            yield final_response

        except Exception as e:
            yield f"Lỗi xử lý hệ thống: {str(e)}"

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
            st.markdown(msg["content"])

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
                response_placeholder.markdown(full_response + "▌")
            
            response_placeholder.markdown(full_response)

            st.session_state.messages.append({"role": "assistant", "content": full_response})

if __name__ == "__main__":
    main()
import os
import glob
import base64
import streamlit as st
import shutil
from pathlib import Path

# --- Imports với xử lý lỗi ---
try:
    from pypdf import PdfReader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_community.vectorstores import FAISS
    from langchain_community.retrievers import BM25Retriever 
    from langchain.retrievers import EnsembleRetriever      
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_core.documents import Document
    from groq import Groq
    # Thư viện Rerank
    from flashrank import Ranker, RerankRequest
    DEPENDENCIES_OK = True
except ImportError as e:
    DEPENDENCIES_OK = False
    IMPORT_ERROR = str(e)

# ==============================================================================
# 1. CẤU HÌNH HỆ THỐNG (CONFIG)
# ==============================================================================

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
    
    # Assets
    LOGO_PROJECT = "LOGO.jpg"
    LOGO_SCHOOL = "LOGO PKS.png"
    
    # [TINH CHỈNH QUAN TRỌNG ĐỂ KHÔNG BỊ LỆCH NGUỒN]
    CHUNK_SIZE = 800        
    CHUNK_OVERLAP = 100     
    # Tăng số lượng tìm kiếm thô lên để cơ hội trúng Tin 12 cao hơn
    RETRIEVAL_K = 30        
    FINAL_K = 5             
    
    # Trọng số: Giảm nhẹ BM25 (Từ khóa) vì sách cũ hay lặp từ khóa
    # Tăng Semantic (Ngữ nghĩa)
    WEIGHT_BM25 = 0.3       
    WEIGHT_FAISS = 0.7      

# ==============================================================================
# 2. XỬ LÝ GIAO DIỆN (UI MANAGER)
# ==============================================================================

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
            html, body, [class*="css"], .stMarkdown, .stButton, .stTextInput, .stChatInput { font-family: 'Inter', sans-serif !important; }
            section[data-testid="stSidebar"] { background-color: #f8f9fa; border-right: 1px solid #e9ecef; }
            .project-card { background: white; padding: 15px; border-radius: 12px; box-shadow: 0 2px 8px rgba(0,0,0,0.05); margin-bottom: 20px; border: 1px solid #dee2e6; }
            .project-title { color: #0077b6; font-weight: 800; font-size: 1.1rem; margin-bottom: 5px; text-align: center; text-transform: uppercase; }
            .project-sub { font-size: 0.8rem; color: #6c757d; text-align: center; margin-bottom: 15px; font-style: italic; }
            .main-header { background: linear-gradient(135deg, #023e8a 0%, #0077b6 100%); padding: 1.5rem 2rem; border-radius: 15px; color: white; margin-bottom: 2rem; display: flex; align-items: center; justify-content: space-between; }
            .header-left h1 { color: #caf0f8 !important; font-weight: 900; margin: 0; font-size: 2.2rem; }
            [data-testid="stChatMessageContent"] { border-radius: 15px !important; padding: 1rem !important; }
            div.stButton > button:hover { background-color: #0077b6; color: white; border-color: #0077b6; }
            #MainMenu {visibility: hidden;} footer {visibility: hidden;}
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
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            with st.expander("📂 Tính năng nâng cao (AI Vision)", expanded=False):
                st.markdown("<small>Tải ảnh lỗi code hoặc file ghi âm</small>", unsafe_allow_html=True)
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
            <div class="header-right">{img_html}</div>
        </div>
        """, unsafe_allow_html=True)

# ==============================================================================
# 3. LOGIC BACKEND (RAG ENGINE + RERANK + MULTIMODAL)
# ==============================================================================

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
            return HuggingFaceEmbeddings(model_name=AppConfig.EMBEDDING_MODEL, model_kwargs={'device': 'cpu'})
        except: return None

    @staticmethod
    @st.cache_resource(show_spinner=False)
    def load_reranker():
        try:
            return Ranker(model_name=AppConfig.RERANK_MODEL_NAME, cache_dir=AppConfig.RERANK_CACHE)
        except: return None

    @staticmethod
    def build_hybrid_retriever(embeddings):
        """
        [FIXED] Sử dụng MMR để đảm bảo tính đa dạng, tránh bị lệch nguồn vào 1 cuốn sách.
        """
        if not embeddings: return None

        vector_db = None
        if os.path.exists(AppConfig.VECTOR_DB_PATH):
            try:
                vector_db = FAISS.load_local(AppConfig.VECTOR_DB_PATH, embeddings, allow_dangerous_deserialization=True)
            except Exception as e: st.warning(f"Reload DB lỗi: {e}")

        # Tạo mới DB nếu chưa có
        if not vector_db:
            if not os.path.exists(AppConfig.PDF_DIR): return None
            
            # Đọc cả PDF và TXT
            files = glob.glob(os.path.join(AppConfig.PDF_DIR, "*.*"))
            docs = []
            
            # Hiển thị tiến trình để GV biết file nào được nạp
            status_bar = st.progress(0)
            status_text = st.empty()
            
            for idx, file_path in enumerate(files):
                filename = os.path.basename(file_path)
                status_text.text(f"Đang đọc: {filename}...")
                
                try:
                    # Logic đọc file đơn giản
                    content_chunks = []
                    if file_path.lower().endswith('.pdf'):
                        reader = PdfReader(file_path)
                        for page_num, page in enumerate(reader.pages):
                            txt = page.extract_text()
                            if txt and len(txt) > 50:
                                # [QUAN TRỌNG] Gắn tên file vào nội dung để AI dễ nhận biết
                                meta = {"source": filename, "page": page_num + 1}
                                content_chunks.append(Document(page_content=f"Tài liệu: {filename}\n{txt}", metadata=meta))
                    
                    if content_chunks:
                        docs.extend(content_chunks)
                except Exception as e:
                    print(f"Lỗi đọc {filename}: {e}")
                
                status_bar.progress((idx + 1) / len(files))

            status_text.empty()
            status_bar.empty()

            if docs:
                splitter = RecursiveCharacterTextSplitter(chunk_size=AppConfig.CHUNK_SIZE, chunk_overlap=AppConfig.CHUNK_OVERLAP)
                splits = splitter.split_documents(docs)
                vector_db = FAISS.from_documents(splits, embeddings)
                vector_db.save_local(AppConfig.VECTOR_DB_PATH)
            else:
                return None
        
        # --- CẤU HÌNH HYBRID SEARCH CHUẨN KHKT ---
        try:
            docstore_docs = list(vector_db.docstore._dict.values())
            bm25_retriever = BM25Retriever.from_documents(docstore_docs)
            bm25_retriever.k = AppConfig.RETRIEVAL_K 
            
            # [THAY ĐỔI QUAN TRỌNG] Dùng search_type="mmr" thay vì similarity
            # MMR (Maximal Marginal Relevance) sẽ tìm các đoạn khác nhau, tránh lấy tập trung 1 chỗ
            faiss_retriever = vector_db.as_retriever(
                search_type="mmr", 
                search_kwargs={
                    "k": AppConfig.RETRIEVAL_K, 
                    "fetch_k": 50,          # Lấy 50 đoạn thô để lọc ra 20 đoạn đa dạng
                    "lambda_mult": 0.5      # 0.5 là cân bằng giữa Giống nhau và Đa dạng
                }
            )
            
            ensemble_retriever = EnsembleRetriever(
                retrievers=[bm25_retriever, faiss_retriever],
                weights=[AppConfig.WEIGHT_BM25, AppConfig.WEIGHT_FAISS]
            )
            return ensemble_retriever
        except:
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
                            {"type": "text", "text": "Mô tả chi tiết code hoặc nội dung trong ảnh này."},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                        ]
                    }]
                )
                vision_desc = resp.choices[0].message.content
            except: vision_desc = "Lỗi đọc ảnh."
        elif uploaded_file.type.startswith('audio'):
            try:
                with open("temp_audio.mp3", "wb") as f: f.write(uploaded_file.getbuffer())
                with open("temp_audio.mp3", "rb") as f:
                    transcription = client.audio.transcriptions.create(
                        file=("temp_audio.mp3", f.read()),
                        model=AppConfig.LLM_AUDIO_MODEL,
                        response_format="text"
                    )
                audio_text = transcription
            except: audio_text = "Lỗi nghe âm thanh."
        return vision_desc, audio_text

    @staticmethod
    def generate_response(client, retriever, query, vision_context=None):
        context_text = ""
        sources = []
        
        # 1. Hybrid Retrieval (BM25 + MMR)
        if retriever:
            initial_docs = retriever.invoke(query)
            
            # 2. Rerank (FlashRank)
            final_docs = []
            try:
                ranker = RAGEngine.load_reranker()
                if ranker and initial_docs:
                    passages = [{"id": str(i), "text": doc.page_content, "meta": doc.metadata} for i, doc in enumerate(initial_docs)]
                    rerank_request = RerankRequest(query=query, passages=passages)
                    ranked_results = ranker.rank(rerank_request)
                    top_results = ranked_results[:AppConfig.FINAL_K]
                    for res in top_results:
                        final_docs.append(Document(page_content=res['text'], metadata=res['meta']))
                else: final_docs = initial_docs[:AppConfig.FINAL_K]
            except: final_docs = initial_docs[:AppConfig.FINAL_K]

            for doc in final_docs:
                src = doc.metadata.get('source', 'Tài liệu')
                page = doc.metadata.get('page', 'Unknown')
                content = doc.page_content.replace("\n", " ").strip()
                # Định dạng rõ ràng để AI hiểu đây là nguồn khác nhau
                context_text += f"\n--- TÀI LIỆU: {src} (Trang {page}) ---\n{content}\n"
                sources.append(f"{src} - Trang {page}")

        extra_instruct = ""
        if vision_context:
            extra_instruct = f"Học sinh có gửi kèm ảnh/code: '{vision_context}'. Kết hợp để trả lời."

        # Prompt ép AI chú ý tên tài liệu
        system_prompt = f"""Bạn là KTC Chatbot - Trợ lý AI giáo dục.
        
        NHIỆM VỤ:
        1. Trả lời dựa trên [NGỮ CẢNH] bên dưới.
        2. [QUAN TRỌNG] Nếu câu hỏi liên quan đến chương trình Tin 10, 11, 12, hãy ƯU TIÊN tìm thông tin trong các file tương ứng (ví dụ: TIN 12_KNTT, TIN 11...).
        3. Nếu câu hỏi về kỹ năng văn phòng chung, mới dùng tài liệu Tin VP.
        4. Trích dẫn nguồn chính xác [Tên_File - Trang X].
        
        {extra_instruct}
        
        [NGỮ CẢNH TÌM ĐƯỢC TỪ CSDL]:
        {context_text}
        """

        try:
            stream = client.chat.completions.create(
                model=AppConfig.LLM_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": query}
                ],
                stream=True, temperature=0.3, max_tokens=2000
            )
            return stream, sorted(list(set(sources)))
        except Exception as e: return f"Lỗi: {str(e)}", []

# ==============================================================================
# 4. MAIN APPLICATION
# ==============================================================================

def main():
    if not DEPENDENCIES_OK:
        st.error(f"⚠️ Lỗi thư viện: {IMPORT_ERROR}")
        st.stop()
        
    UIManager.inject_custom_css()
    UIManager.render_sidebar()
    UIManager.render_header()

    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "👋 Chào bạn! KTC Chatbot (v2.0 Hybrid) sẵn sàng hỗ trợ."}]
    
    groq_client = RAGEngine.load_groq_client()
    
    if "retriever_engine" not in st.session_state:
        with st.spinner("🚀 Đang khởi tạo hệ thống Hybrid Search & MMR..."):
            embeddings = RAGEngine.load_embedding_model()
            st.session_state.retriever_engine = RAGEngine.build_hybrid_retriever(embeddings)
            if st.session_state.retriever_engine: st.toast("✅ Dữ liệu sẵn sàng!", icon="📚")

    for msg in st.session_state.messages:
        bot_avatar = AppConfig.LOGO_PROJECT if os.path.exists(AppConfig.LOGO_PROJECT) else "🤖"
        avatar = "🧑‍🎓" if msg["role"] == "user" else bot_avatar
        with st.chat_message(msg["role"], avatar=avatar): st.markdown(msg["content"])

    if len(st.session_state.messages) < 2:
        st.markdown("##### 💡 Gợi ý ôn tập:")
        cols = st.columns(3)
        if cols[0].button("🐍 Python cơ bản"): st.session_state.temp_input = "Cấu trúc rẽ nhánh if-else trong Python (dựa trên Tin 10/11)."
        if cols[1].button("🤖 Trí tuệ nhân tạo"): st.session_state.temp_input = "AI là gì? Nêu một số ứng dụng của AI (dựa trên Tin 12)."
        if cols[2].button("🌐 Mạng máy tính"): st.session_state.temp_input = "Các thiết bị mạng thông dụng (dựa trên Tin 12)."
        if "temp_input" in st.session_state: st.rerun()

    if "temp_input" in st.session_state:
        user_input = st.session_state.temp_input
        del st.session_state.temp_input
    else: user_input = st.chat_input("Nhập câu hỏi của bạn...")

    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user", avatar="🧑‍🎓"): st.markdown(user_input)

        with st.chat_message("assistant", avatar=AppConfig.LOGO_PROJECT if os.path.exists(AppConfig.LOGO_PROJECT) else "🤖"):
            response_placeholder = st.empty()
            if not groq_client: st.error("❌ Chưa cấu hình API Key.")
            else:
                vision_context = None
                if "uploaded_file_obj" in st.session_state and st.session_state.uploaded_file_obj:
                    with st.status("🖼️ Đang phân tích...", expanded=False):
                        vision_desc, audio_text = RAGEngine.process_multimodal(groq_client, st.session_state.uploaded_file_obj)
                        if audio_text: user_input += f" (Voice: {audio_text})"
                        vision_context = vision_desc

                stream, sources = RAGEngine.generate_response(groq_client, st.session_state.retriever_engine, user_input, vision_context)
                
                full_response = ""
                if isinstance(stream, str): response_placeholder.error(stream)
                else:
                    for chunk in stream:
                        if chunk.choices[0].delta.content:
                            full_response += chunk.choices[0].delta.content
                            response_placeholder.markdown(full_response + "▌")
                    response_placeholder.markdown(full_response)
                
                if sources:
                    with st.expander("📚 Nguồn tài liệu (Đa dạng hóa)"):
                        for src in sources: st.markdown(f"- 📖 *{src}*")
                
                st.session_state.messages.append({"role": "assistant", "content": full_response})
                if "uploaded_file_obj" in st.session_state and st.session_state.uploaded_file_obj:
                    st.session_state.uploaded_file_obj = None

if __name__ == "__main__":
    main()
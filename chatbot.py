import os
import glob
import base64
import streamlit as st
import shutil
from pathlib import Path
import time

# --- Imports với xử lý lỗi ---
try:
    from pypdf import PdfReader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_community.vectorstores import FAISS
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_core.documents import Document
    from langchain.retrievers import EnsembleRetriever
    from langchain_community.retrievers import BM25Retriever
    from groq import Groq
    from flashrank import Ranker, RerankRequest
    DEPENDENCIES_OK = True
except ImportError as e:
    DEPENDENCIES_OK = False
    IMPORT_ERROR = str(e)

# ==============================================================================
# 1. CẤU HÌNH HỆ THỐNG (CONFIG) - CHUẨN KHKT
# ==============================================================================

st.set_page_config(
    page_title="KTC Chatbot - THCS & THPT Phạm Kiệt",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

class AppConfig:
    # Model Config - Cập nhật model Multimodal mới nhất của Groq
    LLM_TEXT_MODEL = 'llama-3.1-8b-instant'
    LLM_VISION_MODEL = 'llama-3.2-11b-vision-preview' # Model nhìn được ảnh
    LLM_AUDIO_MODEL = 'whisper-large-v3'              # Model nghe âm thanh
    
    EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    RERANK_MODEL_NAME = "ms-marco-TinyBERT-L-2-v2"
    
    # Paths
    PDF_DIR = "PDF_KNOWLEDGE"
    VECTOR_DB_PATH = "faiss_db_index"
    RERANK_CACHE = "./opt"
    
    # Assets
    LOGO_PROJECT = "LOGO.jpg"
    LOGO_SCHOOL = "LOGO PKS.png"
    
    # Hybrid RAG Parameters (Công nghệ lõi)
    CHUNK_SIZE = 800        # Giảm size để chính xác hơn
    CHUNK_OVERLAP = 150     
    RETRIEVAL_K = 20        
    FINAL_K = 5             
    WEIGHT_BM25 = 0.4       # Trọng số tìm kiếm từ khóa
    WEIGHT_FAISS = 0.6      # Trọng số tìm kiếm ngữ nghĩa

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
            html, body, .stMarkdown, .stButton, .stTextInput { font-family: 'Inter', sans-serif !important; }
            .project-card { background: white; padding: 15px; border-radius: 12px; border: 1px solid #dee2e6; box-shadow: 0 2px 8px rgba(0,0,0,0.05); margin-bottom: 20px; }
            .project-title { color: #0077b6; font-weight: 800; text-align: center; text-transform: uppercase; }
            .main-header { background: linear-gradient(135deg, #023e8a 0%, #0077b6 100%); padding: 1.5rem; border-radius: 15px; color: white; display: flex; justify-content: space-between; align-items: center; margin-bottom: 2rem; }
            .header-left h1 { color: #caf0f8; font-weight: 900; font-size: 2.2rem; margin: 0; }
            [data-testid="stChatMessageContent"] { border-radius: 15px !important; padding: 1rem !important; }
            div.stButton > button:hover { background-color: #0077b6; color: white; border-color: #0077b6; }
        </style>
        """, unsafe_allow_html=True)

    @staticmethod
    def render_sidebar():
        with st.sidebar:
            if os.path.exists(AppConfig.LOGO_SCHOOL):
                st.image(AppConfig.LOGO_SCHOOL, width=100)
            
            st.markdown("""
            <div class="project-card">
                <div class="project-title">KTC CHATBOT</div>
                <div style="text-align:center; font-size: 0.8rem; color: #666; font-style:italic;">Hybrid RAG & Multimodal AI</div>
                <hr>
                <small><b>Tác giả:</b> Bùi Tá Tùng - Cao Sỹ Bảo Chung<br>
                <b>GVHD:</b> Thầy Nguyễn Thế Khanh</small>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("### 🧠 Đa phương thức (Multimodal)")
            with st.expander("📸 Tải lên Ảnh/Code/Voice", expanded=True):
                uploaded_file = st.file_uploader("Chọn file (Ảnh lỗi code, Sơ đồ, Ghi âm)", type=['png', 'jpg', 'jpeg', 'mp3', 'wav', 'py'])
                if uploaded_file:
                    st.session_state.uploaded_file = uploaded_file
                    st.success(f"Đã nhận file: {uploaded_file.name}")
            
            st.markdown("---")
            if st.button("🗑️ Xóa lịch sử chat", use_container_width=True):
                st.session_state.messages = []
                st.session_state.uploaded_file = None
                st.rerun()

# ==============================================================================
# 3. LOGIC BACKEND (HYBRID RAG + MULTIMODAL)
# ==============================================================================

class RAGEngine:
    @staticmethod
    @st.cache_resource
    def load_groq_client():
        api_key = st.secrets.get("GROQ_API_KEY") or os.environ.get("GROQ_API_KEY")
        return Groq(api_key=api_key) if api_key else None

    @staticmethod
    @st.cache_resource
    def load_embedding_model():
        return HuggingFaceEmbeddings(model_name=AppConfig.EMBEDDING_MODEL)

    @staticmethod
    @st.cache_resource
    def load_reranker():
        return Ranker(model_name=AppConfig.RERANK_MODEL_NAME, cache_dir=AppConfig.RERANK_CACHE)

    @staticmethod
    def build_or_load_retriever(embeddings):
        # 1. Load Vector DB (FAISS)
        vector_store = None
        if os.path.exists(AppConfig.VECTOR_DB_PATH):
            vector_store = FAISS.load_local(AppConfig.VECTOR_DB_PATH, embeddings, allow_dangerous_deserialization=True)
        else:
            # Logic tạo mới DB (rút gọn để tập trung vào logic chính)
            if not os.path.exists(AppConfig.PDF_DIR): return None
            docs = []
            files = glob.glob(os.path.join(AppConfig.PDF_DIR, "*.*"))
            for f in files:
                if f.endswith('.pdf'):
                    reader = PdfReader(f)
                    for i, page in enumerate(reader.pages):
                        txt = page.extract_text()
                        if txt: docs.append(Document(page_content=txt, metadata={"source": os.path.basename(f), "page": i+1}))
            
            if docs:
                splitter = RecursiveCharacterTextSplitter(chunk_size=AppConfig.CHUNK_SIZE, chunk_overlap=AppConfig.CHUNK_OVERLAP)
                splits = splitter.split_documents(docs)
                vector_store = FAISS.from_documents(splits, embeddings)
                vector_store.save_local(AppConfig.VECTOR_DB_PATH)
        
        if not vector_store: return None

        # 2. Tạo Hybrid Retriever (FAISS + BM25) -> ĐIỂM SÁNG KHOA HỌC
        # Lấy lại documents từ vector store để tạo BM25
        # Lưu ý: Trong thực tế nên cache BM25 riêng, nhưng demo thì load từ docstore
        try:
            docstore_docs = list(vector_store.docstore._dict.values())
            bm25_retriever = BM25Retriever.from_documents(docstore_docs)
            bm25_retriever.k = AppConfig.RETRIEVAL_K

            faiss_retriever = vector_store.as_retriever(search_kwargs={"k": AppConfig.RETRIEVAL_K})

            ensemble_retriever = EnsembleRetriever(
                retrievers=[bm25_retriever, faiss_retriever],
                weights=[AppConfig.WEIGHT_BM25, AppConfig.WEIGHT_FAISS]
            )
            return ensemble_retriever
        except:
            return vector_store.as_retriever(search_kwargs={"k": AppConfig.RETRIEVAL_K})

    @staticmethod
    def process_multimodal_input(client, uploaded_file, user_query):
        """Xử lý Ảnh và Âm thanh"""
        vision_content = None
        audio_transcript = None
        
        # Xử lý Ảnh
        if uploaded_file.type in ['image/png', 'image/jpeg', 'image/jpg']:
            # Encode ảnh
            base64_image = base64.b64encode(uploaded_file.getvalue()).decode('utf-8')
            # Gọi Vision Model để mô tả ảnh
            try:
                chat_completion = client.chat.completions.create(
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": "Mô tả chi tiết nội dung trong bức ảnh này liên quan đến Tin học/Lập trình. Nếu là code lỗi, hãy chỉ ra lỗi."},
                                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
                            ],
                        }
                    ],
                    model=AppConfig.LLM_VISION_MODEL,
                )
                vision_content = chat_completion.choices[0].message.content
            except Exception as e:
                vision_content = f"Lỗi đọc ảnh: {str(e)}"

        # Xử lý Âm thanh (Whisper)
        elif uploaded_file.type in ['audio/mpeg', 'audio/wav', 'audio/mp3']:
            try:
                # Cần lưu tạm file để gửi vào API
                with open("temp_audio.mp3", "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                with open("temp_audio.mp3", "rb") as file:
                    transcription = client.audio.transcriptions.create(
                        file=("temp_audio.mp3", file.read()),
                        model=AppConfig.LLM_AUDIO_MODEL,
                        response_format="text",
                        language="vi"
                    )
                audio_transcript = transcription
            except Exception as e:
                audio_transcript = f"Lỗi nghe: {str(e)}"
        
        # Xử lý File Python
        elif uploaded_file.name.endswith('.py'):
             vision_content = f"Nội dung file code học sinh upload:\n```python\n{uploaded_file.getvalue().decode('utf-8')}\n```"

        return vision_content, audio_transcript

    @staticmethod
    def generate_response(client, retriever, query, vision_context=None):
        # 1. Retrieval (Hybrid Search)
        docs = retriever.invoke(query)
        
        # 2. Rerank (FlashRank)
        try:
            ranker = RAGEngine.load_reranker()
            passages = [{"id": str(i), "text": doc.page_content, "meta": doc.metadata} for i, doc in enumerate(docs)]
            rerank_request = RerankRequest(query=query, passages=passages)
            ranked_results = ranker.rank(rerank_request)[:AppConfig.FINAL_K]
            final_docs = [Document(page_content=r['text'], metadata=r['meta']) for r in ranked_results]
        except:
            final_docs = docs[:AppConfig.FINAL_K]

        # 3. Context Construction
        context_text = ""
        sources = []
        for doc in final_docs:
            src = doc.metadata.get('source', 'Unknown')
            page = doc.metadata.get('page', '?')
            context_text += f"\n[Nguồn: {src} - Tr {page}]: {doc.page_content}"
            sources.append(f"{src} - Trang {page}")

        # 4. System Prompt (Pedagogical & Multimodal)
        multimodal_instruction = ""
        if vision_context:
            multimodal_instruction = f"Học sinh có gửi kèm hình ảnh/code với nội dung sau: '{vision_context}'. Hãy kết hợp nội dung này để trả lời."

        system_prompt = f"""Bạn là KTC Chatbot - Trợ lý AI dạy Tin học giỏi cấp Quốc gia.
        
        NHIỆM VỤ:
        1. Trả lời câu hỏi dựa trên [KIẾN THỨC SGK] được cung cấp bên dưới.
        2. Nếu câu hỏi về lập trình, hãy đóng vai 'Reviewer' (Người hướng dẫn): Chỉ ra lỗi sai, giải thích nguyên lý, KHÔNG viết code giải bài tập về nhà thay cho học sinh.
        3. Phong cách: Sư phạm, khích lệ, ngắn gọn.
        
        {multimodal_instruction}
        
        [KIẾN THỨC SGK]:
        {context_text}
        """

        try:
            stream = client.chat.completions.create(
                model=AppConfig.LLM_TEXT_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": query}
                ],
                stream=True,
                temperature=0.3
            )
            return stream, sorted(list(set(sources)))
        except Exception as e:
            return str(e), []

# ==============================================================================
# 4. MAIN RUNTIME
# ==============================================================================

def main():
    UIManager.inject_custom_css()
    UIManager.render_sidebar()
    
    # Header
    logo_b64 = UIManager.get_img_as_base64(AppConfig.LOGO_PROJECT)
    img_tag = f'<img src="data:image/png;base64,{logo_b64}" style="width:80px;border-radius:50%;">' if logo_b64 else ""
    st.markdown(f"""
    <div class="main-header">
        <div class="header-left">
            <h1>KTC CHATBOT AI</h1>
            <p>Trợ lý học tập Tin học :: Hybrid RAG & Vision</p>
        </div>
        {img_tag}
    </div>
    """, unsafe_allow_html=True)

    # Init State
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "Chào em! Thầy là trợ lý AI. Em cần hỏi về bài học hay muốn thầy xem giúp đoạn code nào?"}]
    if "uploaded_file" not in st.session_state:
        st.session_state.uploaded_file = None

    # Load Resources
    groq_client = RAGEngine.load_groq_client()
    embeddings = RAGEngine.load_embedding_model()
    
    if "retriever" not in st.session_state:
        with st.spinner("🚀 Đang khởi tạo hệ thống Hybrid Search..."):
            st.session_state.retriever = RAGEngine.build_or_load_retriever(embeddings)

    # Display Chat
    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).markdown(msg["content"])

    # Chat Input
    if user_input := st.chat_input("Nhập câu hỏi của bạn..."):
        st.session_state.messages.append({"role": "user", "content": user_input})
        st.chat_message("user").markdown(user_input)

        with st.chat_message("assistant"):
            if not groq_client:
                st.error("❌ Chưa kết nối API Groq.")
                st.stop()
            
            # Xử lý Multimodal nếu có file upload
            vision_context = None
            if st.session_state.uploaded_file:
                with st.status("🖼️ Đang phân tích file đính kèm...", expanded=False):
                    vision_context, audio_text = RAGEngine.process_multimodal_input(groq_client, st.session_state.uploaded_file, user_input)
                    if audio_text: # Nếu là file âm thanh, thay thế user_input bằng text đã dịch
                        st.info(f"🎙️ Nội dung ghi âm: {audio_text}")
                        user_input = f"{user_input} (Nội dung nói: {audio_text})"
            
            # Generate Response
            response_placeholder = st.empty()
            stream, sources = RAGEngine.generate_response(
                groq_client, 
                st.session_state.retriever, 
                user_input, 
                vision_context
            )
            
            full_response = ""
            if isinstance(stream, str):
                response_placeholder.error(stream)
            else:
                for chunk in stream:
                    if chunk.choices[0].delta.content:
                        full_response += chunk.choices[0].delta.content
                        response_placeholder.markdown(full_response + "▌")
                response_placeholder.markdown(full_response)
            
            # Show Sources
            if sources:
                with st.expander("📚 Căn cứ khoa học (Trích dẫn SGK)"):
                    for src in sources: st.markdown(f"- {src}")

            st.session_state.messages.append({"role": "assistant", "content": full_response})
            # Reset file sau khi xử lý xong
            if st.session_state.uploaded_file:
                st.session_state.uploaded_file = None

if __name__ == "__main__":
    main()
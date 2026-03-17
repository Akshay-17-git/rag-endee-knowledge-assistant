import streamlit as st
import os
import time
from pathlib import Path
from utils.document_processor import process_document
from utils.embeddings import get_embedding
from utils.endee_client import EndeeClient
from utils.llm import generate_answer

st.set_page_config(
    page_title="AI Knowledge Assistant | Endee",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Space Grotesk', sans-serif;
    }
    
    .main-header {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
        padding: 2rem;
        border-radius: 12px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
    }
    
    .main-header h1 { font-size: 2.5rem; margin: 0; }
    .main-header p { opacity: 0.8; margin: 0.5rem 0 0; }
    
    .metric-card {
        background: #f8f9ff;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
    }
    
    .answer-box {
        background: #f0f4ff;
        border-left: 4px solid #4f46e5;
        border-radius: 8px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
    
    .chunk-card {
        background: white;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
        font-size: 0.875rem;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #4f46e5, #7c3aed);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 2rem;
        font-weight: 600;
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if "endee_client" not in st.session_state:
    st.session_state.endee_client = EndeeClient()
if "documents_indexed" not in st.session_state:
    st.session_state.documents_indexed = 0
if "chunks_indexed" not in st.session_state:
    st.session_state.chunks_indexed = 0
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Header
st.markdown("""
<div class="main-header">
    <h1>🧠 AI Knowledge Assistant</h1>
    <p>Powered by Endee Vector Database + Mistral RAG Pipeline</p>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    
    mistral_key = st.text_input(
        "Mistral API Key",
        type="password",
        placeholder="Enter your Mistral API key",
        help="Get your key at console.mistral.ai"
    )
    
    if mistral_key:
        os.environ["MISTRAL_API_KEY"] = mistral_key
        st.success("✅ API Key set")
    
    st.markdown("---")
    st.markdown("### 📊 Index Stats")
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Documents", st.session_state.documents_indexed)
    with col2:
        st.metric("Chunks", st.session_state.chunks_indexed)
    
    st.markdown("---")
    st.markdown("### ⚡ Endee Settings")
    top_k = st.slider("Top-K Results", 1, 10, 3)
    chunk_size = st.slider("Chunk Size (words)", 50, 300, 150)
    
    st.markdown("---")
    if st.button("🗑️ Clear Index"):
        st.session_state.endee_client.clear()
        st.session_state.documents_indexed = 0
        st.session_state.chunks_indexed = 0
        st.session_state.chat_history = []
        st.success("Index cleared!")
        st.rerun()

# Main tabs
tab1, tab2, tab3 = st.tabs(["📄 Upload & Index", "💬 Ask Questions", "🎯 Interview Prep"])

# ─── Tab 1: Upload & Index ────────────────────────────────────────────────────
with tab1:
    st.markdown("### Upload Documents")
    st.markdown("Upload PDFs or text files to index into the Endee vector database.")
    
    uploaded_files = st.file_uploader(
        "Choose files",
        type=["pdf", "txt"],
        accept_multiple_files=True
    )
    
    # Demo documents
    st.markdown("#### Or try a demo document:")
    demo_col1, demo_col2 = st.columns(2)
    
    with demo_col1:
        if st.button("📚 Load AI/ML Article"):
            demo_text = """
            Artificial Intelligence and Machine Learning Overview
            
            Artificial intelligence (AI) refers to the simulation of human intelligence in machines 
            programmed to think like humans and mimic their actions. Machine learning (ML) is a 
            subset of AI that provides systems the ability to automatically learn and improve from 
            experience without being explicitly programmed.
            
            Deep Learning is a subfield of machine learning that uses neural networks with many 
            layers. These deep neural networks can learn representations of data with multiple 
            levels of abstraction. Convolutional Neural Networks (CNNs) are particularly effective 
            for image recognition tasks. Recurrent Neural Networks (RNNs) and LSTMs are suited 
            for sequential data like text and time series.
            
            Natural Language Processing (NLP) is a branch of AI that helps computers understand, 
            interpret, and manipulate human language. Large Language Models (LLMs) like GPT and 
            Mistral are trained on vast amounts of text data and can generate human-like text,
            answer questions, and perform various language tasks.
            
            Vector databases are specialized databases designed to store and query high-dimensional 
            vector embeddings efficiently. They enable semantic search by finding vectors that are 
            most similar to a query vector using distance metrics like cosine similarity or 
            Euclidean distance. Endee is a high-performance vector database optimized for 
            production AI systems.
            
            Retrieval-Augmented Generation (RAG) combines retrieval systems with generative AI.
            Instead of relying solely on parametric knowledge, RAG systems retrieve relevant 
            documents from a knowledge base and use them as context for generation. This improves 
            accuracy and allows models to work with up-to-date information.
            
            Embeddings are dense numerical representations of data (text, images, etc.) in a 
            high-dimensional space where semantically similar items are placed close together.
            Sentence transformers are models that create embeddings where the entire sentence's 
            meaning is captured in a single vector.
            """
            chunks = process_document(demo_text, "AI_ML_Article", chunk_size)
            for chunk in chunks:
                embedding = get_embedding(chunk["text"])
                st.session_state.endee_client.insert(chunk["id"], embedding, chunk)
            st.session_state.documents_indexed += 1
            st.session_state.chunks_indexed += len(chunks)
            st.success(f"✅ Demo document indexed: {len(chunks)} chunks stored in Endee!")
    
    with demo_col2:
        if st.button("🐍 Load Python Guide"):
            demo_text2 = """
            Python Programming Guide
            
            Python is a high-level, interpreted programming language known for its simplicity 
            and readability. It was created by Guido van Rossum and first released in 1991.
            Python emphasizes code readability with its notable use of significant indentation.
            
            Python supports multiple programming paradigms including procedural, object-oriented,
            and functional programming. Its comprehensive standard library is often cited as one 
            of its greatest strengths.
            
            Key Python features include dynamic typing, automatic memory management through 
            garbage collection, and a large standard library. Python's package manager pip 
            makes it easy to install third-party libraries.
            
            Popular Python frameworks for web development include Django and Flask. For data 
            science and machine learning, NumPy, Pandas, Scikit-learn, TensorFlow, and PyTorch 
            are widely used. Streamlit is a popular framework for building data science 
            web applications quickly.
            
            Python decorators are a powerful feature that allows modifying or enhancing functions 
            without changing their source code. List comprehensions provide a concise way to 
            create lists. Generators and iterators enable lazy evaluation of sequences.
            
            Virtual environments in Python help manage project-specific dependencies. Tools like
            venv, virtualenv, and conda are commonly used. Docker containers provide an even 
            more isolated environment for Python applications.
            """
            chunks = process_document(demo_text2, "Python_Guide", chunk_size)
            for chunk in chunks:
                embedding = get_embedding(chunk["text"])
                st.session_state.endee_client.insert(chunk["id"], embedding, chunk)
            st.session_state.documents_indexed += 1
            st.session_state.chunks_indexed += len(chunks)
            st.success(f"✅ Demo document indexed: {len(chunks)} chunks stored in Endee!")
    
    # Process uploaded files
    if uploaded_files:
        if st.button("🚀 Index Documents into Endee"):
            progress = st.progress(0)
            status = st.empty()
            
            for i, file in enumerate(uploaded_files):
                status.text(f"Processing: {file.name}...")
                
                if file.type == "application/pdf":
                    try:
                        import pdfplumber
                        with pdfplumber.open(file) as pdf:
                            text = "\n".join(p.extract_text() or "" for p in pdf.pages)
                    except ImportError:
                        text = f"[PDF content from {file.name} - install pdfplumber to parse PDFs]"
                else:
                    text = file.read().decode("utf-8", errors="ignore")
                
                chunks = process_document(text, file.name, chunk_size)
                
                for chunk in chunks:
                    embedding = get_embedding(chunk["text"])
                    st.session_state.endee_client.insert(chunk["id"], embedding, chunk)
                
                st.session_state.documents_indexed += 1
                st.session_state.chunks_indexed += len(chunks)
                progress.progress((i + 1) / len(uploaded_files))
            
            status.success(f"✅ {len(uploaded_files)} document(s) indexed with {st.session_state.chunks_indexed} total chunks!")

# ─── Tab 2: Ask Questions ─────────────────────────────────────────────────────
with tab2:
    st.markdown("### Ask Your Documents Anything")
    
    if st.session_state.chunks_indexed == 0:
        st.info("👆 Upload and index documents first, or load a demo document from the Upload tab.")
    else:
        st.success(f"✅ Ready! {st.session_state.chunks_indexed} chunks indexed in Endee.")
        
        # Display chat history
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])
        
        query = st.chat_input("Ask a question about your documents...")
        
        if query:
            st.session_state.chat_history.append({"role": "user", "content": query})
            
            with st.chat_message("user"):
                st.write(query)
            
            with st.chat_message("assistant"):
                with st.spinner("🔍 Searching Endee vector database..."):
                    # Embed the query
                    query_embedding = get_embedding(query)
                    
                    # Retrieve from Endee
                    results = st.session_state.endee_client.search(query_embedding, top_k=top_k)
                    
                    if not results:
                        answer = "I couldn't find relevant information in the indexed documents."
                        st.write(answer)
                    else:
                        # Show retrieved chunks
                        with st.expander(f"📎 Retrieved {len(results)} chunks from Endee"):
                            for r in results:
                                st.markdown(f"""
                                <div class="chunk-card">
                                    <b>📄 {r['metadata'].get('source', 'Unknown')} | Chunk {r['metadata'].get('chunk_id', '?')}</b>
                                    <br>Similarity Score: <code>{r['score']:.4f}</code>
                                    <p>{r['metadata'].get('text', '')[:300]}...</p>
                                </div>
                                """, unsafe_allow_html=True)
                        
                        # Check if Mistral key is set
                        if not os.environ.get("MISTRAL_API_KEY"):
                            context = "\n\n".join([r["metadata"]["text"] for r in results])
                            answer = f"**Retrieved Context (add Mistral API key for full answers):**\n\n{context[:800]}..."
                        else:
                            with st.spinner("🤖 Generating answer with Mistral..."):
                                context = "\n\n".join([r["metadata"]["text"] for r in results])
                                answer = generate_answer(query, context)
                        
                        st.markdown(f"""
                        <div class="answer-box">
                        {answer}
                        </div>
                        """, unsafe_allow_html=True)
                        
                        st.session_state.chat_history.append({"role": "assistant", "content": answer})

# ─── Tab 3: Interview Prep ────────────────────────────────────────────────────
with tab3:
    st.markdown("### 🎯 Resume-based Interview Question Generator")
    st.markdown("Paste your resume text and generate targeted interview questions.")
    
    resume_text = st.text_area(
        "Paste your resume here:",
        height=200,
        placeholder="Paste your resume content..."
    )
    
    interview_type = st.selectbox(
        "Interview Type:",
        ["Technical", "Behavioral", "Mixed", "HR Round"]
    )
    
    if st.button("🎯 Generate Interview Questions"):
        if not resume_text.strip():
            st.warning("Please paste your resume text.")
        elif not os.environ.get("MISTRAL_API_KEY"):
            st.warning("Please enter your Mistral API key in the sidebar.")
        else:
            # Index resume into Endee
            with st.spinner("Indexing resume into Endee..."):
                chunks = process_document(resume_text, "resume", 100)
                for chunk in chunks:
                    embedding = get_embedding(chunk["text"])
                    st.session_state.endee_client.insert(f"resume_{chunk['id']}", embedding, chunk)
                st.session_state.chunks_indexed += len(chunks)
            
            # Query for relevant skills
            query_embedding = get_embedding(f"skills experience projects technical expertise")
            results = st.session_state.endee_client.search(query_embedding, top_k=5)
            context = "\n".join([r["metadata"]["text"] for r in results])
            
            with st.spinner("Generating questions with Mistral..."):
                prompt_context = f"Resume content:\n{context}"
                questions = generate_answer(
                    f"Generate 8 {interview_type} interview questions based on this resume. Number them 1-8.",
                    prompt_context
                )
            
            st.markdown("#### Generated Interview Questions:")
            st.markdown(f"""
            <div class="answer-box">
            {questions}
            </div>
            """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:#888; font-size:0.8rem;'>Built with ❤️ using Endee Vector DB + Mistral + Streamlit</p>",
    unsafe_allow_html=True
)

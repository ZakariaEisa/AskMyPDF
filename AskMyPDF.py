import streamlit as st
from pypdf import PdfReader
import torch
import faiss
import numpy as np
from transformers import DPRContextEncoder, DPRContextEncoderTokenizer, DPRQuestionEncoder, DPRQuestionEncoderTokenizer
from transformers import AutoTokenizer, AutoModelForCausalLM

# ---------------------------
# 1. Load DPR models (cached for efficiency)
# ---------------------------
@st.cache_resource
def load_dpr_models():
    q_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained("facebook/dpr-question_encoder-single-nq-base")
    q_encoder = DPRQuestionEncoder.from_pretrained("facebook/dpr-question_encoder-single-nq-base")
    c_tokenizer = DPRContextEncoderTokenizer.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")
    c_encoder = DPRContextEncoder.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")
    return q_tokenizer, q_encoder, c_tokenizer, c_encoder

q_tokenizer, q_encoder, c_tokenizer, c_encoder = load_dpr_models()

# ---------------------------
# 2. Load LLM model (cached for efficiency)
# ---------------------------
@st.cache_resource
def load_llm_model():
    tokenizer = AutoTokenizer.from_pretrained("tiiuae/falcon-7b-instruct")
    model = AutoModelForCausalLM.from_pretrained("tiiuae/falcon-7b-instruct", torch_dtype=torch.float16)
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    return tokenizer, model

tokenizer, model = load_llm_model()

# ---------------------------
# 3. Helper functions
# ---------------------------
def embed_question(question):
    """Generate DPR embedding for a question"""
    inputs = q_tokenizer(question, return_tensors="pt")
    with torch.no_grad():
        embeddings = q_encoder(**inputs).pooler_output
    return embeddings.detach().numpy().astype("float32")

def embed_context(context):
    """Generate DPR embedding for a context paragraph/chunk"""
    inputs = c_tokenizer(context, return_tensors="pt")
    with torch.no_grad():
        embeddings = c_encoder(**inputs).pooler_output
    return embeddings.detach().numpy().astype("float32")



def read_pdf(file):
    """Read PDF file and return list of pages as text using pypdf."""
    reader = PdfReader(file)
    pages = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            pages.append(text.strip())
    return pages

def chunk_text(text, max_words=150):
    """
    Split long text into smaller chunks with at most max_words words.
    Improves retrieval accuracy in FAISS.
    """
    words = text.split()
    chunks = []
    for i in range(0, len(words), max_words):
        chunk = " ".join(words[i:i + max_words])
        chunks.append(chunk)
    return chunks

def prepare_chunks(pages, max_words=150):
    """Split all pages into smaller chunks"""
    all_chunks = []
    for page in pages:
        chunks = chunk_text(page, max_words)
        all_chunks.extend(chunks)
    return all_chunks

def generate_answer(question, chunks, k=5):
    """
    Retrieve top-k chunks relevant to the question using FAISS + DPR,
    then generate an answer using the LLM.
    """
    # Create embeddings for each chunk
    embeddings = np.vstack([embed_context(c) for c in chunks])
    index = faiss.IndexFlatIP(768)  # DPR embedding dimension
    index.add(embeddings)
    
    # Embed the question and search in FAISS
    q_embedding = embed_question(question)
    distances, indices = index.search(q_embedding, k)
    retrieved_chunks = [chunks[idx] for idx in indices[0]]
    
    # Combine retrieved context and question into prompt for LLM
    combined_context = "Context:\n" + "\n".join(retrieved_chunks) + f"\n\nQuestion: {question}\nAnswer:"
    inputs = tokenizer(combined_context, return_tensors="pt").to(model.device)
    
    # Generate answer
    outputs = model.generate(**inputs, max_new_tokens=200, do_sample=True, temperature=0.7)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    answer = answer.split("Answer:")[-1].strip()
    return answer

# ---------------------------
# 4. Streamlit UI
# ---------------------------
st.title("PDF RAG Question Answering")

# File uploader for PDF
uploaded_file = st.file_uploader("Upload your lecture PDF", type="pdf")

if uploaded_file is not None:
    st.success("PDF uploaded successfully!")
    
    # Read PDF and split pages into text chunks
    pages = read_pdf(uploaded_file)
    chunks = prepare_chunks(pages, max_words=150)
    
    # Input box for user question
    question = st.text_input("Ask a question about the PDF:")

    if st.button("Get Answer") and question:
        with st.spinner("Generating answer..."):
            answer = generate_answer(question, chunks)
        st.markdown("**Answer:**")
        st.write(answer)




import streamlit as st
from pypdf import PdfReader
import torch
import faiss
import numpy as np
from transformers import DPRContextEncoder, DPRContextEncoderTokenizer, DPRQuestionEncoder, DPRQuestionEncoderTokenizer
import requests
import json

# ---------------------------
# 1. Load DPR models (cached)
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
# 2. Helper functions
# ---------------------------
def embed_question(question):
    inputs = q_tokenizer(question, return_tensors="pt")
    with torch.no_grad():
        embeddings = q_encoder(**inputs).pooler_output
    return embeddings.detach().numpy().astype("float32")

def embed_context(context):
    inputs = c_tokenizer(context, return_tensors="pt")
    with torch.no_grad():
        embeddings = c_encoder(**inputs).pooler_output
    return embeddings.detach().numpy().astype("float32")

def read_pdf(file):
    reader = PdfReader(file)
    pages = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            pages.append(text.strip())
    return pages

def chunk_text(text, max_words=150):
    words = text.split()
    chunks = []
    for i in range(0, len(words), max_words):
        chunk = " ".join(words[i:i + max_words])
        chunks.append(chunk)
    return chunks

def prepare_chunks(pages, max_words=150):
    all_chunks = []
    for page in pages:
        chunks = chunk_text(page, max_words)
        all_chunks.extend(chunks)
    return all_chunks

# ---------------------------
# 3. Gemini API function
# ---------------------------
API_KEY = "AIzaSyDq-iicJs7jfR2yqqEiC2SgnY0oLcm7qA0"
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta2/models/gemini-2.0-flash:generateText"

def query_gemini(prompt, max_output_tokens=200, temperature=0.7):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}"
    }
    data = {
        "prompt": {"text": prompt},
        "temperature": temperature,
        "candidateCount": 1,
        "maxOutputTokens": max_output_tokens
    }
    response = requests.post(GEMINI_URL, headers=headers, json=data)
    if response.status_code == 200:
        result = response.json()
        return result["candidates"][0]["output"]
    else:
        return f"Error: {response.status_code} - {response.text}"

# ---------------------------
# 4. Generate answer using RAG + Gemini
# ---------------------------
def generate_answer(question, chunks, k=5):
    # Create embeddings for chunks
    embeddings = np.vstack([embed_context(c) for c in chunks])
    index = faiss.IndexFlatIP(768)
    index.add(embeddings)
    
    # Embed question and search top-k chunks
    q_embedding = embed_question(question)
    distances, indices = index.search(q_embedding, k)
    retrieved_chunks = [chunks[idx] for idx in indices[0]]
    
    # Combine context + question for Gemini
    combined_context = "Context:\n" + "\n".join(retrieved_chunks) + f"\n\nQuestion: {question}\nAnswer:"
    answer = query_gemini(combined_context)
    return answer

# ---------------------------
# 5. Streamlit UI
# ---------------------------
st.title("PDF RAG Question Answering with Gemini 2.0 Flash")

uploaded_file = st.file_uploader("Upload your lecture PDF", type="pdf")
if uploaded_file is not None:
    st.success("PDF uploaded successfully!")
    pages = read_pdf(uploaded_file)
    chunks = prepare_chunks(pages, max_words=150)

question = st.text_input("Ask a question about the PDF:")
if st.button("Get Answer") and question:
    with st.spinner("Generating answer..."):
        answer = generate_answer(question, chunks)
    st.markdown("**Answer:**")
    st.write(answer)


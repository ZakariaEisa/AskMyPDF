import streamlit as st
from pypdf import PdfReader
import torch
import faiss
import numpy as np
from transformers import (
    DPRContextEncoder, DPRContextEncoderTokenizer,
    DPRQuestionEncoder, DPRQuestionEncoderTokenizer,
    AutoTokenizer, AutoModelForSeq2SeqLM
)

# ---------------------------
# 1. Load DPR models (cached)
# ---------------------------
@st.cache_resource
def load_dpr_models():
    q_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained(
        "facebook/dpr-question_encoder-single-nq-base"
    )
    q_encoder = DPRQuestionEncoder.from_pretrained(
        "facebook/dpr-question_encoder-single-nq-base"
    )
    c_tokenizer = DPRContextEncoderTokenizer.from_pretrained(
        "facebook/dpr-ctx_encoder-single-nq-base"
    )
    c_encoder = DPRContextEncoder.from_pretrained(
        "facebook/dpr-ctx_encoder-single-nq-base"
    )
    return q_tokenizer, q_encoder, c_tokenizer, c_encoder

q_tokenizer, q_encoder, c_tokenizer, c_encoder = load_dpr_models()

# ---------------------------
# 2. Load FLAN-T5-Base model
# ---------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@st.cache_resource
def load_llm_model():
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
    model.to(device)
    return tokenizer, model

tokenizer, model = load_llm_model()

# ---------------------------
# 3. Helper functions
# ---------------------------
def embed_question(question):
    inputs = q_tokenizer(question, return_tensors="pt").to(device)
    with torch.no_grad():
        embeddings = q_encoder(**inputs).pooler_output
    embeddings = embeddings / embeddings.norm(dim=1, keepdim=True)
    return embeddings.cpu().numpy().astype("float32")

def embed_context(context):
    inputs = c_tokenizer(context, return_tensors="pt").to(device)
    with torch.no_grad():
        embeddings = c_encoder(**inputs).pooler_output
    embeddings = embeddings / embeddings.norm(dim=1, keepdim=True)
    return embeddings.cpu().numpy().astype("float32")

def read_pdf(file):
    reader = PdfReader(file)
    pages = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            pages.append(text.strip())
    return pages

def chunk_text(text, max_words=250):  
    words = text.split()
    return [" ".join(words[i:i + max_words]) for i in range(0, len(words), max_words)]

def prepare_chunks(pages, max_words=250):
    all_chunks = []
    for page in pages:
        all_chunks.extend(chunk_text(page, max_words))
    return all_chunks

@st.cache_data
def embed_chunks(chunks):
    return np.vstack([embed_context(c) for c in chunks])

@st.cache_resource
def build_faiss_index(embeddings):
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return index

def generate_answer(question, chunks, index, k=5, similarity_threshold=0.2):
    q_embedding = embed_question(question)
    distances, indices = index.search(q_embedding, k)
    
    # فلترة حسب التشابه
    retrieved_chunks = [
        chunks[idx] for idx, dist in zip(indices[0], distances[0]) if dist >= similarity_threshold
    ]
    
    if not retrieved_chunks:
        return "عذراً، لا يوجد معلومات كافية للإجابة عن هذا السؤال."

    # تحسين الـ prompt
    prompt = (
        "Answer the following question in detail based on the context below:\n\n"
        "Context:\n"
        + "\n".join(retrieved_chunks)
        + f"\n\nQuestion: {question}\nAnswer in a detailed manner:"
    )
    
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=500,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        top_k=50
    )
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer

# ---------------------------
# 4. Streamlit UI
# ---------------------------
st.title("PDF RAG Question Answering")

st.image(
    "https://i.pinimg.com/736x/b5/69/f0/b569f0f987b17f314bcd64a6019c4641.jpg",
    use_container_width=True
)

uploaded_file = st.file_uploader("Upload your PDF", type="pdf")
if uploaded_file is not None:
    st.success("PDF uploaded successfully!")
    pages = read_pdf(uploaded_file)
    
    if not pages:
        st.error("Try another PDF.")
    else:
        chunks = prepare_chunks(pages, max_words=250)
        embeddings = embed_chunks(chunks)
        index = build_faiss_index(embeddings)

        question = st.text_input("Ask a question about the PDF:")
        if st.button("Get Answer") and question:
            with st.spinner("Generating answer..."):
                answer = generate_answer(question, chunks, index)
            st.markdown("**Answer:**")
            st.write(answer)













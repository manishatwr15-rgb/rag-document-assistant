import streamlit as st
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import faiss
import numpy as np

# -------------------------
# Load models (run once)
# -------------------------
@st.cache_resource
def load_models():
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")

    model_name = "google/flan-t5-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    return embed_model, tokenizer, model

embed_model, tokenizer, model = load_models()

# -------------------------
# Load and index document
# -------------------------
@st.cache_resource
def create_index():
    with open("mydoc.txt", "r", encoding="utf-8") as f:
        text = f.read()

    chunks = text.split("\n\n")
    embeddings = embed_model.encode(chunks)

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings))

    return index, chunks

index, chunks = create_index()

# -------------------------
# Streamlit UI
# -------------------------
st.title("📄 Local RAG Chatbot")

query = st.text_input("Ask a question about your document:")

if query:
    query_embedding = embed_model.encode([query])
    distances, indices = index.search(np.array(query_embedding), k=2)

    retrieved_text = "\n".join([chunks[i] for i in indices[0]])

    prompt = f"""
Answer the question based only on the context below.

Context:
{retrieved_text}

Question:
{query}

Answer:
"""

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
    output = model.generate(**inputs, max_new_tokens=150)
    answer = tokenizer.decode(output[0], skip_special_tokens=True)

    st.write("### 🤖 Answer")
    st.write(answer)


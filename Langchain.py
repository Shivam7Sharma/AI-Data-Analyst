import os
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from pdfminer.high_level import extract_text
import faiss
import numpy as np

# Function to extract text from PDF


def pdf_to_text(pdf_path):
    return extract_text(pdf_path)

# Function to split text into chunks


def split_text(text, chunk_size=1000):
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]


# Embedding function using Sentence Transformers
model = SentenceTransformer('all-MiniLM-L6-v2')


def embed_text(text_chunks):
    return model.encode(text_chunks)


# Initialize FAISS index
dimension = 384  # Dimension of embeddings
# Use IndexFlatIP for inner product (cosine similarity)
index = faiss.IndexFlatL2(dimension)

# Indexing function for FAISS


def index_chunks(chunks):
    embeddings = embed_text(chunks)
    embeddings = np.array(embeddings).astype('float32')
    index.add(embeddings)  # Adding embeddings to FAISS index

# Semantic search in FAISS to find relevant chunks


def query_faiss(query, top_k=3):
    query_embedding = embed_text([query])[0]
    query_embedding = np.array(query_embedding).astype(
        'float32').reshape(1, -1)
    distances, indices = index.search(query_embedding, top_k)
    return [chunks[idx] for idx in indices[0]]


# Question answering
qa_pipeline = pipeline("question-answering",
                       model="deepset/roberta-base-squad2")


def get_answers(query, context_chunks):
    return [qa_pipeline(question=query, context=chunk) for chunk in context_chunks]


# Main execution
if __name__ == "__main__":
    # Load and prepare your document (replace 'path/to/document.pdf' with your document's path)
    # If already text, skip this
    document_text = pdf_to_text(
        '/home/shivam/Downloads/schoenberger2016sfm.pdf')
    chunks = split_text(document_text)

    # Index the document chunks in FAISS
    index_chunks(chunks)

    # Example query
    query = "What comes after initializing reconstruction?"

    # Find relevant chunks
    relevant_chunks = query_faiss(query, top_k=5)  # Adjust top_k as needed

    # Get answers based on relevant chunks
    answers = get_answers(query, relevant_chunks)
    for answer in answers:
        print(f"Answer: {answer['answer']}, Score: {answer['score']}")

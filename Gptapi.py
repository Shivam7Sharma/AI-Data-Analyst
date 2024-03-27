import os
from sentence_transformers import SentenceTransformer
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
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
index = faiss.IndexFlatL2(dimension)  # Use IndexFlatIP for cosine similarity

# Indexing function for FAISS


def index_chunks(chunks, index=index):
    embeddings = embed_text(chunks)
    embeddings = np.array(embeddings).astype('float32')
    index.add(embeddings)  # Adding embeddings to FAISS index
    return index
# Semantic search in FAISS to find relevant chunks


def query_faiss(index, chunks, query, top_k=3):
    query_embedding = embed_text([query])[0]
    query_embedding = np.array(query_embedding).astype(
        'float32').reshape(1, -1)
    distances, indices = index.search(query_embedding, top_k)
    return [chunks[idx] for idx in indices[0]]


# Initialize GPT-2 for Question Answering
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model_gpt2 = GPT2LMHeadModel.from_pretrained("gpt2")

# Function for question answering with GPT-2


# Function for question answering with GPT-2, using max_new_tokens
def get_answers_gpt2(query, context_chunks):
    answers = []
    for chunk in context_chunks:
        # Prepare the input text, ensure it does not exceed max input size
        input_text = f"Context: {chunk} Question: {query} Answer:"
        input_ids = tokenizer.encode(input_text, return_tensors='pt')

        # Ensure the input is within the model's capacity
        max_model_input_size = model_gpt2.config.n_positions
        if input_ids.shape[1] > max_model_input_size:
            input_ids = input_ids[:, :max_model_input_size]

        # Generate an answer, specifying max_new_tokens
        output = model_gpt2.generate(
            input_ids,
            max_new_tokens=50,  # Generate up to 50 new tokens beyond the input
            num_return_sequences=1
        )
        answer = tokenizer.decode(output[0], skip_special_tokens=True)
        answers.append(answer[len(input_text):])  # Extract the generated text
    return answers


# Main execution
if __name__ == "__main__":
    document_text = pdf_to_text('/home/shivam/Downloads/Resume_Shivam_S.pdf')
    chunks = split_text(document_text)
    index_chunks(chunks)
    query = "where did shivam sharma study for his masters?"
    relevant_chunks = query_faiss(chunks, query, top_k=5)
    answers = get_answers_gpt2(query, relevant_chunks)
    for answer in answers:
        print(f"Answer: {answer}")

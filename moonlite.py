# impport necassary libarries

# define a fuction to handle CLI input

    # set up arguemnt parser

    # add argumnet for query input

    # parse the arugment s

# define a fuction to load vecotr database

    # check if databsae exists

    # if not, creat it

    # return databse connction

# define a fuction to process querries

    # take user inpt

    #convrt input into vectoor embedding

    # serch for closest matcing results in db

    # retrun results

# define main fuction

    # check if user proivded a querry

    # if so, process it

    # else , show hlp msg

# excute main fuction if script is run direcly

# i am building a cli rag system which means this program will take a query from the user, search for relevant information and return useful results
    # since everything in the program depends on what the user asks for, handling user input seems to be a good place to start
    # i will structure the input first creating an entry point for the system, if we dont handle input correctly, nothing else can function properly. 
# user inputs query, 

# import argparse
# import faiss
# import numpy as np
# from sentence_transformers import SentenceTransformer

# # Load embedding model
# model = SentenceTransformer("all-MiniLM-L6-v2")›

# Initialize FAISS vector store
# dim = 384  # Vector size for MiniLM
# index = faiss.IndexFlatL2(dim)

# def process_input():
#     """Handles CLI input"""
#     parser = argparse.ArgumentParser(description="Moonlite Systems CLI")
#     parser.add_argument("--query", "-q", type=str, help="Enter query")
#     args = parser.parse_args()

#     cleaned_input = args.query.strip().lower() if args.query else None

#     if cleaned_input == "exit":
#         print("Exiting...")
#         return None

#     return cleaned_input

# def embed_text(text):
#     """Convert text into a vector embedding."""
#     return model.encode([text])[0]

# def add_to_db(text):
#     """Convert text to vector and add it to the FAISS database."""
#     vector = np.array([embed_text(text)], dtype="float32")
#     index.add(vector)
#     print(f"Added to database: {text}")

# def search_db(query, top_k=3):
#     """Search the FAISS database for similar results."""
#     query_vector = np.array([embed_text(query)], dtype="float32")
#     distances, indices = index.search(query_vector, top_k)
#     return indices, distances

# if __name__ == "__main__":
#     user_query = process_input()
#     if user_query:
#         if user_query.lower() == "search":
#             query_text = input("Enter search query: ")
#             results = search_db(query_text)
#             print("Search Results:", results)
#         else:
#             add_to_db(user_query)  # ✅ Ensure this function exists

import argparse
import faiss
import numpy as np
import os
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer

# Initialize embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Initialize FAISS vector store
dim = 384  # Vector size for MiniLM
index = faiss.IndexFlatL2(dim)

# Store text data alongside embeddings
text_data = []  # List to hold original text

# File paths for saving FAISS index and text data
FAISS_INDEX_PATH = "faiss_index.bin"
TEXT_DATA_PATH = "text_data.txt"

def save_data():
    """Save FAISS index and text data to disk."""
    faiss.write_index(index, FAISS_INDEX_PATH)
    with open(TEXT_DATA_PATH, "w") as f:
        f.write("\n".join(text_data))
    print("✅ FAISS index and text data saved.")

def load_data():
    """Load FAISS index and text data from disk if available."""
    global index, text_data
    if os.path.exists(FAISS_INDEX_PATH):
        index = faiss.read_index(FAISS_INDEX_PATH)
        print("✅ FAISS index loaded.")
    else:
        print("⚠️ No FAISS index found. Starting fresh.")

    if os.path.exists(TEXT_DATA_PATH):
        with open(TEXT_DATA_PATH, "r") as f:
            text_data = f.read().splitlines()
        print("✅ Text data loaded.")
    else:
        print("⚠️ No text data found. Starting fresh.")

def embed_text(text):
    """Convert text into a vector embedding."""
    return model.encode([text])[0]

def add_to_db(text):
    """Convert text to vector and add it to FAISS."""
    if text in text_data:
        print(f"⚠️ Duplicate entry detected: \"{text}\". Skipping...")
        return
    
    vector = np.array([embed_text(text)], dtype="float32")
    index.add(vector)
    text_data.append(text)  # Store text separately
    save_data()  # Save after adding
    print(f"✅ Added to database: {text}")

def search_db(query, top_k=3):
    """Search FAISS for the most similar results and return unique text matches with normalized scores."""
    query_vector = np.array([embed_text(query)], dtype="float32").reshape(1, -1)
    distances, indices = index.search(query_vector, top_k)

    # Use a dictionary to store unique results
    unique_results = {}

    # Normalize scores (convert FAISS distance into similarity score)
    min_dist, max_dist = min(distances[0]), max(distances[0]) if distances[0].size > 0 else (0, 1)

    # Handle case where min_dist == max_dist (prevents all scores from being 1.0 or 0.0)
    if max_dist - min_dist < 1e-9:
        min_dist -= 1  # Shift min slightly so normalization works

    for i, dist in zip(indices[0], distances[0]):
        if i < len(text_data):
            text_entry = text_data[i]
            normalized_score = 1 - ((dist - min_dist) / (max_dist - min_dist + 1e-9))  # Avoid division by zero
            normalized_score = max(0.01, normalized_score)  # Ensure no score is exactly 0.0000
            if text_entry not in unique_results:  # Prevent duplicate entries
                unique_results[text_entry] = normalized_score

    # Sort by highest similarity score
    sorted_results = sorted(unique_results.items(), key=lambda x: x[1], reverse=True)
    final_texts, final_scores = zip(*sorted_results) if sorted_results else ([], [])

    return final_texts, final_scores

def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file."""
    doc = fitz.open(pdf_path)
    text = "\n".join([page.get_text("text") for page in doc])
    return text.strip()

import re

def chunk_text(text, max_chunk_size=500):
    """Splits text into paragraph-based chunks while preserving readability."""
    paragraphs = text.split("\n")  # Split on newlines
    chunks = []
    current_chunk = []
    current_length = 0

    print(f"🔍 Debug: Extracted {len(paragraphs)} paragraphs from PDF")  # Debug print

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue  # Skip empty lines

        # Check if adding this paragraph exceeds chunk size
        if current_length + len(para) > max_chunk_size:
            chunks.append(" ".join(current_chunk))  # Store completed chunk
            current_chunk = [para]  # Start new chunk
            current_length = len(para)
        else:
            current_chunk.append(para)
            current_length += len(para)

    if current_chunk:
        chunks.append(" ".join(current_chunk))  # Store last chunk

    print(f"🔍 Debug: Created {len(chunks)} chunks from text")  # Debug print
    return chunks

def process_pdf(pdf_path):
    """Extract and add PDF text chunks to FAISS."""
    text = extract_text_from_pdf(pdf_path)
    chunks = chunk_text(text)
    print(f"📄 Extracted {len(chunks)} chunks from {pdf_path}")
    
    for chunk in chunks:
        add_to_db(chunk)

def process_input():
    """Handles CLI input."""
    parser = argparse.ArgumentParser(description="Moonlite Systems CLI")
    parser.add_argument("--query", "-q", type=str, help="Enter query")
    parser.add_argument("--add", "-a", type=str, help="Add text to FAISS")
    parser.add_argument("--pdf", "-p", type=str, help="Add PDF text to FAISS")  # Ensure this line is here
    
    return parser.parse_args()

if __name__ == "__main__":
    load_data()  # Load FAISS index and text at startup

    args = process_input()

    if args.add:
        add_to_db(args.add)
    elif args.pdf:
        process_pdf(args.pdf)
    elif args.query:
        results, distances = search_db(args.query)
        print(f"🔎 Search results:")
        for i, (result, dist) in enumerate(zip(results, distances)):
            print(f"{i+1}. {result} (Score: {dist:.4f})")
    else:
        print("Please provide either --add, --query, or --pdf.")

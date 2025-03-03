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
    """Convert text to vector and add it to FAISS, ensuring no duplicates."""
    if text in text_data:
        print(f"⚠️ Duplicate entry detected: \"{text}\". Skipping...")
        return
    
    vector = np.array([embed_text(text)], dtype="float32")
    index.add(vector)
    text_data.append(text)  # Store text separately
    save_data()  # Save after adding
    print(f"✅ Added to database: {text}")


def search_db(query, top_k=3):
    """Search FAISS for the most similar results and return unique text matches."""
    query_vector = np.array([embed_text(query)], dtype="float32").reshape(1, -1)

    print(f"🔍 Query vector shape: {query_vector.shape}")  # Debugging print

    distances, indices = index.search(query_vector, top_k)

    # Use a set to store unique results
    unique_results = {}
    
    for i, dist in zip(indices[0], distances[0]):
        if i < len(text_data) and dist < 1e+10:  # Ignore invalid FAISS results
            text_entry = text_data[i]
            if text_entry not in unique_results:  # Only add unique entries
                unique_results[text_entry] = dist

    # Convert back to list for proper output formatting
    sorted_results = sorted(unique_results.items(), key=lambda x: x[1])  # Sort by distance
    final_texts, final_distances = zip(*sorted_results) if sorted_results else ([], [])

    return final_texts, final_distances


def process_input():
    """Handles CLI input."""
    parser = argparse.ArgumentParser(description="Moonlite Systems CLI")
    parser.add_argument("--query", "-q", type=str, help="Enter query")
    parser.add_argument("--add", "-a", type=str, help="Add text to FAISS")
    
    return parser.parse_args()

if __name__ == "__main__":
    load_data()  # Load FAISS index and text at startup

    args = process_input()

    if args.add:
        add_to_db(args.add)
    elif args.query:
        results, distances = search_db(args.query)
        print(f"🔎 Search results:")
        for i, (result, dist) in enumerate(zip(results, distances)):
            print(f"{i+1}. {result} (Score: {dist:.4f})")
    else:
        print("Please provide either --add or --query.")

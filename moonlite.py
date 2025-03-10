import argparse
import faiss
import numpy as np
import os
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer

class MoonliteSystem:
    def __init__(self):
        # Initialize embedding model
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

        # Initialize FAISS vector store
        self.dim = 384  # Vector size for MiniLM
        self.index = faiss.IndexFlatL2(self.dim)

        # Store text data alongside embeddings
        self.text_data = []  # List to hold original text

        # File paths for saving FAISS index and text data
        self.FAISS_INDEX_PATH = "faiss_index.bin"
        self.TEXT_DATA_PATH = "text_data.txt"
        
        # Load existing data (if any)
        self.load_data()

    def save_data(self):
        """Save FAISS index and text data to disk."""
        faiss.write_index(self.index, self.FAISS_INDEX_PATH)
        with open(self.TEXT_DATA_PATH, "w") as f:
            f.write("\n".join(self.text_data))
        print("✅ FAISS index and text data saved.")

    def load_data(self):
        """Load FAISS index and text data from disk if available."""
        if os.path.exists(self.FAISS_INDEX_PATH):
            self.index = faiss.read_index(self.FAISS_INDEX_PATH)
            print("✅ FAISS index loaded.")
        else:
            print("⚠️ No FAISS index found. Starting fresh.")

        if os.path.exists(self.TEXT_DATA_PATH):
            with open(self.TEXT_DATA_PATH, "r") as f:
                self.text_data = f.read().splitlines()
            print("✅ Text data loaded.")
        else:
            print("⚠️ No text data found. Starting fresh.")

    def embed_text(self, text):
        """Convert text into a vector embedding."""
        return self.model.encode([text])[0]

    def add_to_db(self, text):
        """Convert text to vector and add it to FAISS."""
        if text in self.text_data:
            print(f"⚠️ Duplicate entry detected: \"{text}\". Skipping...")
            return
        
        vector = np.array([self.embed_text(text)], dtype="float32")
        self.index.add(vector)
        self.text_data.append(text)  # Store text separately
        self.save_data()  # Save after adding
        print(f"✅ Added to database: {text}")

    def search_db(self, query, top_k=3):
        """Search FAISS for the most similar results and return unique text matches with normalized scores."""
        query_vector = np.array([self.embed_text(query)], dtype="float32").reshape(1, -1)
        distances, indices = self.index.search(query_vector, top_k)

        # Use a dictionary to store unique results
        unique_results = {}

        # Normalize scores (convert FAISS distance into similarity score)
        min_dist, max_dist = min(distances[0]), max(distances[0]) if distances[0].size > 0 else (0, 1)

        # Handle case where min_dist == max_dist (prevents all scores from being 1.0 or 0.0)
        if max_dist - min_dist < 1e-9:
            min_dist -= 1  # Shift min slightly so normalization works

        for i, dist in zip(indices[0], distances[0]):
            if i < len(self.text_data):
                text_entry = self.text_data[i]
                normalized_score = 1 - ((dist - min_dist) / (max_dist - min_dist + 1e-9))  # Avoid division by zero
                normalized_score = max(0.01, normalized_score)  # Ensure no score is exactly 0.0000
                if text_entry not in unique_results:  # Prevent duplicate entries
                    unique_results[text_entry] = normalized_score

        # Sort by highest similarity score
        sorted_results = sorted(unique_results.items(), key=lambda x: x[1], reverse=True)
        final_texts, final_scores = zip(*sorted_results) if sorted_results else ([], [])

        return final_texts, final_scores

    def extract_text_from_pdf(self, pdf_path):
        """Extract text from a PDF file."""
        doc = fitz.open(pdf_path)
        text = "\n".join([page.get_text("text") for page in doc])
        return text.strip()

    def chunk_text(self, text, max_chunk_size=500):
        """Splits text into paragraph-based chunks while preserving readability."""
        paragraphs = text.split("\n")  # Split on newlines
        chunks = []
        current_chunk = []
        current_length = 0

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
        return chunks

    def process_pdf(self, pdf_path):
        """Extract and add PDF text chunks to FAISS."""
        text = self.extract_text_from_pdf(pdf_path)
        chunks = self.chunk_text(text)
        for chunk in chunks:
            self.add_to_db(chunk)

    def process_input(self):
        """Handles CLI input."""
        parser = argparse.ArgumentParser(description="Moonlite Systems CLI")
        parser.add_argument("--query", "-q", type=str, help="Enter query")
        parser.add_argument("--add", "-a", type=str, help="Add text to FAISS")
        parser.add_argument("--pdf", "-p", type=str, help="Add PDF text to FAISS")  # Ensure this line is here

        return parser.parse_args()

if __name__ == "__main__":
    system = MoonliteSystem()  # Create an instance of the class

    args = system.process_input()

    if args.add:
        system.add_to_db(args.add)
    elif args.pdf:
        system.process_pdf(args.pdf)
    elif args.query:
        results, distances = system.search_db(args.query)
        for i, (result, dist) in enumerate(zip(results, distances)):
            print(f"{i+1}. {result} (Score: {dist:.4f})")
    else:
        print("Please provide either --add, --query, or --pdf.")

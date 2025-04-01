# moonlite-systems: Retrieval-Augmented Generation Knowledgebase

**moonlite-systems** is a lightweight, modular RAG (retrieval-augmented generation) system designed to answer questions using a custom knowledge base. It combines embedding-based document search with OpenAI’s generative models to return contextually grounded, accurate responses.

Built as part of Moonlite’s broader mission to create tools for introspection, alignment, and clarity, this system reflects a hybrid mindset: design-forward, developer-aware, and focused on signal over noise.

---

## 🧠 What It Does

- Accepts user questions as input  
- Searches a local knowledge index for relevant context  
- Uses OpenAI’s API to generate answers grounded in that context  
- Returns clear, conversational responses  

This system is optimized for **personal and project-based knowledge organization**, designed to scale with future decks, documents, and Moonlite's internal creative systems.

---

## ⚙️ Stack & Tools

- `LangChain` – orchestrates RAG pipeline  
- `FAISS` – vector similarity search for fast document retrieval  
- `OpenAI API` – GPT model for natural language responses  
- `NumPy`, `dotenv`, and `tiktoken` – environment config, encoding, and support tooling  
- Input documents: PDFs, MDs, plaintext (Moonlite format standard in development)

---

## 🧪 How It Works

1. **Ingest & Chunk**: Parses PDFs or text files, splits into chunks for embedding  
2. **Embed & Index**: Generates vector embeddings and stores them in a FAISS index  
3. **Query**: User submits a question  
4. **Retrieve**: Relevant chunks retrieved based on vector similarity  
5. **Generate**: GPT model produces an answer based on retrieved context

---

## 📁 Folder Structure


# Sanskrit RAG System


This repository implements a **CPU-only Retrieval-Augmented Generation (RAG) pipeline** for Sanskrit documents, as required in the ImmverseAI AI/ML intern assignment. It ingests Sanskrit stories, builds a vector index, retrieves relevant passages for a query, and generates answers using a lightweight multilingual sequence-to-sequence model on CPU. 

## Project structure

- `code/`
  - `rag_sanskrit/config.py` – paths, chunking parameters, model names and generation settings.
  - `rag_sanskrit/ingest.py` – document loader and simple character-level chunker.
  - `rag_sanskrit/vectorstore.py` – sentence-transformer embeddings + FAISS index for retrieval.
  - `rag_sanskrit/generator.py` – CPU-only multilingual generator (`google/byt5-small`) wrapped in a simple interface.
  - `rag_sanskrit/pipeline.py` – end-to-end console RAG loop (build/load index, query, retrieve, generate).
- `data/` – Sanskrit `.txt` files (converted from the assignment’s Sanskrit DOCX stories). 
- `report/` – technical report (PDF) describing architecture, preprocessing, retrieval, generation, and performance.
- `requirements.txt` – Python dependencies.
- `main.py` – entry point that calls the RAG pipeline.
- `README.md` – this file. 

## Requirements

- Python 3.10+ (tested with 3.12).
- CPU-only environment (no GPU is required or used).
- Internet access on first run to download Hugging Face models:
  - `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` for embeddings.
  - `google/byt5-small` for multilingual generation. 

Install dependencies:
pip install -r requirements.txt

Optional (for web UI):
pip install streamlit

## Data preparation

1. All Sanskrit source stories from the assignment (e.g., *मूर्खभृत्यस्य*, *चतुरस्य कालीदासस्य*, *वृद्धायाः चार्तुयम्*) are converted from DOCX to UTF‑8 `.txt` files. 
2. Place these `.txt` files inside the `data/` directory.
3. On first run, the script loads all `.txt` files, concatenates their contents, and splits them into overlapping chunks controlled by `CHUNK_SIZE` and `CHUNK_OVERLAP` in `config.py`. 

## Running the console RAG app

From the project root:
python main.py


The application will:

1. Build a vector index over the Sanskrit corpus (first run may take longer because it computes embeddings and downloads models). 
2. Enter an interactive loop:

   - When prompted with:

     ```
     Enter query (or 'exit'):
     ```

   - Type a question in **Sanskrit** or **English / transliterated text** related to the stories, for example:

     - `कः अस्मिन् कथायां मुख्यः भृत्यः अस्ति ?`
     - `What is the moral of the story about the foolish servant?`

3. For each query, the system:

   - Retrieves the top‑K most similar chunks using FAISS.
   - Builds a prompt that injects these Sanskrit passages.
   - Calls the `google/byt5-small` model on CPU to generate an answer. 

The console prints:

- `===== ANSWER =====` – model-generated answer (may be noisy for complex Sanskrit).
- A **top retrieved context snippet**, so the user can verify that retrieval picked the correct passage even if generation is imperfect.

Exit by typing `exit` or `quit`.

## Running the web UI (optional)

A simple Streamlit web interface is provided in `app.py`.

From the project root, run:
streamlit run app.py

The browser UI allows you to:

- Optionally upload additional UTF‑8 `.txt` Sanskrit files (these are saved into the `data/` folder).
- Ask questions in Sanskrit or English via a text box.
- See the generated answer and the top retrieved Sanskrit context passage.

PDF documents can be used by first converting them to plain UTF‑8 `.txt` files and then placing or uploading those text files into the `data/` folder.



## Design choices and limitations (high level)

- **Retriever**: `paraphrase-multilingual-MiniLM-L12-v2` is used because it supports many languages (including Indic scripts) and is lightweight enough for CPU semantic search over small corpora. 
- **Generator**: `google/byt5-small` is a byte-level multilingual model that runs on CPU without GPU, keeping the assignment within resource constraints. For these specialised Sanskrit passages, answer quality is limited; the model sometimes produces partially garbled text. This trade-off is documented in the report, and the generator module is kept modular so a stronger Sanskrit‑specific model can be plugged in later. 
- **CPU-only**: All models are explicitly moved to CPU (`model.to("cpu")`) and there is no GPU-specific code.
- **Index persistence**: FAISS index and chunk list are stored in `data/index/` so they can be reused across runs (if enabled in the pipeline), which reduces CPU cost for repeated experiments. 

For more details on architecture, preprocessing, retrieval, generation, and performance metrics (latency and memory usage on a CPU-only machine), see the report in `report/`. 


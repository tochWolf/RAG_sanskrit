import sys
import os

# Add the 'code' folder to Python path
sys.path.append(os.path.abspath("code"))

from pathlib import Path
from typing import Optional

import streamlit as st

# Make sure we can import from the "code" folder
PROJECT_ROOT = Path(__file__).resolve().parent
CODE_DIR = PROJECT_ROOT / "code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from rag_sanskrit.pipeline import build_or_load_index, answer_query  # type: ignore
from rag_sanskrit.generator import SanskritGenerator  # type: ignore


@st.cache_resource(show_spinner=True)
def get_vectorstore_and_generator():
    """
    Build the index and load the generator model once per session.
    """
    st.write("Building index and loading models (first time may take a while)...")
    vs, num_chunks = build_or_load_index()
    gen = SanskritGenerator()
    return vs, gen, num_chunks


def main():
    st.title("Sanskrit Document RAG Demo (CPU Only)")
    st.write(
        "Upload Sanskrit `.txt` documents, then ask questions in Sanskrit or English. "
        "The system retrieves relevant passages and generates an answer using a CPU-only multilingual model."
    )

    # ---------------------------
    # Step 1: optional .txt upload
    # ---------------------------
    st.subheader("Step 1: (Optional) Upload additional Sanskrit .txt files")
    uploaded_files = st.file_uploader(
        "You can upload one or more UTF-8 `.txt` files to add them to the corpus for this session.",
        type=["txt"],
        accept_multiple_files=True,
    )

    if uploaded_files:
        data_dir = PROJECT_ROOT / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        for f in uploaded_files:
            target_path = data_dir / f.name
            content = f.read().decode("utf-8", errors="ignore")
            target_path.write_text(content, encoding="utf-8")
        st.success(f"Saved {len(uploaded_files)} file(s) into the data/ folder. The index will include them.")

    # ---------------------------
    # Step 2: build index + model
    # ---------------------------
    vs: Optional[object]
    gen: Optional[object]
    num_chunks: int

    with st.spinner("Building index and loading models (first time may take a while)..."):
        vs, gen, num_chunks = get_vectorstore_and_generator()

    st.success(f"Index ready with {num_chunks} chunks.")

    # ---------------------------
    # Step 3: query interface
    # ---------------------------
    st.subheader("Step 2: Ask a question")
    query = st.text_area(
        "Enter your question (Sanskrit or English/transliteration):",
        height=80,
    )

    if st.button("Get answer") and query.strip():
        with st.spinner("Retrieving and generating answer..."):
            answer, contexts = answer_query(vs, gen, query.strip())

        st.markdown("### Answer")
        st.write(answer)

        if contexts:
            st.markdown("### Top retrieved context")
            st.write(contexts[0][:800])


if __name__ == "__main__":
    main()

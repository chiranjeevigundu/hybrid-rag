# Mini RAG (TF‑IDF Retriever)

Tiny retrieval‑augmented "answerer" using TF‑IDF over local `.txt` files. No API keys, runs locally.

## Quickstart
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python src/index_build.py     # builds index over ./data/*.txt
streamlit run src/app.py
```

## How it works
- Uses `TfidfVectorizer` to index local docs
- Retrieves top‑k passages
- Generates a concise answer by extracting and compressing the best snippet

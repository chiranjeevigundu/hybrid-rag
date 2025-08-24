from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

DATA_DIR = Path(__file__).parent.parent / "data"
OUT = Path(__file__).parent / "tfidf.joblib"

def load_docs():
    docs = []
    for p in DATA_DIR.glob("*.txt"):
        docs.append((p.name, p.read_text()))
    return docs

def main():
    docs = load_docs()
    texts = [d[1] for d in docs]
    names = [d[0] for d in docs]
    vec = TfidfVectorizer(stop_words="english")
    X = vec.fit_transform(texts)
    joblib.dump({"vectorizer": vec, "matrix": X, "names": names, "texts": texts}, OUT)
    print(f"Indexed {len(texts)} docs into {OUT}")

if __name__ == "__main__":
    main()

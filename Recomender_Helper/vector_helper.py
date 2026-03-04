import json
import os
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Union
import tqdm

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

BOOK_VECTORS = os.path.join(
    BASE_DIR, "data", "combined_books.jsonl"
)

def _load_book_data(file_path):
    book_map = {}
    print("loading vec data")
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                record = json.loads(line)
                isbn = record.get("isbn")
                if isbn:
                    # Store the whole record (or just the vectors) keyed by ISBN
                    book_map[isbn] = record
    except FileNotFoundError:
        print(f"Warning: {file_path} not found.")
    return book_map

# Global constant loaded when the script starts
BOOK_DATA_CACHE = _load_book_data(BOOK_VECTORS)

def get_vector_by_isbn(isbn: str, vector_type: str):

    valid_types = {"emotion_intensity", "emotion", "empath", "tf_idf", "word2vec", "doc2vec"}

    if vector_type not in valid_types:
        raise ValueError(f"vector_type must be one of {valid_types}")

    record = BOOK_DATA_CACHE.get(isbn)
    if record:
        return record.get(vector_type)

    return None

def graphTF_IDF(results, title):
    plt.figure()
    plt.bar(range(len(results)), results)
    plt.xlabel("Term Index")
    plt.ylabel("TF-IDF Weight")
    plt.title(title)
    plt.show()

def graphDictVector(results, title):
    r = dict(results)
        
    plt.barh(range(len(r)), list(r.values()), align='center')
    plt.yticks(range(len(r)), list(r.keys()))

    #plt.xlabel('Emotion')
    plt.title(title)
    plt.show()

def graphVector(results, title):
    if isinstance(results, list):
        graphTF_IDF(results, title)
    else:
        graphDictVector(results, title)

def concat(vec1, vec2):
    if isinstance(vec1, dict):
        vec1 = list(vec1.values())
    if isinstance(vec2, dict):
        vec2 = list(vec2.values())
    return vec1 + vec2

def cosine_similarity(vec1, vec2):
    if type(vec1) is not type(vec2):
        raise ValueError("Both vectors must be the same type.")

    # Case 1: List vectors (e.g., tf_idf)
    if isinstance(vec1, list):
        if len(vec1) != len(vec2):
            raise ValueError("Both list vectors must have same length.")

        v1 = np.array(vec1, dtype=float)
        v2 = np.array(vec2, dtype=float)

    # Case 2: Dict vectors (e.g., emotion/empath)
    elif isinstance(vec1, dict):
        if set(vec1.keys()) != set(vec2.keys()):
            raise ValueError("Both dict vectors must have same keys.")

        # Ensure consistent ordering
        keys = sorted(vec1.keys())
        v1 = np.array([vec1[k] for k in keys], dtype=float)
        v2 = np.array([vec2[k] for k in keys], dtype=float)

    else:
        raise TypeError("Unsupported vector type.")

    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return float(np.dot(v1, v2) / (norm1 * norm2))


def average_vectors(vectors: List[Union[List[float], dict]]):
    """
    Averages a list of vectors.
    
    Works for:
        - List[float]  (e.g., tf_idf)
        - Dict[str, float] (e.g., emotion_intensity, emotion, empath)

    Returns:
        Averaged vector (same structure as input)
    """

    if not vectors:
        raise ValueError("Vector list is empty.")

    first = vectors[0]

    # Case 1: List-based vector (e.g., tf_idf)
    if isinstance(first, list):
        return np.mean(np.array(vectors), axis=0).tolist()

    # Case 2: Dict-based vector (emotion/empath)
    elif isinstance(first, dict):
        keys = first.keys()

        # Safety check
        if not all(v.keys() == keys for v in vectors):
            raise ValueError("All dict vectors must have same keys.")

        return {
            key: sum(v[key] for v in vectors) / len(vectors)
            for key in keys
        }

    else:
        raise TypeError("Unsupported vector type.")

if __name__ == "__main__":
    isbn_1 = "0876054122"
    isbn_2 = "0135658217"
    isbn_3 = "0764114638"
    isbn_4 = "1569717451"
    vec_1 = get_vector_by_isbn(isbn_1, "emotion_intensity")
    vec_2 = get_vector_by_isbn(isbn_1, "empath")
    concat = concat(vec_1, vec_2)
    print(concat)

import json
#from Vectorizer.EmotionVectorMaker import EmotionVectorMaker
from Vectorizer.Word2VecVectorMaker import Word2VecVectorMaker
from Vectorizer.Doc2VecVectorMaker import Doc2VecVectorMaker
import os
from tqdm import tqdm

# Paths
JSONL_PATH = "data/books_with_subjects_complete.jsonl"
YOUTH_ISBN_PATH = "data/books_read_by_youth.txt"

def get_descriptions():
    with open(YOUTH_ISBN_PATH, "r", encoding="utf-8") as f:
        youth_isbns = {line.strip() for line in f if line.strip()}

    descriptions = []

    with open(JSONL_PATH, "r", encoding="utf-8") as f:
        for line in f:
            book = json.loads(line)
            isbn = book.get("ISBN")

            if isbn in youth_isbns:
                description = book.get("description")
                if description:  # avoid None or empty
                    descriptions.append((isbn, description))
    
    return descriptions

book_descriptions = get_descriptions()

# emo_intensity_vec_maker = EmotionVectorMaker()

# emo_vec_maker = EmotionVectorMaker(use_intensity=False)


all_descriptions = [text for _, text in book_descriptions]

word2vec_vec_maker = Word2VecVectorMaker(all_descriptions)
doc2vec_vec_maker = Doc2VecVectorMaker(all_descriptions)

print("Vector Makers all made")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
output_path = os.path.join(
    BASE_DIR, "data", "book_vectors.jsonl"
)

with open(output_path, "w", encoding="utf-8") as outfile:
    i = 0
    for isbn, description in tqdm(book_descriptions):
        # emo_intensity_vec = emo_intensity_vec_maker.getEmotionVector(description, removeObj=True)
        # emo_vec = emo_vec_maker.getEmotionVector(description, removeObj=True)
        word2vec_vec = word2vec_vec_maker.getVector(description)
        doc2vec_vec = doc2vec_vec_maker.getVector(description)
        book = {
            "isbn" : isbn,
            # "emotion_intensity" : emo_intensity_vec,
            # "emotion" : emo_vec,
            "word2vec": word2vec_vec,
            "doc2vec": doc2vec_vec
        }
        outfile.write(json.dumps(book) + "\n")

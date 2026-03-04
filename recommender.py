import json
import os
from tqdm import tqdm
from collections import Counter
from Recomender_Helper.vector_helper import get_vector_by_isbn, cosine_similarity, average_vectors, concat

# ---- Paths ----

TEST_DATA_FILE = "data/users_with_one_STEM_book_and_six_plus_high_rated_books_formatted.json"
#TEST_DATA_FILE = "user_eval_sets/test.json"
STEM_BOOKS_FILE = "data/stem_books.jsonl"


def handle_book(isbn, emotion_type, topic_type):
    emotion_vec = get_vector_by_isbn(isbn, emotion_type)
    if emotion_vec is None:
        raise Exception(f"Do not have vector for {isbn}")
    topic_vec = get_vector_by_isbn(isbn, topic_type)
    combined = concat(emotion_vec, topic_vec)

    return combined

def general_stem_topic_vec_maker(topic_type):
    stem_isbns = set()
    print("Retreiving General STEM isbns")
    with open(STEM_BOOKS_FILE, 'r') as file:
        for line in tqdm(file):
            book = json.loads(line)
            stem_isbns.add(book["ISBN"])
    stem_vecs = []
    print("Retrieving STEM vectors")
    for isbn in tqdm(stem_isbns):
        temp = get_vector_by_isbn(isbn, topic_type)
        if temp:
            stem_vecs.append(temp)
    return average_vectors(stem_vecs)

def make_candidate_profile(profile_books, emotion_type, general_stem_topic_vec):
    emotion_vectors = []
    for book in profile_books:
        isbn = book['isbn']
        try:
            emotion_vectors.append(get_vector_by_isbn(isbn, emotion_type))
        except Exception as e:
            print(e)
    if len(emotion_vectors) == 0:
        raise Exception(f"Missing all profile books")
    
    emotion_profile = average_vectors(emotion_vectors)
    return concat(emotion_profile, general_stem_topic_vec)



def recomend(test_data_file, emotion_type = "emotion_intensity", topic_type = "doc2vec"):
    general_stem_topic_vec = general_stem_topic_vec_maker(topic_type)
    with open(test_data_file, 'r') as file:
        data = json.load(file)
    
    for item in tqdm(data):
        profile_books = item['candidate_profile']
        try:
            candidate_profile = make_candidate_profile(profile_books, emotion_type, general_stem_topic_vec)
            recomendation_books = item['recommendation_list']
            for book in recomendation_books:
                try:
                    book_vec = handle_book(book['isbn'], emotion_type, topic_type)
                    cos = cosine_similarity(candidate_profile, book_vec)
                    book['cos'] = cos
                except Exception as e:
                    print(e)
        except Exception as e:
            print(e)
            print(item['user_id'])
    
    output_file_name = f"recommendations/recommendation_using_{emotion_type}_{topic_type}.json"
    with open(output_file_name, 'w') as f:
        json.dump(data, f, indent=4)


recomend(TEST_DATA_FILE)
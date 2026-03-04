import json
from tqdm import tqdm

file_vectors = "data/book_vectors.jsonl"      # word2vec + doc2vec file
file_features = "book_vectors.jsonl"    # emotion + tfidf file
output_file = "combined_books.jsonl"

# ---- Load vector file ----
vectors = {}

with open(file_vectors, "r") as f:
    for line in tqdm(f, desc="Loading vector file"):
        item = json.loads(line)
        vectors[item["isbn"]] = item

# ---- Merge with second file ----
with open(file_features, "r") as f_in, open(output_file, "w") as f_out:
    for line in tqdm(f_in, desc="Merging files"):
        item = json.loads(line)
        isbn = item["isbn"]

        if isbn not in vectors:
            print(f"Missing vector for {isbn}")
            continue

        merged = {**item, **vectors[isbn]}

        json.dump(merged, f_out)
        f_out.write("\n")

print("Finished merging files.")
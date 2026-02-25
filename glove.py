import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import string

# "glove.6B.100d.txt"
# "wiki_giga_2024_100_MFT20_vectors_seed_2024_alpha_0.75_eta_0.05.050_combined.txt"
# "wiki_giga_2024_200_MFT20_vectors_seed_2024_alpha_0.75_eta_0.05_combined.txt"
# "glove.42B.300d.txt"
# "wiki_giga_2024_300_MFT20_vectors_seed_2024_alpha_0.75_eta_0.05_combined.txt"


class GloVeReviewer():
    def __init__(self, glove_path="wiki_giga_2024_300_MFT20_vectors_seed_2024_alpha_0.75_eta_0.05_combined.txt", expected_dim=300):
        print("Loading GloVe embeddings...")
        self.embeddings = self.load_glove(glove_path, expected_dim)
        self.dim = len(next(iter(self.embeddings.values())))

    def load_glove(self, file_path, expected_dim):
        """Load pretrained word embeddings safely (handles stray punctuation lines).""" 
        embeddings = {}
        bad_lines = 0

        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line_num, line in enumerate(f):
                parts = line.strip().split()
                if len(parts) < expected_dim + 1:
                    bad_lines += 1
                    continue

                word = parts[0]
                if all(ch in string.punctuation for ch in word):
                    continue

                try:
                    vec = np.array(parts[1:], dtype=float)
                    # Only keep vectors of the expected size
                    if vec.shape[0] != expected_dim:
                        bad_lines += 1
                        continue
                    embeddings[word] = vec
                except ValueError:
                    bad_lines += 1
                    continue

        print(f"Loaded {len(embeddings):,} embeddings. Skipped {bad_lines:,} malformed lines.")
        return embeddings

    def tokenize(self, text):
        return re.findall(r"\b\w+\b", text.lower())

    def text_to_vector(self, text):
        words = self.tokenize(text)
        vectors = [self.embeddings[w] for w in words if w in self.embeddings]
        if len(vectors) == 0:
            return np.zeros(self.dim)
        return np.mean(vectors, axis=0)

    def trained_review(self, book_list):
        descriptions = [b[1] for b in book_list]
        labels = np.array([b[2] for b in book_list])  # 1 = STEM, 0 = non-STEM

        # --- Convert text to averaged GloVe embeddings ---
        X = np.array([self.text_to_vector(desc) for desc in descriptions])

        # --- Split into train/test ---
        X_train, X_test, y_train, y_test = train_test_split(
            X, labels, test_size=0.2, random_state=42, stratify=labels
        )

        # --- Train classifier ---
        clf = LogisticRegression(max_iter=1000)
        clf.fit(X_train, y_train)

        # --- Evaluate ---
        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"\nClassification Accuracy: {acc:.3f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=["Non-STEM", "STEM"]))
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
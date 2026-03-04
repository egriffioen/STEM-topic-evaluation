from gensim.models import Word2Vec, KeyedVectors
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np
import re

class Word2VecReviewer():
    def __init__(self, word2vec_path=None):
        if word2vec_path:
            print("Loading pretrained Word2Vec model...")
            self.w2v = KeyedVectors.load_word2vec_format(word2vec_path, binary=True)
        else:
            self.w2v = None  # will train locally if not provided

    def tokenize(self, text):
        return re.findall(r"\b\w+\b", text.lower())

    def text_to_vector(self, text):
        words = self.tokenize(text)
        vectors = [
            self.w2v[word] for word in words if word in self.w2v
        ]
        if len(vectors) == 0:
            return np.zeros(self.w2v.vector_size)
        return np.mean(vectors, axis=0)

    def trained_review(self, book_list, use_word_list=False):
        descriptions = [b[1] for b in book_list]
        labels = np.array([b[2] for b in book_list])  # 1 = STEM, 0 = non-STEM

        # --- Train or load Word2Vec model ---
        tokenized_texts = [self.tokenize(desc) for desc in descriptions]
        if self.w2v is None:
            print("Training Word2Vec model from data...")
            self.w2v = Word2Vec(
                sentences=tokenized_texts,
                vector_size=100,
                window=5,
                min_count=2,
                workers=4
            ).wv

        # --- Convert each description to an averaged vector ---
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
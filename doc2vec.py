from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np
import re

class Doc2VecReviewer():
    def __init__(self):
        self.doc2vec = None

    def tokenize(self, text):
        return re.findall(r"\b\w+\b", text.lower())

    def trained_review(self, book_list):
        descriptions = [b[1] for b in book_list]
        labels = np.array([b[2] for b in book_list])  # 1 = STEM, 0 = non-STEM

        # --- Prepare tagged documents for Doc2Vec ---
        tagged_docs = [TaggedDocument(words=self.tokenize(desc), tags=[str(i)]) 
                       for i, desc in enumerate(descriptions)]

        # --- Train Doc2Vec model ---
        print("Training Doc2Vec model...")
        self.doc2vec = Doc2Vec(
            documents=tagged_docs,
            vector_size=100,
            window=5,
            min_count=2,
            workers=4,
            epochs=20,
            dm=1  # 1 = Distributed Memory (better for semantics)
        )

        # --- Get vectors for each document ---
        X = np.array([self.doc2vec.dv[str(i)] for i in range(len(descriptions))])

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
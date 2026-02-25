import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.decomposition import PCA
import numpy as np
from book_list_getter import WordListGetter

class TF_IDF_Reviewer():
    def __init__(self):
        self.word_list_getter = WordListGetter()
        pass

    def save_trained(self, book_list):
        descriptions = [b[1] for b in book_list]
        labels = np.array([b[2] for b in book_list]) 
        vectorizer = TfidfVectorizer(
            stop_words='english',
            max_features=5000,
            ngram_range=(1, 2)
        )

        X = vectorizer.fit_transform(descriptions)

        X_train, X_test, y_train, y_test = train_test_split(
            X, labels, test_size=0.2, random_state=42, stratify=labels
        )

        # --- Train a simple classifier ---
        clf = LogisticRegression(max_iter=1000)
        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"\nClassification Accuracy: {acc:.3f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=["Non-STEM", "STEM"]))
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, y_pred))

        self.vectorizer = vectorizer
        self.clf = clf

    def test_single_book(self, description):
    # Ensure the model has been trained
        if not hasattr(self, "vectorizer") or not hasattr(self, "clf"):
            raise ValueError("Model not trained. Run trained_review() first.")
        
        # Transform the new description using the trained vectorizer
        X_new = self.vectorizer.transform([description])
        
        # Predict label and probabilities
        probs = self.clf.predict_proba(X_new)[0]
        pred = np.argmax(probs)
        
        # Convert numeric label to string
        label = "STEM" if pred == 1 else "Non-STEM"
        prob = probs[pred]
        
        return label, prob, X_new


    def trained_review(self, book_list, use_word_list=False):
        descriptions = [b[1] for b in book_list]
        labels = np.array([b[2] for b in book_list])  # 1 = STEM, 0 = non-STEM

        # --- TF-IDF vectorization ---
        if use_word_list: 
            stem_words = self.word_list_getter.empath_word_list()
            vectorizer = TfidfVectorizer(
                stop_words='english',
                max_features=5000,
                vocabulary=stem_words,
                ngram_range=(1, 2)
            )
        else:
            vectorizer = TfidfVectorizer(
                stop_words='english',
                max_features=5000,
                ngram_range=(1, 2)
            )

        X = vectorizer.fit_transform(descriptions)

        # --- Split into train/test ---
        X_train, X_test, y_train, y_test = train_test_split(
            X, labels, test_size=0.2, random_state=42, stratify=labels
        )

        # --- Train a simple classifier ---
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

        # --- PCA Visualization ---
        pca = PCA(n_components=2, random_state=42)
        X_reduced = pca.fit_transform(X.toarray())

        plt.figure(figsize=(8, 6))
        colors = ['blue' if s else 'red' for s in labels]
        plt.scatter(X_reduced[:, 0], X_reduced[:, 1],
                    c=colors, s=100, alpha=0.7, edgecolor='k')

        plt.xlabel('PCA Component 1')
        plt.ylabel('PCA Component 2')
        plt.title(f'Book Descriptions (TF-IDF Projection) — Accuracy: {acc:.2f}')
        plt.legend(handles=[
            plt.Line2D([], [], marker='o', color='w', markerfacecolor='blue', label='STEM', markersize=8),
            plt.Line2D([], [], marker='o', color='w', markerfacecolor='red', label='Non-STEM', markersize=8)
        ])
        plt.show()

    def base_review(self, book_list, use_word_list = False):
        descriptions = [b[1] for b in book_list]
        labels = np.array([b[2] for b in book_list])

        if use_word_list: 
            stem_words = self.word_list_getter.empath_word_list()
            vectorizer = TfidfVectorizer(
                stop_words='english',
                max_features=5000,
                vocabulary=stem_words  # comment this line out to use all words
            )

        else:
            vectorizer = TfidfVectorizer(
                stop_words='english',
                max_features=5000,
            )

        X = vectorizer.fit_transform(descriptions)

        pca = PCA(n_components=2, random_state=42)
        X_reduced = pca.fit_transform(X.toarray())

        plt.figure(figsize=(8, 6))
        colors = ['blue' if s else 'red' for s in labels]

        plt.scatter(X_reduced[:, 0], X_reduced[:, 1],
                    c=colors, s=100, alpha=0.7, edgecolor='k')

        plt.xlabel('PCA Component 1')
        plt.ylabel('PCA Component 2')
        plt.title('Book Descriptions (TF-IDF Projection)')
        plt.legend(handles=[
            plt.Line2D([], [], marker='o', color='w', markerfacecolor='blue', label='STEM', markersize=8),
            plt.Line2D([], [], marker='o', color='w', markerfacecolor='red', label='Non-STEM', markersize=8)
        ])
        plt.show()

    def training_review(self, book_list):
        descriptions = [b[1] for b in book_list]
        y = np.array([b[2] for b in book_list])

        vectorizer = TfidfVectorizer(
            max_features=5000,      # limit vocabulary size
            stop_words='english',    # remove common words
            ngram_range=(1,2)        # optionally include bigrams
        )

        X = vectorizer.fit_transform(descriptions)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        clf = LogisticRegression(max_iter=1000)
        clf.fit(X_train, y_train)

        feature_names = vectorizer.get_feature_names_out()
        coefs = clf.coef_[0]

        top_stem_words = sorted(zip(feature_names, coefs), key=lambda x: x[1], reverse=True)[:20]
        print(top_stem_words)

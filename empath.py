from empath import Empath
import itertools


class EmpathReviewer():
    def __init__(self):
        self.lexicon = Empath()

    def category_check(self):
        all_keys = []
        for i in range(30):
            lex = Empath()
            cats_keys = set(vars(lex)['cats'].keys())
            all_keys.append(cats_keys)
        
        first = all_keys[0]
        for i, keys in enumerate(all_keys[1:], start=2):
            if keys != first:
                print(f"⚠️ Difference found between run 1 and run {i}")
                print("Only in run 1:", first - keys)
                print("Only in run", i, ":", keys - first)
                break
        else:
            print("✅ All 30 Empath() instances have identical category keys.")

        # Optional: print number of categories and a sample
        print(f"Total categories: {len(first)}")
        print("Example categories:", list(sorted(first))[:10])



    def make_combinations(self, items):
        all_combinations = []
        for r in range(1, len(items) + 1):
            combinations_r = itertools.combinations(items, r)
            all_combinations.extend(combinations_r)

        # Convert each combination from tuple to list (optional)
        all_combinations = [list(comb) for comb in all_combinations]

        return all_combinations

    def test_different_category_combinations(self, book_list):
        all_cat = self.make_combinations(["science", "technology", "engineering", "math", "mathematics"])

        for cat in all_cat:
            TP = 0
            FP = 0
            TN = 0
            FN = 0
            print(cat)

            for book in book_list:  
                _, summary, is_stem = book

                book_eval = self.lexicon.analyze(summary, cat)
                score = sum(book_eval.values())

                if is_stem:
                    if score > 0: TP += 1
                    else: FN += 1
                else:
                    if score > 0: FP +=1
                    else: TN += 1

            precision = TP / (TP + FP) if (TP + FP) > 0 else 0
            recall = TP / (TP + FN) if (TP + FN) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

            print(f"TP: {TP}, FP: {FP}, FN: {FN}, TN: {TN}")
            print(f"Precision: {precision:.3f}, Recall: {recall:.3f}, F1 Score: {f1:.3f}")
            print()

    def output_test(self, book_list, base_cat = ["science", "technology", "engineering", "math", "mathematics"]):
        for book in book_list:
            _, summary, is_stem = book
            book_eval = self.lexicon.analyze(summary, base_cat)
            print(f"ground truth: {is_stem}, eval: {book_eval}")

    def math_test(self, base_cat = ["science", "technology", "engineering", "math", "mathematics"]):
        test_1 = "Glencoe Algebra 1 is a key program in our vertically aligned high school mathematics series developed to help all students achieve a better understanding of mathematics and improve their mathematics scores on today s high-stakes assessments."
        book_eval = self.lexicon.analyze(test_1, base_cat)
        print(book_eval)

    def var_explorer(self):
        print(vars(self.lexicon)['cats'].keys())

    def base_test(self, book_list, base_cat = ["science", "technology", "engineering", "math", "mathematics"]):
        TP = 0
        FP = 0
        TN = 0
        FN = 0

        for book in book_list:  
            _, summary, is_stem = book

            book_eval = self.lexicon.analyze(summary, base_cat)
            score = sum(book_eval.values())

            if is_stem:
                if score > 0: TP += 1
                else: FN += 1
            else:
                if score > 0: FP +=1
                else: TN += 1

        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        print(f"TP: {TP}, FP: {FP}, FN: {FN}, TN: {TN}")
        print(f"Precision: {precision:.3f}, Recall: {recall:.3f}, F1 Score: {f1:.3f}")


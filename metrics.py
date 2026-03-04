import json
import numpy as np
import math
from sklearn.metrics import ndcg_score
import scipy.stats as stats

def get_rr(y_true, threshold=7):
    x = 1
    for y in y_true:
        if y >= threshold:
            return 1.0/x
        x+=1
    return 0.0


def precision_at_k(ranked_ratings, k, threshold=7):
    top_k = ranked_ratings[:k]
    if len(top_k) == 0:
        return 0.0
    relevant = sum(1 for r in top_k if r >= threshold)
    return relevant / k

def boost_stem_ratings(user_obj, boost_amount):
    recommendations = user_obj.get("recommendation_list", [])

    for item in recommendations:
        if item.get("is_stem") is True:
            item["rating"] += boost_amount

    return user_obj

def handle_user(record):
    recommendations = record["recommendation_list"]

    filtered = [rec for rec in recommendations if rec["rating"] != 0]
    # The filtering should be redundent but just in case 

    y_score = np.array([[rec["cos"] for rec in filtered]])
    y_true = np.array([[rec["rating"] for rec in filtered]])

    ndcg = ndcg_score(y_true, y_score)
    ndcg_5 = ndcg_score(y_true, y_score, k=5)
    ndcg_10 = ndcg_score(y_true, y_score, k=10)

    recommendations_sorted = sorted(filtered, key=lambda r: r["cos"], reverse=True)
    ranked_ratings = [rec["rating"] for rec in recommendations_sorted]
    rr = get_rr(ranked_ratings)

    scores = [rec["cos"] for rec in filtered]
    ratings = [rec["rating"] for rec in filtered]
    rho, p_value = stats.spearmanr(scores, ratings)
    if math.isnan(rho):
        print(record["user"])

    # Precision@K
    p1 = precision_at_k(ranked_ratings, k=1)
    p3 = precision_at_k(ranked_ratings, k=3)
    p5 = precision_at_k(ranked_ratings, k=5)

    return ndcg, ndcg_5, ndcg_10, rho, p_value, rr, p1, p3, p5



file_name = "recommendation_using_emotion_intensity_doc2vec.json"
file_path = f"recommendations/{file_name}"

with open(file_path, "r", encoding="utf-8") as f:
    records = json.load(f)

ndcg_list = []
ndcg_5_list = []
ndcg_10_list = []
rho_list = []
rr_list = []
p1_list = []
p3_list = []
p5_list = []


for record in records:
    update_record = boost_stem_ratings(record, boost_amount=2)

    ndcg, ndcg_5, ndcg_10, rho, p_value, rr, p1, p3, p5 = handle_user(record)
    ndcg_list.append(ndcg)
    ndcg_5_list.append(ndcg_5)
    ndcg_10_list.append(ndcg_10)
    if rho is not None and not math.isnan(rho):
        rho_list.append(rho)
    rr_list.append(rr)
    p1_list.append(p1)
    p3_list.append(p3)
    p5_list.append(p5)

overall_ndcg = np.mean(ndcg_list)
overall_ndcg_5 = np.mean(ndcg_5_list)
overall_ndcg_10 = np.mean(ndcg_10_list)
overall_rho = np.mean(rho_list)
mrr = np.mean(rr_list)
overall_p1 = np.mean(p1_list)
overall_p3 = np.mean(p3_list)
overall_p5 = np.mean(p5_list)


print(file_name)
print("Overall NDCG:", overall_ndcg)
print("Overall NDCG@5:", overall_ndcg_5)
print("Overall NDCG@10:", overall_ndcg_10)
print("Overall Spearman rho:", overall_rho)
print("MRR:", mrr)
print("P@1:", overall_p1)
print("P@3:", overall_p3)
print("P@5:", overall_p5)

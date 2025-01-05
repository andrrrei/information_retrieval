import numpy as np

ideal_ranking = [
    [1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
    [2, 1, 1, 0, 0, 0, 0, 0, 0, 0],
    [2, 1, 0, 0, 0, 0, 0, 0, 0, 0],
]

scores = {
    "TF": [
        [1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 2, 2, 1, 0, 0, 0, 0, 0],
        [2, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ],
    "TF-IDF": [
        [1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 2, 2, 1, 0, 0, 0, 0, 0, 0],
        [2, 0, 1, 0, 0, 0, 0, 0, 0, 0],
    ],
    "LM (λ=0.5)": [
        [0, 0, 0, 0, 0, 0, 0, 1, 1, 0],
        [0, 2, 1, 2, 0, 0, 0, 0, 0, 0],
        [2, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    ],
    "LM (λ=0.9)": [
        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        [2, 2, 1, 0, 0, 0, 0, 0, 0, 0],
        [2, 0, 1, 0, 0, 0, 0, 0, 0, 0],
    ],
    "FastText": [
        [0, 0, 2, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 2, 2, 0, 0, 0, 0, 0, 0],
        [2, 0, 1, 0, 0, 0, 0, 0, 0, 0],
    ],
    "FastText+IDF": [
        [0, 2, 0, 0, 0, 0, 0, 0, 0, 0],
        [2, 1, 2, 0, 0, 0, 0, 0, 0, 0],
        [2, 0, 1, 0, 0, 0, 0, 0, 0, 0],
    ],
}


def dcg(scores, k=10):
    scores = scores[:k]
    return sum(score / np.log2(idx + 2) for idx, score in enumerate(scores))


def ndcg_fixed(ideal_ranking, scores, k=10):
    ndcg_values = {}
    detailed_results = {}

    ideal_dcg_per_query = [
        dcg(sorted(query, reverse=True), k) for query in ideal_ranking
    ]

    for method, method_scores in scores.items():
        ndcg_per_query = []
        for query_idx, query_scores in enumerate(method_scores):
            ideal_dcg = ideal_dcg_per_query[query_idx]
            if ideal_dcg == 0:
                ndcg_per_query.append(0)
            else:
                ndcg_per_query.append(dcg(query_scores, k) / ideal_dcg)
        ndcg_values[method] = np.mean(ndcg_per_query)
        detailed_results[method] = ndcg_per_query

    return ndcg_values, detailed_results


ndcg_results_fixed, detailed_results_fixed = ndcg_fixed(ideal_ranking, scores)

print(ndcg_results_fixed, detailed_results_fixed)

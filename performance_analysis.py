import numpy as np
from sklearn.metrics import average_precision_score

def compute_cmc_rankn(similarity_matrix, query_ids, gallery_ids, max_rank=10):
    """Computes CMC Curve & Rank-N Accuracy"""
    num_queries = len(query_ids)
    cmc_curve = np.zeros(max_rank)

    for i in range(num_queries):
        # Sort gallery images by similarity score
        sorted_indices = np.argsort(similarity_matrix[i])[::-1]

        # Get ranked list of predicted IDs
        ranked_vehicle_ids = gallery_ids[sorted_indices]

        # Find rank of the first correct match
        correct_match_ranks = np.where(ranked_vehicle_ids == query_ids[i])[0]

        if len(correct_match_ranks) > 0:
            first_correct_rank = correct_match_ranks[0]
            cmc_curve[first_correct_rank:] += 1  # Increment all ranks ≥ first match

    cmc_curve /= num_queries  # Normalize

    # Rank-N Metrics
    rank_1 = cmc_curve[0] * 100
    rank_5 = cmc_curve[4] * 100 if max_rank >= 5 else None
    rank_10 = cmc_curve[9] * 100 if max_rank >= 10 else None

    return cmc_curve, rank_1, rank_5, rank_10

def compute_map(similarity_matrix, query_ids, gallery_ids):
    """Computes mean Average Precision (mAP)"""
    num_queries = len(query_ids)
    average_precisions = []

    for i in range(num_queries):
        # Sort gallery images by similarity score
        sorted_indices = np.argsort(similarity_matrix[i])[::-1]

        # Get binary relevance vector (1 if correct, 0 otherwise)
        relevance = (gallery_ids[sorted_indices] == query_ids[i]).astype(int)

        # Compute Average Precision (AP)
        if relevance.sum() > 0:
            ap = average_precision_score(relevance, similarity_matrix[i, sorted_indices])
            average_precisions.append(ap)

    return np.mean(average_precisions)
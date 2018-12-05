# Retrieve: Takes context and resources. Uses Word Mover's Distance to obtain
# relevant resource candidates.

import numpy as np


def retrieve(context, resources, gensim_model):
    """
    Calculates the similarity of each resource and the context. The similarity
    is calculated by averaging the embeddings and then calculating the cosine
    similarity.
    """

    # avg_context = np.mean(context, axis=0)
    # similarities = []

    distances = []

    for resource in resources:
        distance = gensim_model.wmdistance(resource, context)
        distances.append(-distance)
        # avg_resource = np.mean(resource, axis=0)
        # similarity = cosine_similarity(avg_context, avg_resource)
        # similarities.append(similarity)

    # ranked_resources = rank_resources(resources, similarities)
    print(distances)
    return distances
    # return similarities



# def cosine_similarity(a, b):
#     """
#     Takes 2 vectors a, b and returns the cosine similarity according
#     to the definition of the dot product. If b only consists of UNK embeddings,
#     the similarity is 0.
#     """
#
#     if not np.any(b):
#         return 0
#
#     dot_product = np.dot(a, b)
#     norm_a = np.linalg.norm(a)
#     norm_b = np.linalg.norm(b)
#     return dot_product / (norm_a * norm_b)


# def rank_resources(resources, similarities):
#     """
#     Sorts the similarites in decreasing order and sorts the resources according
#     to this order.
#     """
#
#     sorted_indices = np.argsort(similarities)[::-1]
#     sorted_resources = [resources[i] for i in sorted_indices]
#     return sorted_resources

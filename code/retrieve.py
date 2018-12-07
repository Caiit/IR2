# Retrieve: Takes context and resources. Uses Word Mover's Distance to obtain
# relevant resource candidates.

import numpy as np


def retrieve(context, resources, gensim_model):
    """
    Calculates the similarity of each resource and the context. The similarity
    is calculated by averaging the embeddings and then calculating the cosine
    similarity.
    """

    similarities = []

    for resource in resources:
        distance = gensim_model.wmdistance(resource, context)
        similarities.append(-distance)

    return similarities

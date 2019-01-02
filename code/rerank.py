import numpy as np


def rerank(resources, class_indices, similarities, class_prediction):
    """
    Reranks the resources according to the class prediction and the similarity
    that was measured between each resource and the context.
    """

    scores = []

    for i in range(len(similarities)):
        score = similarities[i]* (1 / class_prediction[0][class_indices[i]])
        scores.append(score)

    sorted_indices = np.argsort(scores)[::-1]
    sorted_resources = [resources[i] for i in sorted_indices]
    sorted_classes = [class_indices[i] for i in sorted_indices]
    return sorted_resources, sorted_classes

# Implementation from https://github.com/stephenhky/PyWMD


from itertools import product
from collections import defaultdict
import numpy as np
from scipy.spatial.distance import euclidean
import pulp


singleindexing = lambda m, i, j: m*i+j
unpackindexing = lambda m, k: (k/m, k % m)


def tokens_to_fracdict(tokens):
    """
    Normalized count of how often each token occurs in the document.
    """

    count = defaultdict(lambda : 0)
    for token in tokens:
        count[token] += 1
    total_count = sum(count.values())
    return {token: float(count)/total_count for token, count in count.items()}


# use PuLP
def word_mover_distance_probspec(tokens1, tokens2, wvmodel, lpFile=None):
    """
    Calculate Word Mover's Distance between two sentences.
    """

    all_tokens = list(set(tokens1+tokens2))
    wordvecs = {token: wvmodel[token] for token in all_tokens}

    tokens1_nbow = tokens_to_fracdict(tokens1)
    tokens2_nbow = tokens_to_fracdict(tokens2)

    T = pulp.LpVariable.dicts("T_matrix", list(product(all_tokens, all_tokens)),
                              lowBound=0)

    prob = pulp.LpProblem("WMD", sense=pulp.LpMinimize)
    prob += pulp.lpSum([T[token1, token2]*euclidean(wordvecs[token1],
                        wordvecs[token2]) for token1, token2 in
                        product(all_tokens, all_tokens)])
    for token2 in tokens2_nbow:
        prob += pulp.lpSum([T[token1, token2] for token1 in
                            tokens1_nbow])==tokens2_nbow[token2]
    for token1 in tokens1_nbow:
        prob += pulp.lpSum([T[token1, token2] for token2 in
                            tokens2_nbow])==tokens1_nbow[token1]

    if lpFile!=None:
        prob.writeLP(lpFile)

    prob.solve()

    return prob


def word_mover_distance(tokens1, tokens2, wvmodel, lpFile=None):
    """
    Calculate Word Mover's Distance.
    """

    prob = word_mover_distance_probspec(tokens1, tokens2, wvmodel,
                                        lpFile=lpFile)
    return pulp.value(prob.objective)

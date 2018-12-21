# Retrieve: Takes context and resources. Uses Word Mover's Distance to obtain
# relevant resource candidates.

import argparse
from gensim.models import Word2Vec, KeyedVectors
import numpy as np
from tqdm import tqdm

import data_utils


def retrieve(context, resources, gensim_model):
    """
    Calculates the similarity of each resource and the context. The similarity
    is calculated by averaging the embeddings and then calculating the cosine
    similarity.
    """

    similarities = []

    for resource in resources:
        distance = gensim_model.wmdistance(resource, context)
        similarities.append(distance)
    return similarities


def compute_recall(data, word2vec):
    k = 3
    recalls = []
    for example in tqdm(data):
        resources = []
        embedded_resources = []
        class_indices = []

        data_utils.get_resources(example["documents"]["comments"], resources,
                                 embedded_resources)
        num_comments = len(resources)
        data_utils.get_resources(example["documents"]["fact_table"], resources,
                                 embedded_resources)
        num_facts = len(resources) - num_comments
        data_utils.get_resources(example["documents"]["plot"], resources,
                                 embedded_resources)
        num_plots = len(resources) - num_comments - num_facts
        data_utils.get_resources(example["documents"]["review"], resources,
                                 embedded_resources)
        num_reviews = len(resources) - num_comments - num_facts - num_plots

        # Keep track of where each resource originated from.
        class_indices += [2]*num_comments
        class_indices += [3]*num_facts
        class_indices += [0]*num_plots
        class_indices += [1]*num_reviews

        chat = example["chat"]
        labels = example["labels"]

        # Loop over each of the last three utterances in the chat (the context).
        for i in range(3, len(chat)-1):
            # Don't take the None into account, since it's for speaker 1
            if labels[i + 1] < 4:
                last_utterances = chat[i-3:i]
                response = chat[i+1]
                if len(response) > 0:
                    embedded_utterances = [data_utils.embed_sentence(utterance) for
                                           utterance in last_utterances]
                    context, embedded_context = data_utils.get_context(last_utterances)

                    # Retrieve: Takes context and resources. Uses Word Mover's Distance
                    # to obtain relevant resource candidates.
                    similarities = retrieve(context, resources, word2vec)

                    recalls.append(int(labels[i+1] in [class_indices[s] for s in sorted(range(len(similarities)), key=lambda i: similarities[i])[:k]]))

    print(np.mean(recalls))



def main(args):
    word2vec = KeyedVectors.load(args.word2vec, mmap='r')
    data = data_utils.load_data(args.file)

    if args.use_gensim:
        data_utils.load_gensim(args.word2vec)
    else:
        data_utils.load_embeddings(args.embeddings)
        data_utils.load_w2i(args.w2i)

    compute_recall(data, word2vec)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", help="path to file of the dataset.", default="../data/train_data.json")
    parser.add_argument("--embeddings", help="path to file of the saved embeddings", default="../embeddings/glove_50d.pkl")
    parser.add_argument("--word2vec", help="path to file of the word2vec embeddings.", default="../embeddings/w2v_vectors.kv")
    parser.add_argument("--w2i", help="path to file of the saved w2i", default="../embeddings/w2i.pkl")
    parser.add_argument("--use_gensim", help="indicate whether gensim vectors should be used", type=bool, default=False)
    args = parser.parse_args()

    main(args)

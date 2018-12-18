import argparse
from gensim.models import Word2Vec, KeyedVectors
import numpy as np
from pprint import pprint
import time

from rerank import rerank
from retrieve import retrieve
# from rewrite import Rewrite
from resource_prediction import ResourcePrediction
from data_utils import load_data, load_pickle, get_context, embed_sentence, \
    clean_sentence, get_resources, get_templates

global w2i


def run(data, gensim_model, embeddings, w2i):
    """
    Retrieve, rerank, rewrite.
    """
    prediction = ResourcePrediction(args.prediction_model_folder)
    # templates  = get_templates("../data/templates.pkl")
    # rewrite    = Rewrite(FOLDER, embeddings, w2i)

    for example in data:
        resources = []
        embedded_resources = []
        class_indices = []

        get_resources(example["documents"]["comments"], resources,
                      embedded_resources, embeddings, w2i)
        num_comments = len(resources)
        get_resources(example["documents"]["fact_table"], resources,
                      embedded_resources, embeddings, w2i)
        num_facts = len(resources) - num_comments
        get_resources(example["documents"]["plot"], resources,
                      embedded_resources, embeddings, w2i)
        num_plots = len(resources) - num_comments - num_facts
        get_resources(example["documents"]["review"], resources,
                      embedded_resources, embeddings, w2i)
        num_reviews = len(resources) - num_comments - num_facts - num_plots

        # Keep track of where each resource originated from.
        class_indices += [2]*num_comments
        class_indices += [3]*num_facts
        class_indices += [0]*num_plots
        class_indices += [1]*num_reviews

        chat = example["chat"]


        # Loop over each of the last three utterances in the chat (the context).
        for i in range(3, len(chat)-1):
            last_utterances = chat[i-3:i]
            response = chat[i+1]
            
            if len(response) > 0:
                embedded_utterances = [embed_sentence(utterance, embeddings, w2i) for utterance in
                                       last_utterances]
                context, embedded_context = get_context(last_utterances, embeddings, w2i)

                # Retrieve: Takes context and resources. Uses Word Mover's Distance
                # to obtain relevant resource candidates.
                similarities = retrieve(context, resources, gensim_model)

                # Predict: Takes context and predicts the category of the resource.
                # Take the maximum length as max and pad the context to maximum
                # length if it is too short.
                last_utterance = embedded_utterances[-2]
                padded_utterance = last_utterance[-args.max_length:]
                padded_utterance = np.pad(padded_utterance,
                    ((0, args.max_length - len(padded_utterance)), (0, 0)),
                    "constant", constant_values=(len(w2i)))
                predicted = prediction.predict(np.expand_dims(padded_utterance, 0))

                # Rerank: Takes ranked resource candidates and class prediction and
                # reranks them.
                ranked_resources, ranked_classes = rerank(resources, class_indices,
                                                          similarities, predicted)

                # Rewrite: Takes best resource candidate and its template and
                # generates response.
                best_response, best_template = rewrite.rerank(templates, ranked_resources, ranked_classes)

                # Response here is still embedding!
                response = rewrite.rewrite(best_response, best_template)
                # Rewrite from embedding to words:
                print("Final response: \n", response)
        #     return
        # return


def main(args):
    gensim_model = KeyedVectors.load(args.word2vec, mmap='r')

    data = load_data(args.file)
    embeddings = load_pickle(args.embeddings)
    w2i = load_pickle(args.w2i)
    run(data, gensim_model, embeddings, w2i)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", help="path to file of the dataset.", default="../data/train_data.json")
    parser.add_argument("--embeddings", help="path to file of the saved embeddings", default="../embeddings/glove_50d.pkl")
    parser.add_argument("--word2vec", help="path to file of the word2vec embeddings.", default="../embeddings/w2v_vectors.kv")
    parser.add_argument("--w2i", help="path to file of the saved w2i", default="../embeddings/w2i.pkl")
    parser.add_argument("--max_length", default=110, help="max context length for prediction")
    parser.add_argument("--prediction_model_folder", help="path to the folder that contains"
                                                          " the prediction model", default="../models/prediction")
    parser.add_argument("--gensim", help="indicate whether gensim vectors should be used", type=boolean, default=False)
    args = parser.parse_args()

    main(args)

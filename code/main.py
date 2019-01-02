import argparse
from gensim.models import Word2Vec, KeyedVectors
import numpy as np
from pprint import pprint
import time

from rerank import rerank
from retrieve import retrieve
from rewrite import Rewrite
from resource_prediction import ResourcePrediction
from data_utils import get_w2emb
import data_utils
import torch
from rouge import Rouge
from tqdm import tqdm

import sys
sys.path.append("rerankwrite")
from saliency_model import SaliencyPrediction


def run(data, word2vec):
    """
    Retrieve, rerank, rewrite.
    """
    prediction = ResourcePrediction(args.prediction_model_folder)
    templates  = data_utils.get_templates(args.templates)
    w2emb = get_w2emb(args.w2emb)
    emb_size = len(data_utils.embeddings[0])
    SOS_token = torch.Tensor([i for i in range(emb_size)]).unsqueeze(0).to(args.use_gpu)
    EOS_token = torch.Tensor([i+1 for i in range(emb_size)]).unsqueeze(0).to(args.use_gpu)
    w2emb["SOS_token"] = SOS_token.cpu()
    w2emb["EOS_token"] = EOS_token.cpu()
    cut_templates = [[temp[-args.max_length:] for temp in part_templ] for part_templ in templates]
    flattened_templates_emb_padded = [[np.pad(temp2, ((0, args.max_length - len(temp2)), (0, 0)),
                                            "constant", constant_values=(len(data_utils.w2i))) for temp2 in temp1]
                                            for temp1 in cut_templates]
    templates_padd = [torch.Tensor(class_tm) for class_tm in flattened_templates_emb_padded]
    rewrite = Rewrite(args.rewrite_model_folder, data_utils.embeddings, data_utils.w2i, SOS_token, EOS_token, templates_padd,
                        w2emb, args.use_gpu)

    rouge = Rouge()
    total = 0
    avg_rouge1 = 0
    avg_rouge2 = 0
    avg_rougeL = 0

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

        # Loop over each of the last three utterances in the chat (the context).
        for i in range(3, len(chat)-1):
            last_utterances = chat[i-3:i]
            response = chat[i+1]

            if len(response) > 0:
                embedded_utterances = [data_utils.embed_sentence(utterance) for
                                       utterance in last_utterances]
                context, embedded_context = data_utils.get_context(last_utterances)

                # Retrieve: Takes context and resources. Uses Word Mover's Distance
                # to obtain relevant resource candidates.
                similarities = retrieve(context, resources, word2vec)

                # Predict: Takes context and predicts the category of the resource.
                # Take the maximum length as max and pad the context to maximum
                # length if it is too short.
                if args.use_gensim:
                    constant_values = len(data_utils.embeddings.index2word)
                else:
                    constant_values = len(data_utils.w2i)

                last_utterance = embedded_utterances[-2]
                padded_utterance = last_utterance[-args.max_length:]
                padded_utterance = np.pad(padded_utterance,
                    ((0, args.max_length - len(padded_utterance)), (0, 0)),
                    "constant", constant_values=(constant_values))
                predicted = prediction.predict(np.expand_dims(padded_utterance, 0))

                # Rerank: Takes ranked resource candidates and class prediction and
                # reranks them.
                ranked_resources, ranked_classes = rerank(embedded_resources, class_indices,
                                                          similarities, predicted)

                best_resource, best_template = rewrite.rerank(ranked_resources[0], ranked_classes[0])
                best_response = rewrite.rewrite(best_resource, best_template)
                total += 1
                rouge_scores = rouge.get_scores(best_response, response)[0]
                avg_rouge1 += rouge_scores["rouge-1"]["f"]
                avg_rouge2 += rouge_scores["rouge-2"]["f"]
                avg_rougeL += rouge_scores["rouge-l"]["f"]

                # print(data_utils.convert_to_words(best_resource, w2emb))
                # print("___")
                # print(data_utils.convert_to_words(best_template, w2emb))
                # print("---")
                # print(best_response)

                # # Rewrite: Takes best resource candidate and its template and
                # # generates response.
                # best_response, best_template = rewrite.rerank(templates, ranked_resources, ranked_classes)
                #
                # # Response here is still embedding!
                # response = rewrite.rewrite(best_response, best_template)
                # # Rewrite from embedding to words:
                # print("Final response: \n", response)
    print("Average rouge1: " + str(avg_rouge1/total))
    print("Average rouge2: " + str(avg_rouge2/total))
    print("Average rougel: " + str(avg_rougeL/total))


def main(args):
    word2vec = KeyedVectors.load(args.word2vec, mmap='r')
    data = data_utils.load_data(args.file)

    if args.use_gensim:
        data_utils.load_gensim(args.word2vec)
    else:
        data_utils.load_embeddings(args.embeddings)
        data_utils.load_w2i(args.w2i)

    run(data, word2vec)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", help="path to file of the dataset.", default="../data/test_data.json")
    parser.add_argument("--embeddings", help="path to file of the saved embeddings", default="../embeddings/glove_50d.pkl")
    parser.add_argument("--word2vec", help="path to file of the word2vec embeddings.", default="../embeddings/w2v_vectors.kv")
    parser.add_argument("--w2i", help="path to file of the saved w2i", default="../embeddings/w2i.pkl")
    parser.add_argument("--max_length", default=110, help="max context length for prediction")
    parser.add_argument("--prediction_model_folder", help="path to the folder that contains"
                                                          " the prediction model", default="../models/prediction")
    parser.add_argument("--use_gensim", help="indicate whether gensim vectors should be used", type=bool, default=False)
    parser.add_argument("--use_gpu", help="what to use: cuda/cpu", default="cuda")
    parser.add_argument("--rewrite_model_folder", help="where the rewrite models are", default="../models/rewrite/")
    parser.add_argument("--w2emb", help="folder where w2emb is", default="../embeddings/w2emb.pkl")
    parser.add_argument("--templates", help="path to file of the templates", default="../data/templates.pkl")
    args = parser.parse_args()

    main(args)

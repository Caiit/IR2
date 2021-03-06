# torch.save(model.state_dict(), PATH)
# torch.save(model, PATH)

from saliency_model import SaliencyPrediction
from data_utils import load_data, load_pickle, get_context, embed_sentence, \
    clean_sentence, get_resources, get_templates
import torch
import torch.nn as nn
import torch.optim as optim
from rouge import Rouge
import matplotlib.pyplot as plt
import argparse
import numpy as np
from tqdm import tqdm
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def convert_to_words(complete_sent_emb, w2emb):
    output_sentence = []
    for word in complete_sent_emb:
        emb_dists = [torch.norm(torch.Tensor(word) - torch.Tensor(embs)).item() for embs in list(w2emb.values())]
        index = np.argmin(emb_dists)
        output_sentence.append(list(w2emb.keys())[index])

    return " ".join(output_sentence)


def make_templates(args, templates_emb, w2emb, w2i):
    """
    Gets embedded + unembedded templates if they don't exist.
    """

    if os.path.exists("templates.pkl"):
        templates = load_pickle("templates.pkl")
        templates_emb_pad = load_pickle("templates_emb.pkl")
    else:
        # Flatten templates
        templates_emb = [y for x in templates_emb for y in x]
        # Cut templates to maximum length
        templates_emb = [temp[-args.max_length:] for temp in templates_emb]
        # Pad embeddings
        templates_emb_pad = [np.pad(temp1, ((0, args.max_length - len(temp1)),
                             (0, 0)), "constant", constant_values=(len(w2i)))
                             for temp1 in templates_emb]
        # Convert embedded templates to word templates
        templates = [convert_to_words(sent, w2emb) for sent in templates_emb]

        save_pickle("templates.pkl", templates)
        save_pickle("templates_emb.pkl", templates_emb_pad)
    return templates, templates_emb_pad

def train(args):
    print("Load data...")
    data_train = load_data(args.folder + "/train_data.json")
    data_test = load_data(args.folder + "/dev_data.json")
    embeddings = load_pickle(args.embeddings)
    w2i = load_pickle(args.w2i)
    w2emb = load_pickle(args.w2emb)
    templates_emb  = get_templates("../../data/templates.pkl")


    print("Do the templates...")
    templates, templates_emb = make_templates(args, templates_emb, w2emb, w2i)


    print("Now load the model...")
    emb_size = len(embeddings[0])
    model = SaliencyPrediction(emb_size*args.max_length, device).to(device)
    loss_func = nn.BCELoss()
    optimizer = optim.SGD(model.parameters(), lr = 0.001, momentum=0.9)
    rouge = Rouge()


    print("Read in train data...")
    resources = []
    embedded_resources = []
    for example in tqdm(data_train):
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


    print("Read in test data...")
    resources_test = []
    embedded_resources_test = []
    for example in tqdm(data_test):
        get_resources(example["documents"]["comments"], resources_test,
                      embedded_resources_test, embeddings, w2i)
        num_comments = len(resources)
        get_resources(example["documents"]["fact_table"], resources_test,
                      embedded_resources_test, embeddings, w2i)
        num_facts = len(resources) - num_comments
        get_resources(example["documents"]["plot"], resources_test,
                      embedded_resources_test, embeddings, w2i)
        num_plots = len(resources) - num_comments - num_facts
        get_resources(example["documents"]["review"], resources_test,
                      embedded_resources_test, embeddings, w2i)
        num_reviews = len(resources) - num_comments - num_facts - num_plots

    print("Now learn.....")
    total_resources = len(embedded_resources)
    all_temps = torch.Tensor(templates_emb)

    for epoch in range(5):
        print("Epoch: " + str(epoch))
        avg_loss = 0
        for i, resource in tqdm(enumerate(embedded_resources)):
            sent = " ".join(resources[i])
            if sent == "" or sent == "eod":
                continue;
            optimizer.zero_grad()

            padd_resource = resource[-args.max_length:]
            padd_resource = np.pad(padd_resource, ((0, args.max_length -
                                   len(padd_resource)), (0, 0)), "constant",
                                   constant_values=(len(w2i)))

            actual_scores = []
            all_res = torch.Tensor(padd_resource).unsqueeze(0).repeat(20, 1, 1)
            size_inp = all_res.size()
            for j, template in enumerate(templates_emb):
                try:
                    actual_score = rouge.get_scores(templates[j], " ".join(resources[i]))[0]["rouge-1"]["f"]
                except:
                    actual_score = 0
                actual_scores.append(actual_score)
            x1 = all_res.reshape(size_inp[0], size_inp[1]*size_inp[2]).to(device)
            x2 = all_temps.reshape(size_inp[0], size_inp[1]*size_inp[2]).to(device)
            actual_scores = torch.Tensor(actual_scores).unsqueeze(1).to(device)
            scores = model(x1, x2)
            loss = loss_func(scores, actual_scores)
            avg_loss += loss.item()
            loss.backward()
            optimizer.step()
        print("For this epoch, we found avg_loss: " + str(avg_loss/total_resources))
        torch.save(model, "../../models/rewrite/saliency_" + str(epoch) + ".pt")

        model.eval()

        with torch.no_grad():
            total_loss = 0
            amount_res = len(resources)
            for i, resource in tqdm(enumerate(embedded_resources_test)):
                sent = " ".join(resources_test[i])
                if sent == "" or sent == "eod":
                    continue;
                padd_resource = resource[-args.max_length:]
                padd_resource = np.pad(padd_resource, ((0, args.max_length -
                                       len(padd_resource)), (0, 0)), "constant",
                                       constant_values=(len(w2i)))
                actual_scores = []
                all_res = torch.Tensor(padd_resource).unsqueeze(0).repeat(20, 1, 1)
                size_inp = all_res.size()
                for j, template in enumerate(templates_emb):
                    try:
                        actual_score = rouge.get_scores(templates[j], " ".join(resources[i]))[0]["rouge-1"]["f"]
                    except:
                        actual_score = 0
                    actual_scores.append(actual_score)
                x1 = all_res.reshape(size_inp[0], size_inp[1]*size_inp[2]).to(device)
                x2 = all_temps.reshape(size_inp[0], size_inp[1]*size_inp[2]).to(device)
                actual_scores = torch.Tensor(actual_scores).unsqueeze(1).to(device)

                scores = model(x1, x2)
                loss = loss_func(scores, actual_scores)
                total_loss += loss.item()
            print("Average loss is: " + str(total_loss/amount_res))
        model.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", help="path to file of the dataset.", default="../../data")
    parser.add_argument("--embeddings", help="path to file of the saved embeddings", default="../../embeddings/glove_100d.pkl")
    parser.add_argument("--word2vec", help="path to file of the word2vec embeddings.", default="../../embeddings/w2v_vectors.kv")
    parser.add_argument("--w2i", help="path to file of the saved w2i", default="../../embeddings/w2i.pkl")
    parser.add_argument("--w2emb", help="path to the file of the saved w2emb", default="../../embeddings/w2emb.pkl")
    parser.add_argument("--max_length", help="max length of sentences", default=110)

    args = parser.parse_args()
    train(args)

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

def train(args):
    templates  = get_templates("../data/templates.pkl")

    data_train = load_data(args.folder + "/train_data.json")
    data_test = load_data(args.folder + "/test_data.json")
    embeddings = load_pickle(args.embeddings)
    w2i = load_pickle(args.w2i)
    flattened_templates = [embed_sentence(item, embeddings, w2i) for templates_class in templates for item in templates_class]
    flattened_templates_raw = [item for templates_class in templates for item in templates_class]

    # Embedding size of length sent?
    saliency_model = SaliencyPrediction(len(embeddings[0]))
    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.Adam(saliency_model.parameters())
    rouge = Rouge()

    for example in data_train:
        resources = []
        embedded_resources = []
        class_indices = []

        num_comments = get_resources(example["documents"]["comments"],
                                     resources, embedded_resources, embeddings, w2i)
        num_facts = get_resources(example["documents"]["fact_table"], resources,
                                  embedded_resources, embeddings, w2i)
        num_plots = get_resources(example["documents"]["plot"], resources,
                                  embedded_resources, embeddings, w2i)
        num_reviews = get_resources(example["documents"]["review"], resources,
                                    embedded_resources, embeddings, w2i)

    # Not yet batch
    total_resources = len(embedded_resources)
    for epoch in range(5):
        print("Epoch: " + str(epoch))
        for i, resource in enumerate(embedded_resources):
            optimizer.zero_grad()
            loss = 0
            for j, template in enumerate(flattened_templates):
                score = saliency_model.forward(resource, template)
                actual_score = rouge.get_scores(flattened_templates_raw[j], resources[i])

                loss += loss_func(score, actual_score)
            # Normalize
            loss /= (j+1)
            loss.backward()
            optimizer.step()
            print("Step: " + str(i) + "/" + total_resources + ", the loss is: " + str(loss))

    torch.save(saliency_model, "../models/rewrite/saliency.pt")

    for example in data_test:
        resources = []
        embedded_resources = []
        class_indices = []

        num_comments = get_resources(example["documents"]["comments"],
                                     resources, embedded_resources, embeddings, w2i)
        num_facts = get_resources(example["documents"]["fact_table"], resources,
                                  embedded_resources, embeddings, w2i)
        num_plots = get_resources(example["documents"]["plot"], resources,
                                  embedded_resources, embeddings, w2i)
        num_reviews = get_resources(example["documents"]["review"], resources,
                                    embedded_resources, embeddings, w2i)

    total_resources = len(embedded_resources)
    for i, resource in enumerate(embedded_resources):
        loss = 0
        for j, template in enumerate(flattened_templates):
            score = saliency_model.forward(resource, template)
            actual_score = rouge.get_scores(flattened_templates_raw[j], resources[i])

            loss += loss_func(score, actual_score)
    loss /= total_resources
    print("Loss on the test set: " + str(loss))


    return




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", help="path to file of the dataset.", default="../data")
    parser.add_argument("--embeddings", help="path to file of the saved embeddings", default="../embeddings/glove_50d.pkl")
    parser.add_argument("--word2vec", help="path to file of the word2vec embeddings.", default="../embeddings/w2v_vectors.kv")
    parser.add_argument("--w2i", help="path to file of the saved w2i", default="../embeddings/w2i.pkl")

    args = parser.parse_args()
    train(args)

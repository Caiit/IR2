import argparse
import json
from nltk.tokenize import word_tokenize
import numpy as np
import pickle
import re
import string
import torch.nn.functional as F
import torch
from templategenerator import TemplateGenerator

global embeddings
global w2i

def load_data(filename):
    """
    Loads the data. Data: chat, chat_id, documents (comments, fact_table, plot,
    review), imdb_id, labels, movie_name, spans
    """

    with open(filename, encoding="utf-8") as f:
        data = json.load(f)
    return data

def load_pickle(file_path):
    """
    Loads the saved Pickle file.
    """

    with open(file_path, "rb") as f:
        output = pickle.load(f)
    return output

def batches(data, batch_size):
    for i in range(0, len(data), batch_size):
        yield data[i:i + batch_size]


def run(data, resource_type):
    """
    Generate template sentences
    """
    embedding_dim = 50
    batch_size = 16
    max_length = 50
    resources = []
    for example in data:
        resources = get_resources(example["documents"][resource_type], resources, max_length)

    model = TemplateGenerator(batch_size, len(w2i), embedding_dim)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.00000001)

    # AANAPSSEN!!!!
    batchs = list(batches(resources, batch_size))

    for i, batch in enumerate(batchs):
        batch = np.array(batch)
        batch_input = torch.FloatTensor(batch[:-1])
        batch_target = torch.LongTensor(batch[1:])
        batch_preds = model.forward(batch_input)

        max_sent_size = len(batch_input[0])

        batch_target = batch_target.float()
        total_sum = sum([torch.exp(torch.dot(batch_preds[i][w], batch_target[i][w])) for w in range(max_length) for i in range(batch_input.shape[0] - 1)])
        loss = sum([torch.exp(torch.dot(batch_preds[i][w], batch_target[i][w]))/total_sum for w in range(max_length) for i in range(batch_input.shape[0] - 1)])/(max_sent_size * max_length)
        print(i, " loss ", loss)
        #optimizer.grad()
        loss.backward()
        optimizer.step()
    generate_samples(model, max_sent_size, 1)



def embed_sentence(sentence, max_length):
    """
    Embeds a sentence after cleaning it.
    """

    global embeddings
    global w2i
    global w2emb

    sentence = clean_sentence(sentence)
    sentence = word_tokenize(sentence)
    sentence = sentence[:max_length]
    # add start and stop symbol
    sentence = ['-'] + sentence + ['.']

    embedded_sentence = np.ones((max_length, embeddings[0].shape[0])) * len(w2i)

    for i, word in enumerate(sentence):
        if i == max_length: break
        if w2i.get(word):
            index = w2i[word]
            embedding = embeddings[index]
            embedded_sentence[i] = embedding
            w2emb[word] = embedding
        else:
            embedded_sentence[i] = 0
    return embedded_sentence


def clean_sentence(sentence):
    """
    Cleans a sentence by lowercasing it and removing punctuation.
    """

    sentence = sentence.lower()
    translator = str.maketrans('', '', string.punctuation)
    sentence = sentence.translate(translator)
    return sentence


def get_resources(document, resources, max_length):
    """
    Obtains the resources in a document.
    """

    regex = re.compile("[@_!#$%^&*()<>?/\|}{~:]")
    # print(document)

    if type(document) is list:
        for resource in document:
            # Check if the string isn't just special characters, e.g. {}
            if not regex.search(resource):
                embedded_resource = embed_sentence(resource, max_length)
                resources.append(embedded_resource)
    elif type(document) is str:
        embedded_resource = embed_sentence(document, max_length)
        resources.append(embedded_resource)
    elif type(document) is dict:
        for key, value in document.items():
            embedded_resource = embed_sentence(value, max_length)
            resources.append(embedded_resource)
    return resources


def generate_samples(model, sequence_length, temperature):
    generated_samples = []
    for _ in range(5):
        init_word = '-'
        index = w2i[init_word]
        input_embedding = torch.Tensor(embeddings[index]).unsqueeze(0).unsqueeze(0)
        outputs = [input_embedding]
        # generate sentence of seq_length
        for i in range(5):
            print(i)
            output_embedding = model.forward(input_embedding)[:, -1,:].unsqueeze(0)
            print(input_embedding.shape)
            print(output_embedding.shape)
            input_embedding = torch.cat((input_embedding, output_embedding), 1)
            outputs.append(output_embedding.squeeze(0).squeeze(0))
        output_sentence = []
        for output in outputs:
            emb_dists = [torch.norm(output - torch.Tensor(embs)).item() for embs in list(w2emb.values())]
            index = np.argmin(emb_dists)
            output_sentence.append(list(w2emb.keys())[index])

        generated_samples.append(" ".join(output_sentence))
        print(generated_samples)
    return generated_samples


def main(args):
    global embeddings
    global w2i
    global w2emb

    data = load_data(args.file)
    embeddings = load_pickle(args.embeddings)
    w2i = load_pickle(args.w2i)
    w2emb = dict()
    run(data, args.resource_type)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("file", help="path to file of the dataset.")
    parser.add_argument("embeddings", help="path to file of the saved embeddings")
    parser.add_argument("w2i", help="path to file of the saved w2i")
    parser.add_argument("resource_type", type=str, help="which resource_type")
    args = parser.parse_args()

    main(args)

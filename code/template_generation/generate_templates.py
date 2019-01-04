import argparse
import json
from nltk.tokenize import word_tokenize
import numpy as np
import pickle
import re
import string
import torch.nn.functional as F
import torch
#from templategenerator import TemplateGenerator
from gensim.summarization.summarizer import summarize
from gensim.summarization.textcleaner import clean_text_by_sentences
from sklearn.cluster import KMeans


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
    """ Generates the five templates.
    """

    global embeddings
    global w2i

    max_length = 110
    resources = []
    for example in data:
        resources = get_resources2(example["documents"][resource_type], resources, max_length)

    print('Now doing the embeddings....')
    all_embedded = []
    avg_embedded = []
    all_sents = []
    for res in resources:
        all_res = " ".join(res)
        if len(clean_text_by_sentences(all_res)) > 1:
            sent_temp = clean_sentence(summarize(all_res))
            sent_temp = sent_temp.split()
            embedded_sentence = np.ones((len(sent_temp), embeddings[0].shape[0])) * len(w2i)
            for i, w in enumerate(sent_temp):
                if w2i.get(w):
                    index = w2i[w]
                    embedding = embeddings[index]
                    embedded_sentence[i] = embedding
                    w2emb[w] = embedding
                else:
                    embedded_sentence[i] = 0
            if len(embedded_sentence) > 0 and len(embedded_sentence) < max_length:
                avg_embedded.append(avg_embed_sentence(embedded_sentence))
                all_embedded.append(embedded_sentence)
                all_sents.append(sent_temp)

    print('Now doing kmeans....')
    kmeans = KMeans(n_clusters=args.n_templates)
    trying = kmeans.fit_predict(avg_embedded)
    clusters = kmeans.cluster_centers_

    print('Now getting the', args.n_templates, 'sentences.....')
    best_sents = []
    for i in range(len(clusters)):
        cluster = clusters[i]
        # cosine similarity
        cos_sim_max = 0
        for j, sent in enumerate(avg_embedded):
            cos_sim = np.dot(cluster, sent)/(np.linalg.norm(cluster)*np.linalg.norm(sent))
            if cos_sim > cos_sim_max:
                cos_sim_max = cos_sim
                best_sent = all_embedded[j]
                best_sentt = all_sents[j]
        print(best_sentt)
        best_sents.append(best_sent)


    print('Saving now')

    pkl_file_name = "embeddings_" + str(args.n_templates) + "_" + resource_type + ".pkl"
    text_file = open(pkl_file_name, "wb")
    pickle.dump(best_sents, text_file)
    text_file.close()

    print('Saved....')


def clean_sentence2(sentence):
    """
    Cleans a sentence by removing punctuation but not '.'.
    """
    translator = str.maketrans('', '', '!"#$%&\'()*+,-/:;<=>?@[\\]^_`{|}~')
    sentence = sentence.translate(translator)
    return sentence

def get_resources2(document, resources, max_length):
    """
    Obtains the resources in a document.
    """

    regex = re.compile("[@_!#$%^&*()<>?/\|}{~:]")
    # print(document)

    if type(document) is list:
        for resource in document:
            # Check if the string isn't just special characters, e.g. {}
            if not regex.search(resource):
                sentence = clean_sentence2(resource)
                sentence = sentence.split(' ')
                sentence = sentence[:max_length]
                # add start and stop symbol
                resources.append(sentence)
    elif type(document) is str:
        sentence = clean_sentence2(document)
        sentence = sentence.split(' ')
        sentence = sentence[:max_length]
        # add start and stop symbol
        resources.append(sentence)
    elif type(document) is dict:
        for key, value in document.items():
            #double if else it does not work for some reason
            if type(value) != list:
                if type(value) != float:
                    sentence = clean_sentence2(value)
                    sentence = sentence.split(' ')
                    sentence = sentence[:max_length]
                    # add start and stop symbol
                    resources.append(sentence)
    return resources


def avg_embed_sentence(embedded_sentence):
    """
    Averages the word embeddings of one sentence.
    """
    avg_sent = sum(embedded_sentence)/len(embedded_sentence)
    return avg_sent


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
    parser.add_argument("--file", help="path to file of the dataset.", default="../../data/train_data.json")
    parser.add_argument("--embeddings", help="path to file of the saved embeddings", default="../../embeddings/glove_100d.pkl")
    parser.add_argument("--w2i", help="path to file of the saved w2i", default="../../embeddings/w2i.pkl")
    parser.add_argument("--resource_type", type=str, help="which resource type: plot, fact_table, comments, review", default="plot")
    parser.add_argument("--n_templates", type=int, help="amount templates per resource", default="5")
    args = parser.parse_args()

    main(args)

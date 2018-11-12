import argparse
import json
import numpy as np
import pickle
from pprint import pprint
import re
import string

from retrieve import retrieve

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


def train(data):
    """
    Retrieve, rerank, rewrite.
    """

    for example in data:
        resources = []

        context = get_context(example["chat"])
        resources = get_resources(example["documents"]["comments"], resources)
        resources = get_resources(example["documents"]["fact_table"], resources)
        resources = get_resources(example["documents"]["plot"], resources)
        resources = get_resources(example["documents"]["review"], resources)

        # Retrieve: Takes context and resources. Uses cosine similarity to
        # obtain relevant resource candidates.
        ranked_resources = retrieve(context, resources)
        # Rerank: Takes relevant resource candidates and templates (picked from
        # required response category).
        # Rewrite: Takes best resource candidate and its template and generates
        # response.

        return


def get_context(conversation):
    """
    Takes the last two utterances from a conversation and the current utterance
    to generate a context.
    """

    last_utterances = conversation[-3:]
    context = " ".join([str(string) for string in last_utterances])
    embedded_context = embed_sentence(context)
    return embedded_context


def embed_sentence(sentence):
    """
    Embeds a sentence after cleaning it.
    """

    global embeddings
    global w2i

    sentence = clean_sentence(sentence)
    sentence = sentence.split()

    embedded_sentence = np.zeros((len(sentence), embeddings[0].shape[0]))

    for i, word in enumerate(sentence):
        if w2i.get(word):
            index = w2i[word]
            embedding = embeddings[index]
            embedded_sentence[i] = embedding
    return embedded_sentence


def clean_sentence(sentence):
    """
    Cleans a sentence by lowercasing it and removing punctuation.
    """

    sentence = sentence.lower()
    translator = str.maketrans('', '', string.punctuation)
    sentence = sentence.translate(translator)
    return sentence


def get_resources(document, resources):
    """
    Obtains the resources in a document.
    """

    regex = re.compile("[@_!#$%^&*()<>?/\|}{~:]")
    # print(document)

    if type(document) is list:
        for resource in document:
            # Check if the string isn't just special characters, e.g. {}
            if not regex.search(resource):
                embedded_resource = embed_sentence(resource)
                resources.append(embedded_resource)
    elif type(document) is str:
        embedded_resource = embed_sentence(document)
        resources.append(embedded_resource)
    elif type(document) is dict:
        for key, value in document.items():
            embedded_resource = embed_sentence(value)
            resources.append(embedded_resource)
    return resources


def main(args):
    global embeddings
    global w2i

    data = load_data(args.file)
    embeddings = load_pickle(args.embeddings)
    w2i = load_pickle(args.w2i)
    train(data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", help="path to file of the dataset.")
    parser.add_argument("--embeddings", help="path to file of the saved " +
                        "embeddings")
    parser.add_argument("--w2i", help="path to file of the saved w2i")
    args = parser.parse_args()

    main(args)

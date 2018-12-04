# Terminal example:
# python main.py --file=data/train_data.json --embeddings=data/embeddings.pkl
# --w2i=w2i.pkl


import argparse
import json
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import numpy as np
import pickle
from pprint import pprint
import re
import string

from rerank import rerank
from retrieve import retrieve
from resource_prediction import ResourcePrediction

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


def run(data):
    """
    Retrieve, rerank, rewrite.
    """
    prediction = ResourcePrediction(args.prediction_model_folder)

    for example in data:
        resources = []
        class_indices = []

        resources, num_comments = \
            get_resources(example["documents"]["comments"], resources)
        resources, num_facts = get_resources(example["documents"]["fact_table"],
                                             resources)
        resources, num_plots = get_resources(example["documents"]["plot"],
                                             resources)
        resources, num_reviews = get_resources(example["documents"]["review"],
                                               resources)

        # Keep track of where each resource originated from.
        class_indices += [2]*num_comments
        class_indices += [3]*num_facts
        class_indices += [0]*num_plots
        class_indices += [1]*num_reviews

        chat = example["chat"]

        # Loop over each of the last three utterances in the chat (the context).
        for i in range(3, len(chat)+1):
            last_utterances = chat[i-3:i]
            embedded_utterances = [embed_sentence(utterance) for utterance in
                                   last_utterances]
            concat_context = get_context(last_utterances)

            # Retrieve: Takes context and resources. Uses cosine similarity to
            # obtain relevant resource candidates.
            similarities = retrieve(concat_context, resources)

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

            return

        return


def get_context(last_utterances):
    """
    Takes the last two utterances from a conversation and the current utterance
    to generate a context.
    """

    context = " ".join([str(string) for string in last_utterances])
    embedded_context = embed_sentence(context)
    return embedded_context


def embed_sentence(sentence):
    """
    Embeds a sentence.
    """

    global embeddings
    global w2i

    sentence = clean_sentence(sentence)
    embedded_sentence = np.zeros((len(sentence), embeddings[0].shape[0]))

    for i, word in enumerate(sentence):
        if w2i.get(word):
            index = w2i[word]
            embedding = embeddings[index]
            embedded_sentence[i] = embedding
    return embedded_sentence


def clean_sentence(sentence):
    """
    Cleans a sentence by lowercasing it, removing punctuation and stop words,
    and tokenizing it.
    """

    translator = str.maketrans('', '', string.punctuation)
    stop_words = set(stopwords.words('english'))

    sentence = sentence.lower()
    sentence = sentence.translate(translator)
    sentence = word_tokenize(sentence)
    # TODO: Remove stop words as below.
    # sentence = [word for word in sentence if word not in stop_words]

    return sentence


def get_resources(document, resources):
    """
    Obtains the resources in a document and returns how many resources of a
    category were obtained.
    """

    regex = re.compile("[@_!#$%^&*()<>?/\|}{~:]")
    amount_resources = len(resources)

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

    amount_category = len(resources) - amount_resources
    return resources, amount_category


def main(args):
    global embeddings
    global w2i

    data = load_data(args.file)
    embeddings = load_pickle(args.embeddings)
    w2i = load_pickle(args.w2i)
    run(data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("file", help="path to file of the dataset.")
    parser.add_argument("embeddings", help="path to file of the saved embeddings")
    parser.add_argument("w2i", help="path to file of the saved w2i")
    parser.add_argument("--max_length", default=110, help="max context length for prediction")
    parser.add_argument("prediction_model_folder", help="path to the folder that contains"
                                                          " the prediction model")
    args = parser.parse_args()

    main(args)

# Terminal exmaple:
# python dataset.py --file=data/train_data.json --glove=glove/glove.6B.50d.txt
# --embeddings=embeddings.pkl --w2i=w2i.pkl

import argparse
from collections import Counter
import json
import numpy as np
import pickle
from pprint import pprint
import re
import string

TEXT = []


def read_data(filename):
    """
    Reads the JSON data.
    """

    with open(filename, encoding="utf-8") as f:
        data = json.load(f)
    return data


def preprocess_data(data, glove_file):
    """
    Preprocesses the data by obtaining a vocabulary and the corresponding
    embeddings.
    """

    vocab = get_vocab(data)
    w2i = {word: idx for idx, word in enumerate(vocab)}
    i2w = {idx: word for idx, word in enumerate(vocab)}
    embeddings = load_embeddings(glove_file, w2i)
    return embeddings, w2i


def get_vocab(data):
    """
    Obtains the vocabulary from the data.
    """

    for item in data:
        get_text(item)
    # Only keep unique words
    text = str(set(TEXT))
    # Lowercase words
    text = text.lower()
    # Remove punctuation
    translator = str.maketrans('', '', string.punctuation)
    text = text.translate(translator)
    # Count occurrences
    word_count = Counter(text.split())
    vocab = reduce_vocab(word_count)
    return vocab


def get_text(data_entry):
    """
    Retrieves the text in the data by looking in the following keys: chat,
    documents[comments], documents[fact_table], documents[plot],
    documents[review], movie_name and spans.
    """

    for key, value in data_entry.items():
        if key != "chat_id" and key != "imdb_id" and key != "labels":
            if isinstance(value, dict):
                get_text(value)
            elif isinstance(value, list):
                TEXT.append(" ".join([str(item) for item in value]))
            else:
                TEXT.append(value)


def reduce_vocab(counter):
    """
    Reduces the vocabulary by only keeping words the occur 20 times or more.
    """

    sorted_counter = counter.most_common()
    vocab = []

    for word, count in sorted_counter:
        if count >= 20:
            vocab.append(word)
        else:
            break
    return vocab


def load_embeddings(file_path, w2i, embedding_dim=50):
    """
    Uses a text file of GloVe embeddings to load all possible embeddings into
    a dictionary.
    """

    with open(file_path) as f:
        embeddings = np.zeros((len(w2i), embedding_dim))

        for line in f.readlines():
            split_line = line.split()
            word = split_line[0]
            index = w2i.get(word)

            if index:
                embedding = np.array(split_line[1:], dtype="float32")
                embeddings[index] = embedding
    return embeddings


def save_pickle(file_path, input):
    """
    Saves the input as a Pickle file.
    """

    with open(file_path, "wb") as f:
        pickle.dump(input, f)


def main(args):
    # Data: chat, chat_id, documents (comments, fact_table, plot, review),
    # imdb_id, labels, movie_name, spans
    data = read_data(args.file)
    embeddings, w2i = preprocess_data(data, args.glove)
    save_pickle(args.embeddings, embeddings)
    save_pickle(args.w2i, w2i)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", help="path to file of the dataset.")
    parser.add_argument("--glove", help="path to glove file.")
    parser.add_argument("--embeddings", help="path to where embeddings file " +
                        " will be saved.")
    parser.add_argument("--w2i", help="path to w2i file")
    args = parser.parse_args()

    main(args)

import argparse
from collections import Counter
import json
from nltk.tokenize import word_tokenize
import numpy as np
import pickle
from pprint import pprint
import string
import matplotlib.pyplot as plt

TEXT = []
LABELS = []


def read_data(filename):
    """
    Reads the JSON data.
    """

    with open(filename, encoding="utf-8") as f:
        data = json.load(f)
    return data


def preprocess_data(data, glove_file, embedding_dim):
    """
    Preprocesses the data by obtaining a vocabulary and the corresponding
    embeddings.
    """

    vocab, word_count = get_vocab(data)
    w2i = {word: idx for idx, word in enumerate(vocab)}
    i2w = {idx: word for idx, word in enumerate(vocab)}
    embeddings, w2emb = load_embeddings(glove_file, w2i, embedding_dim)
    return embeddings, w2i, word_count, w2emb


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
    word_count = Counter(word_tokenize(text))
    vocab = reduce_vocab(word_count)
    return vocab, word_count


def get_text(data_entry):
    """
    Retrieves the text in the data by looking in the following keys: chat,
    documents[comments], documents[fact_table], documents[plot],
    documents[review], movie_name and spans, and obtain the labels of the
    responses.
    """
    for key, value in data_entry.items():
        if key != "chat_id" and key != "imdb_id" and key != "labels":
            if isinstance(value, dict):
                get_text(value)
            elif isinstance(value, list):
                TEXT.append(" ".join([str(item) for item in value]))
            else:
                TEXT.append(value)
        if key == "labels":
            LABELS.extend(value)


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


def load_embeddings(file_path, w2i, embedding_dim):
    """
    Uses a text file of GloVe embeddings to load all possible embeddings into
    a dictionary.
    """

    w2emb = dict()
    with open(file_path) as f:
        embeddings = np.zeros((len(w2i), embedding_dim))

        for line in f.readlines():
            split_line = line.split()
            word = split_line[0]
            index = w2i.get(word)

            if index:
                try:
                    embedding = np.array(split_line[1:], dtype="float32")
                    embeddings[index] = embedding
                    w2emb[word] = embedding
                except ValueError:
                    pass
    #print("Embedding:", count)
    return embeddings, w2emb


def save_pickle(file_path, input):
    """
    Saves the input as a Pickle file.
    """

    with open(file_path, "wb") as f:
        pickle.dump(input, f)


def compute_word_distribution(word_count):
    print("Creating plot...")
    n = 100
    word_count = word_count.most_common(n)
    labels, values = zip(*word_count)

    indexes = np.arange(len(labels))
    width = 1

    plt.bar(indexes, values, width)
    if n < 200:
        plt.xticks(indexes + width * 0.25, labels, rotation=90)
        plt.tick_params(axis='both', which='major', labelsize=6)

    plt.show()


def compute_label_distribution():
    label_indices, values = zip(*Counter(LABELS).items())

    indexes = np.arange(len(label_indices))
    width = 1

    label_names = ["Plot", "Review", "Comments", "Fact", "None"]

    labels = [label_names[i] for i in label_indices]

    plt.bar(indexes, values, width)
    plt.xticks(indexes + width * 0.25, labels, rotation=90)
    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.ylabel("#Occurences")
    plt.title("Resource category distribution")

    plt.show()


def main(args):
    data = read_data(args.file)
    embeddings, w2i, word_count, w2emb = preprocess_data(data, args.glove,
                                                  args.embedding_dim)
    save_pickle(args.embeddings, embeddings)
    save_pickle(args.w2i, w2i)
    save_pickle(args.w2emb, w2emb)

    if args.stats:
        compute_word_distribution(word_count)
        compute_label_distribution()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", help="path to file of the dataset.", default="../data/train_data.json")
    parser.add_argument("--glove", help="path to glove file.", default="../glove/glove.6B.50d.txt")
    parser.add_argument("--embedding_dim", type=int, help="dimension of the embeddings", default=50)
    parser.add_argument("--embeddings", help="path to where embeddings file will be saved.", default="../embeddings/glove_50d.pkl")
    parser.add_argument("--w2i", help="path to w2i file", default="../embeddings/w2i.pkl")
    parser.add_argument("--w2emb", help="path to w2emb file", default="../embeddings/w2emb.pkl")
    parser.add_argument("--stats", type=bool, default=False, help="compute statistics about the dataset")
    args = parser.parse_args()

    main(args)

import json
import numpy as np
import pickle
import string
from nltk.tokenize import word_tokenize

global use_gensim
global embeddings
global w2i


def load_data_and_labels(filename, data_folder, max_length, using_gensim,
                         embedding_file, w2i_file):
    """Load sentences and labels"""
    global embeddings
    global  w2i
    global use_gensim

    use_gensim = using_gensim

    with open(filename, encoding="utf-8") as f:
        data = json.load(f)
    with open(embedding_file, "rb") as f:
        embeddings = pickle.load(f)
    with open(w2i_file, "rb") as f:
        w2i = pickle.load(f)

    x_raw = []
    y_raw = []
    for example in data:
        chat = example["chat"]
        labels = example["labels"]
        # Get current utterance and response label
        for i in range(len(chat) - 1):
            # Don't take the None into account, since its for speaker 1
            if labels[i + 1] < 4:
                utterance = chat[i]  # Current utterance
                label_hot = np.zeros(4)
                label_hot[labels[i + 1]] = 1  # Label of response
                x_raw.append(embed_sentence(utterance, max_length))
                y_raw.append((label_hot))
        # OLD: get whole context as input and response label
        # for i in range(3, len(chat) - 1):
        #     # Don't take the None into account, since its for speaker 1
        #     if labels[i + 1] < 4:
        #         sentence = " ".join(chat[i - 3: i])
        #         if len(sentence.split()) > 0:
        #             x_raw.append(embed_sentence(sentence, max_length))
        #             label_hot = np.zeros(4)
        #             label_hot[labels[i + 1]] = 1  # Label of response
        #             y_raw.append(label_hot)
    return x_raw, y_raw


def embed_sentence(sentence, max_length):
    """
    Embeds a sentence after cleaning it.
    """

    global embeddings
    global w2i
    global use_gensim

    sentence = clean_sentence(sentence)
    sentence = word_tokenize(sentence)
    sentence = sentence[-max_length:]

    if use_gensim:
        embedding_dim = embeddings.vector_size
    else:
        embedding_dim = embeddings[0].shape[0]

    embedded_sentence = np.ones((max_length, embedding_dim)) * len(w2i)

    for i, word in enumerate(sentence):
        if i == max_length: break

        if use_gensim:
            try:
                embedded_sentence[i] = embeddings[word]
            except KeyError:
                embedded_sentence[i] = 0
        else:
            if w2i.get(word):
                index = w2i[word]
                embedding = embeddings[index]
                embedded_sentence[i] = embedding
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


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """Iterate the data batch by batch"""
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int(data_size / batch_size) + 1

    for epoch in range(num_epochs):
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
           shuffled_data = data

        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]


if __name__ == '__main__':
    input_file = '../../data/train_data.jsons'
    load_data_and_labels(input_file)

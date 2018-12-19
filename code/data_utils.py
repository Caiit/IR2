from gensim.models import Word2Vec, KeyedVectors
import json
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import numpy as np
import pickle
import re
import string
import torch

global w2i
global embeddings
global use_gensim


def load_gensim(filename):
    """
    Loads Gensim model.
    """

    global use_gensim
    global embeddings

    use_gensim = True
    embeddings = KeyedVectors.load(filename, mmap="r")


def load_data(filename):
    """
    Loads the data. Data: chat, chat_id, documents (comments, fact_table, plot,
    review), imdb_id, labels, movie_name, spans
    """

    with open(filename, encoding="utf-8") as f:
        data = json.load(f)
    return data


def load_embeddings(file_path):
    """
    Loads the saved Pickle file.
    """

    global embeddings
    global use_gensim

    use_gensim = False

    with open(file_path, "rb") as f:
        embeddings = pickle.load(f)


def load_w2i(file_path):
    """
    Loads the saved Pickle file.
    """

    global w2i

    with open(file_path, "rb") as f:
        w2i = pickle.load(f)


def get_context(last_utterances):
    """
    Takes the last two utterances from a conversation and the current utterance
    to generate a context.
    """

    context = " ".join([str(string) for string in last_utterances])
    clean_context = clean_sentence(context)
    embedded_context = embed_sentence(context)
    return clean_context, embedded_context


def embed_sentence(sentence):
    """
    Embeds a sentence.
    """

    global w2i
    global embeddings
    global use_gensim

    sentence = clean_sentence(sentence)

    if use_gensim:
        embedding_dim = embeddings.vector_size
    else:
        embedding_dim = embeddings[0].shape[0]

    embedded_sentence = np.zeros((len(sentence), embedding_dim))

    for i, word in enumerate(sentence):
        # Embed sentence from from Gensim embeddings
        if use_gensim:
            try:
                embedded_sentence[i] = embeddings[word]
            except KeyError:
                continue
        # Embed sentence from Glove embeddings
        else:
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


def get_resources(document, resources, embedded_resources):
    """
    Obtains the resources in a document.
    """

    regex = re.compile("[@_!#$%^&*()<>?/\|}{~:]")

    if type(document) is list:
        for resource in document:
            # Check if the string isn't just special characters, e.g. {}
            if not regex.search(resource):
                get_resources(resource, resources, embedded_resources)
    elif type(document) is str:
        embedded_resource = embed_sentence(document)
        embedded_resources.append(embedded_resource)
        resources.append(clean_sentence(document))
    elif type(document) is dict:
        for key, value in document.items():
            get_resources(value, resources, embedded_resources)


def convert_to_words(complete_sent_emb, w2emb):
    '''
    Convert embeddings to wordsself.
    '''
    output_sentence = []
    for word in complete_sent_emb:
        emb_dists = [torch.norm(torch.Tensor(word) - torch.Tensor(embs)).item() for embs in list(w2emb.values())]
        index = np.argmin(emb_dists)
        output_sentence.append(list(w2emb.keys())[index])

    return " ".join(output_sentence)


def get_templates(filename):
    '''
    Get the templates from a pickle file.
    '''
    with open(filename, "rb") as f:
        return pickle.load(f)

def get_w2emb(filename):
    '''
    Get the w2emb from a pickle file.
    '''
    with open(filename, "rb") as f:
        return pickle.load(f)

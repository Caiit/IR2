import json
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import numpy as np
import pickle
import re
import string


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


def get_context(last_utterances, embeddings, w2i):
    """
    Takes the last two utterances from a conversation and the current utterance
    to generate a context.
    """

    context = " ".join([str(string) for string in last_utterances])
    clean_context = clean_sentence(context)
    embedded_context = embed_sentence(context, embeddings, w2i)
    return clean_context, embedded_context


def embed_sentence(sentence, embeddings, w2i):
    """
    Embeds a sentence.
    """

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


def get_resources(document, resources, embedded_resources, embeddings, w2i):
    """
    Obtains the resources in a document.
    """

    regex = re.compile("[@_!#$%^&*()<>?/\|}{~:]")

    if type(document) is list:
        for resource in document:
            # Check if the string isn't just special characters, e.g. {}
            if not regex.search(resource):
                get_resources(resource, resources, embedded_resources,
                              embeddings, w2i)
    elif type(document) is str:
        embedded_resource = embed_sentence(document, embeddings, w2i)
        embedded_resources.append(embedded_resource)
        resources.append(clean_sentence(document))
    elif type(document) is dict:
        for key, value in document.items():
            get_resources(value, resources, embedded_resources, embeddings, w2i)


def get_templates(filename):
    return load_pickle(filename)


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

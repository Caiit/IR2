import json
from gensim.models import Word2Vec
import matplotlib.pyplot as plt
from nltk.tokenize import word_tokenize
from sklearn.decomposition import PCA
import string


def read_data(filename):
    """
    Reads the JSON data.
    """

    with open(filename, encoding="utf-8") as f:
        data = json.load(f)
    return data


def get_gensim_model(data):
    """
    Retrieves and saves Word2Vec vectors that were created from input data.
    """

    TEXT = []

    for item in data:
        get_text(item, TEXT)
    TEXT = [item for item in TEXT if len(item) > 0]

    model = Word2Vec(TEXT, size=100, window=5, min_count=1, workers=4)
    return model


def get_text(data_entry, TEXT):
    """
    Retrieves the text in the data by looking in the following keys: chat,
    documents[comments], documents[fact_table], documents[plot],
    documents[review], movie_name and spans.
    """

    for key, value in data_entry.items():
        if key != "chat_id" and key != "imdb_id" and key != "labels":
            if isinstance(value, dict):
                get_text(value, TEXT)
            elif isinstance(value, list):
                # print([clean_text(str(item)) for item in value])
                TEXT.extend([clean_text(str(item)) for item in value])
            else:
                text = clean_text(str(value))
                TEXT.append(text)


def clean_text(text):
    """
    Cleans input text by tokenizing, removing punctuation and tokenizing.
    """

    # Lowercase
    text = text.lower()
    # Remove punctuation
    translator = str.maketrans('', '', string.punctuation)
    text = text.translate(translator)
    # Tokenize
    text = word_tokenize(text)
    return text


def visualize_embeddings(model):
    """
    Uses PCA to visualize the Word2Vec embeddings that were created.
    """

    X = model[model.wv.vocab]
    pca = PCA(n_components=2)
    result = pca.fit_transform(X)
    plt.scatter(result[:, 0], result[:, 1])
    words = list(model.wv.vocab)

    for i, word in enumerate(words):
        plt.annotate(word, xy=(result[i, 0], result[i, 1]))
    plt.show()

if __name__ == "__main__":
    data = read_data("../data/train_data.json")
    model = get_gensim_model(data)
    visualize_embeddings(model)

    # word_vectors = model.wv
    # word_vectors.save("../data/w2v_vectors.kv")

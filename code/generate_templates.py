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

def run3(data, resource_type):
    global embeddings
    global w2i

    max_length = 110
    resources = []
    for example in data:
        resources = get_resources2(example["documents"][resource_type], resources, max_length)

    print('now doing the embeddings....')
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
                    print(w)
                    embedded_sentence[i] = 0
            if len(embedded_sentence) > 0 and len(embedded_sentence) < max_length:
                avg_embedded.append(avg_embed_sentence(embedded_sentence))
                all_embedded.append(embedded_sentence)
                all_sents.append(sent_temp)

    print('now doing kmeans....')
    kmeans = KMeans(n_clusters=5)
    trying = kmeans.fit_predict(avg_embedded)
    clusters = kmeans.cluster_centers_

    print('now getting the 5 sentences.....')
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

    output_sentence =[]
    i = 0
    for senti in best_sents[-1]:
        i += 1
        emb_dists = [torch.norm(senti - torch.Tensor(embs)).item() for embs in list(w2emb.values())]
        index = np.argmin(emb_dists)
        output_sentence.append(list(w2emb.keys())[index])

    print(" ".join(output_sentence))

    print('gonna save now......')

    pkl_file_name = "embeddings_5_" + resource_type + ".pkl"
    text_file = open(pkl_file_name, "wb")
    pickle.dump(best_sents, text_file)
    text_file.close()

    print('saved....')

def run2(data, resource_type):
    global embeddings
    global w2i

    max_length = 1000
    resources = []
    for example in data:
        resources = get_resources2(example["documents"][resource_type], resources, max_length)

    all_embedded = []
    for res in resources:
        all_res = " ".join(res)
        if len(clean_text_by_sentences(all_res)) > 1:
            print(len(clean_text_by_sentences(all_res)))
            print(all_res)
            print(summarize(all_res))
            sent_temp = summarize(all_res)
            embedded_sentence = np.ones((len(sent_temp), embeddings[0].shape[0])) * len(w2i)
            for i, w in enumerate(sent_temp):
                if w2i.get(w):
                    index = w2i[w]
                    embedding = embeddings[index]
                    embedded_sentence[i] = embedding
                    w2emb[w] = embedding
                else:
                    embedded_sentence[i] = 0
            all_embedded.append(embedded_sentence)
            print('____________________________________________')

    text_file = open("embeddings_review.pkl", "wb")
    pickle.dump(all_embedded, text_file)
    text_file.close()
    print('Saved succesfully')

def clean_sentence2(sentence):
    """
    Cleans a sentence by lowercasing it and removing punctuation.
    """

    #sentence = sentence.lower()
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
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)

    # AANAPSSEN!!!!
    batchs = list(batches(resources, batch_size))

    for epoch in range(args.epochs):
        print('Epoch: ', epoch)
        avg_loss = 0
        for i, batch in enumerate(batchs):
            batch = np.array(batch)
            batch_input = torch.FloatTensor(batch[:-1])
            batch_target = torch.LongTensor(batch[1:])
            batch_preds = model.forward(batch_input)

            torch.nn.utils.clip_grad_norm(model.parameters(), max_norm=0.2)

            max_sent_size = len(batch_input[0])

            batch_target = batch_target.float()
            #total_sum = sum([torch.exp(torch.dot(batch_preds[i][w], batch_target[i][w])) for w in range(max_length) for i in range(batch_input.shape[0] - 1)])
            #loss = sum([-torch.dot(batch_preds[i][w], batch_target[i][w]) + torch.log(total_sum) for w in range(max_length) for i in range(batch_input.shape[0] - 1)])/(max_sent_size * max_length)
            loss = criterion(batch_preds, batch_target)
            avg_loss += loss
            #print(i, " loss ", loss)
            #print(total_sum, -torch.dot(batch_preds[0][1], batch_target[1][2]))
            a = list(model.parameters())[0].clone()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            b = list(model.parameters())[0].clone()
            #print(list(model.parameters())[0].grad)
            #print(torch.equal(a.data, b.data))
        avg_loss = avg_loss / i
        print('Loss: ', avg_loss)
        generate_samples(model, max_sent_size, 1)

def avg_embed_sentence(embedded_sentence):
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


def generate_samples(model, sequence_length, temperature):
    generated_samples = []
    for _ in range(5):
        init_word = '-'
        index = w2i[init_word]
        input_embedding = torch.Tensor(embeddings[index]).unsqueeze(0).unsqueeze(0)
        outputs = [input_embedding]
        # generate sentence of seq_length
        for i in range(5):
            output_embedding = model.forward(input_embedding)[:, -1,:].unsqueeze(0)
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
    #run(data, args.resource_type)
    #run2(data, args.resource_type)
    run3(data, args.resource_type)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("file", help="path to file of the dataset.")
    parser.add_argument("embeddings", help="path to file of the saved embeddings")
    parser.add_argument("w2i", help="path to file of the saved w2i")
    parser.add_argument("resource_type", type=str, help="which resource_type")
    parser.add_argument("epochs", type=int, help="how many epochs?")
    args = parser.parse_args()

    main(args)

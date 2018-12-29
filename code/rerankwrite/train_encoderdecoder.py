from encoder_decoder import DecoderRNN, EncoderRNN, CreateResponse
from saliency_model import SaliencyPrediction
from data_utils import load_data, load_pickle, get_context, embed_sentence, \
    clean_sentence, get_resources, get_templates
from retrieve import retrieve
from gensim.models import Word2Vec, KeyedVectors
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from rouge import Rouge
import matplotlib.pyplot as plt
import argparse
import numpy as np
from tqdm import tqdm
import pickle
import os


global SOS_token
global EOS_token
global device
global embedding_size


def convert_to_word(word, w2emb):
    """
    Converts an embedding into a word.
    """

    emb_dists = [torch.norm(word - torch.Tensor(embs)).item() for embs in
                 list(w2emb.values())]
    index = np.argmin(emb_dists)
    real_w = list(w2emb.keys())[index]
    return real_w


def load_saliency_model():
    """
    Loads the saliency model.
    """

    model = torch.load("../../models/rewrite/saliency_0.pt")
    model.eval()
    return model


def get_data(filename, data, embeddings, w2i, gensim_model, args):
    """
    Retrieves all data. Load it from a Pickle file if it exists, and create it
    otherwise.
    """

    global num_words

    if os.path.exists(filename):
        all_examples = load_pickle(filename)
    else:
        all_examples = []

        for example in tqdm(data):
            resources = []
            embedded_resources = []
            class_indices = []
            get_resources(example["documents"]["comments"], resources,
                          embedded_resources, embeddings, w2i)
            num_comments = len(resources)
            get_resources(example["documents"]["fact_table"], resources,
                          embedded_resources, embeddings, w2i)
            num_facts = len(resources) - num_comments
            get_resources(example["documents"]["plot"], resources,
                          embedded_resources, embeddings, w2i)
            num_plots = len(resources) - num_comments - num_facts
            get_resources(example["documents"]["review"], resources,
                          embedded_resources, embeddings, w2i)
            num_reviews = len(resources) - num_comments - num_facts - num_plots

            # Keep track of where each resource originated from.
            class_indices += [2]*num_comments
            class_indices += [3]*num_facts
            class_indices += [0]*num_plots
            class_indices += [1]*num_reviews

            chat = example["chat"]

            # Loop over each of the last three utterances in the chat (the context).
            for i in range(3, len(chat)-1):
                last_utterances = chat[i-3:i]
                response = chat[i+1]

                if len(response) > 0:
                    exp = []
                    embedded_utterances = [embed_sentence(utterance, embeddings,
                                           w2i) for utterance in
                                           last_utterances]
                    context, embedded_context = get_context(last_utterances,
                                                            embeddings, w2i)

                    # Retrieve: Takes context and resources. Uses Word Mover's Distance
                    # to obtain relevant resource candidates.
                    similarities = retrieve(context, resources, gensim_model)

                    padd_resource = embedded_resources[np.argmax(similarities)][-args.max_length:]
                    padd_resource = np.pad(padd_resource, ((0, args.max_length -
                                           len(padd_resource)), (0, 0)),
                                           "constant",
                                           constant_values=(num_words))

                    exp.append(padd_resource)
                    #exp.append(embed_sentence(chat[i+1], embeddings, w2i))
                    exp.append(clean_sentence(chat[i+1]))
                    all_examples.append(tuple(exp))
        save_data(filename, all_examples)
    return all_examples


def save_data(filename, data):
    """
    Saves data to Pickle file.
    """

    with open(filename, "wb") as f:
        pickle.dump(data, f)


def get_current_epoch(filename, epoch):
    """
    Calculates the current epoch so that the model is saved with the correct
    filename.
    """

    last_epoch = filename.split("_")[-1]
    last_epoch = int(last_epoch.split(".")[0])
    current_epoch = last_epoch + epoch
    return current_epoch


def get_all(args):
    """
    Gets the training and test data, and templates.
    """

    global embedding_size
    global num_words

    print("Load data...")
    data_train = load_data(args.folder + "/train_data.json")
    data_test = load_data(args.folder + "/dev_data.json")
    data_train = data_train
    data_test = data_test
    print(len(data_train))
    embeddings = load_pickle(args.embeddings)
    embedding_size = len(embeddings[0])
    w2i = load_pickle(args.w2i)
    num_words = len(w2i)
    w2emb = load_pickle(args.w2emb)
    templates_emb  = get_templates("../../data/templates.pkl")
    gensim_model = KeyedVectors.load(args.word2vec, mmap='r')

    print("Do the templates...")
    templates_emb = [y for x in templates_emb for y in x]
    cut_templates = [temp[-args.max_length:] for temp in templates_emb]
    templates_emb = [np.pad(temp1, ((0, args.max_length-len(temp1)), (0, 0)),
    "constant", constant_values=(num_words)) for temp1 in
    cut_templates]
    templates_emb = torch.Tensor(templates_emb)

    print("Go through training data...")
    # In the form of (start sent, resource, target)
    all_training_data = get_data(args.saved_train, data_train, embeddings, w2i,
    gensim_model, args)
    all_test_data = get_data(args.saved_test, data_test, embeddings, w2i,
    gensim_model, args)
    return all_training_data, all_test_data, templates_emb, w2emb, w2i


def save_model(model, encoder_optim, decoder_optim, epoch):
    """
    Saves the model so that we can continue training.
    """

    filename = "../../models/rewrite/model_encoder_" + str(epoch) + ".pt"

    checkpoint = {
        "epoch": epoch,
        "state_dict": model.state_dict(),
        "encoder_optimizer": encoder_optim.state_dict(),
        "decoder_optimizer": decoder_optim.state_dict()
    }
    torch.save(checkpoint, filename)


def run(args):
    """
    Run model by training and testing it.
    """

    global SOS_token
    global EOS_token
    global device
    global embedding_size
    global num_words

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    training_data, test_data, templates, w2emb, w2i = get_all(args)

    print("Now load the model...")
    hidden_size = 128
    rewrite_model = CreateResponse(embedding_size, 128, embedding_size, 0.3,
                                   args.max_length, device).to(device)
    saliency_model = load_saliency_model().to(device)
    # encoder_optimizer = optim.SGD(rewrite_model.encoder.parameters(), lr=0.01,
    #                               momentum=0.9)
    # decoder_optimizer = optim.SGD(rewrite_model.decoder.parameters(), lr=0.01,
    #                               momentum=0.9)

    encoder_optimizer = optim.Adam(rewrite_model.encoder.parameters())
    decoder_optimizer = optim.Adam(rewrite_model.decoder.parameters())

    SOS_token = torch.Tensor([i for i in
                              range(embedding_size)]).unsqueeze(0).to(device)
    EOS_token = torch.Tensor([i+1 for i in
                              range(embedding_size)]).unsqueeze(0).to(device)
    w2emb["SOS_token"] = SOS_token.cpu()
    w2emb["EOS_token"] = EOS_token.cpu()

    if args.saved_model:
        checkpoint = torch.load(args.saved_model)
        rewrite_model.load_state_dict(checkpoint["state_dict"])
        encoder_optimizer.load_state_dict(checkpoint["encoder_optimizer"])
        decoder_optimizer.load_state_dict(checkpoint["decoder_optimizer"])

    if args.evaluate:
        test(rewrite_model, saliency_model, test_data, templates, w2emb, w2i)
    else:
        train(rewrite_model, saliency_model, encoder_optimizer,
        decoder_optimizer, training_data, templates, w2emb, w2i)
        test(rewrite_model, saliency_model, test_data, templates, w2emb, w2i)

def train(rewrite_model, saliency_model, encoder_optim, decoder_optim,
          training_data, templates, w2emb, w2i):
    """
    Train model on training data.
    """

    global SOS_token
    global EOS_token
    global num_words

    print(len(w2i), len(w2emb))

    #emb2i = {emb: i for w in w2emb.keys()}

    #loss_func = nn.MSELoss()
    loss_func = nn.CrossEntropyLoss()
    all_temps = torch.Tensor(templates).to(device)

    rewrite_model.train()

    for epoch in range(100):
        print("Epoch: " + str(epoch))
        np.random.shuffle(training_data)
        print("Now do the training...")
        total_loss = 0

        for ex in tqdm(training_data):
            resource = ex[0]
            target = []
            target_embs = []
            for w in ex[1]:
                if w in w2i.keys():
                    target.append(w2i[w])
                else:
                    target.append(0)
                if w in w2emb.keys():
                    target_embs.append(w2emb[w])
                else:
                    target_embs.append([0]*100)


            #target = [w2i[w] for w in ex[1] if w in w2i else 0]
            target = torch.Tensor(target).to(device)

            padd_resource = resource[-args.max_length:]
            padd_resource = np.pad(padd_resource, ((0, args.max_length -
                                   len(padd_resource)), (0, 0)), "constant",
                                   constant_values=(num_words))

            all_res = torch.Tensor(padd_resource).unsqueeze(0).repeat(20,
                                   1, 1).to(device)
            size_inp = all_res.size()
            x1 = all_res.reshape(size_inp[0], size_inp[1]*size_inp[2])
            x2 = all_temps.reshape(size_inp[0], size_inp[1]*size_inp[2])
            scores = saliency_model(x1, x2)
            best_template = all_temps[torch.argmax(scores)].squeeze(0)


            # resource = torch.cat((SOS_token, all_res[0], EOS_token))
            # template = torch.cat((SOS_token, best_template, EOS_token))
            # final_input = torch.cat((resource, template), dim=1).unsqueeze(0)

            final_input = torch.cat((SOS_token, all_res[0], EOS_token,
                                     SOS_token, best_template,
                                     EOS_token)).unsqueeze(0)

            encoder_hidden = rewrite_model.encoder.initHidden()
            encoder_optim.zero_grad()
            decoder_optim.zero_grad()
            input_length = final_input.size(1)
            target_length = target.size(0)

            encoder_outputs = \
                torch.zeros(args.max_length*2 + 4,
                            rewrite_model.encoder.hidden_size).to(device)
            loss = 0

            for ei in range(input_length):
                encoder_output, encoder_hidden = \
                    rewrite_model.encoder(final_input[:, ei].unsqueeze(0),
                                          encoder_hidden)
                encoder_outputs[ei] = encoder_output[0, 0]


            decoder_input = SOS_token.unsqueeze(0)
            decoder_hidden = encoder_hidden

            print(target)

            # Teacher forcing: Feed the target as the next input
            for di in range(target_length-1):
                decoder_output, decoder_hidden, decoder_attention = \
                    rewrite_model.decoder(decoder_input, decoder_hidden, encoder_outputs)
                new_tar = target[di+1].unsqueeze(0)
                #print(w2i[word], word)
                #loss += loss_func(decoder_output, new_tar)
                loss += loss_func(decoder_output, new_tar.long())
                decoder_input = torch.Tensor(target_embs[di+1]).unsqueeze(0).unsqueeze(0).to(device)

            total_loss += loss
            print(loss)

            loss.backward()

            encoder_optim.step()
            decoder_optim.step()
        print("Total_loss: " + str(total_loss.item()))

        if args.saved_model:
            real_epoch = get_current_epoch(args.saved_model, epoch)
        else:
            real_epoch = epoch
        save_model(rewrite_model, encoder_optim, decoder_optim, real_epoch)


# def train(rewrite_model, saliency_model, encoder_optim, decoder_optim,
#           training_data, templates):
#     """
#     Train model on training data.
#     """
#
#     global SOS_token
#     global EOS_token
#     global num_words
#
#     loss_func = nn.MSELoss()
#     all_temps = torch.Tensor(templates).to(device)
#
#     rewrite_model.train()
#
#     for epoch in range(100):
#         print("Epoch: " + str(epoch))
#         np.random.shuffle(training_data)
#         print("Now do the training...")
#         total_loss = 0
#
#         for ex in tqdm(training_data):
#             resource = ex[0]
#             target = torch.Tensor(ex[1]).to(device)
#
#             padd_resource = resource[-args.max_length:]
#             padd_resource = np.pad(padd_resource, ((0, args.max_length -
#                                    len(padd_resource)), (0, 0)), "constant",
#                                    constant_values=(num_words))
#
#             all_res = torch.Tensor(padd_resource).unsqueeze(0).repeat(20,
#                                    1, 1).to(device)
#             size_inp = all_res.size()
#             x1 = all_res.reshape(size_inp[0], size_inp[1]*size_inp[2])
#             x2 = all_temps.reshape(size_inp[0], size_inp[1]*size_inp[2])
#             scores = saliency_model(x1, x2)
#             best_template = all_temps[torch.argmax(scores)].squeeze(0)
#
#             final_input = torch.cat((SOS_token, all_res[0], EOS_token,
#                                      SOS_token, best_template,
#                                      EOS_token)).unsqueeze(0)
#
#             encoder_hidden = rewrite_model.encoder.initHidden()
#             encoder_optim.zero_grad()
#             decoder_optim.zero_grad()
#             input_length = final_input.size(1)
#             target_length = target.size(0)
#
#             encoder_outputs = \
#                 torch.zeros(args.max_length*2 + 4,
#                             rewrite_model.encoder.hidden_size).to(device)
#             loss = 0
#
#
#             for ei in range(input_length):
#                 encoder_output, encoder_hidden = \
#                     rewrite_model.encoder(final_input[:, ei].unsqueeze(0),
#                                           encoder_hidden, length)
#                 encoder_outputs[ei] = encoder_output[0, 0]
#
#
#             decoder_input = SOS_token.unsqueeze(0)
#             decoder_hidden = encoder_hidden
#
#             # Teacher forcing: Feed the target as the next input
#             for di in range(target_length):
#                 decoder_output, decoder_hidden = \
#                     rewrite_model.decoder(decoder_input, decoder_hidden)
#                 new_tar = target[di].unsqueeze(0)
#                 loss += loss_func(decoder_output, new_tar)
#                 decoder_input = new_tar.unsqueeze(0)
#
#             total_loss += loss
#             loss.backward()
#
#             encoder_optim.step()
#             decoder_optim.step()
#         print("Total_loss: " + str(total_loss.item()))
#
#         if args.saved_model:
#             real_epoch = get_current_epoch(args.saved_model, epoch)
#         else:
#             real_epoch = epoch
#         save_model(rewrite_model, encoder_optim, decoder_optim, real_epoch)


# def test(rewrite_model, saliency_model, test_data, templates, w2emb):
#     """
#     Test model on evaluation data.
#     """
#
#     global SOS_token
#     global EOS_token
#     global num_words
#
#     all_temps = torch.Tensor(templates).to(device)
#
#     rewrite_model.eval()
#     print("Now do the testing.. ")
#     for ex in tqdm(test_data):
#         with torch.no_grad():
#             resource = ex[0]
#             target = torch.Tensor(ex[1]).to(device)
#
#             padd_resource = resource[-args.max_length:]
#             padd_resource = np.pad(padd_resource, ((0, args.max_length -
#                                    len(padd_resource)), (0, 0)), "constant",
#                                    constant_values=(num_words))
#
#             all_res = torch.Tensor(padd_resource).unsqueeze(0).repeat(20, 1,
#                                    1).to(device)
#             size_inp = all_res.size()
#             x1 = all_res.reshape(size_inp[0], size_inp[1]*size_inp[2])
#             x2 = all_temps.reshape(size_inp[0], size_inp[1]*size_inp[2])
#             scores = saliency_model(x1, x2)
#             best_template = all_temps[torch.argmax(scores)].squeeze(0)
#
#             final_input = torch.cat((SOS_token, all_res[0], EOS_token,
#                                      SOS_token, best_template,
#                                      EOS_token)).unsqueeze(0)
#             encoder_hidden = rewrite_model.encoder.initHidden()
#             input_length = final_input.size(1)
#             target_length = target.size()
#
#             encoder_outputs = torch.zeros(args.max_length*2 + 4,
#                                           rewrite_model.encoder.hidden_size)
#             loss = 0
#
#             for ei in range(input_length):
#                 encoder_output, encoder_hidden = \
#                     rewrite_model.encoder(final_input[:, ei].unsqueeze(0),
#                                           encoder_hidden)
#                 encoder_outputs[ei] = encoder_output[0, 0]
#
#             decoder_input = SOS_token.unsqueeze(0)
#             decoder_hidden = encoder_hidden
#
#             decoded_words = []
#             for di in range(args.max_length):
#                 decoder_output, decoder_hidden = \
#                     rewrite_model.decoder(decoder_input, decoder_hidden)
#                 word = convert_to_word(decoder_output.cpu(), w2emb)
#
#                 if word == "EOS_token":
#                     decoded_words.append("<EOS>")
#                     break
#                 else:
#                     decoded_words.append(word)
#                 decoder_input = decoder_output.unsqueeze(0)
#
#             print(decoded_words)

def test(rewrite_model, saliency_model, test_data, templates, w2emb, w2i):
    """
    Test model on evaluation data.
    """

    global SOS_token
    global EOS_token
    global num_words

    all_temps = torch.Tensor(templates).to(device)

    rewrite_model.eval()
    print("Now do the testing.. ")
    for ex in tqdm(test_data):
        with torch.no_grad():
            resource = ex[0]
            target = []
            target_embs = []
            for w in ex[1]:
                if w in w2i.keys():
                    target.append(w2i[w])
                else:
                    target.append(0)
                if w in w2emb.keys():
                    target_embs.append(w2emb[w])
                else:
                    target_embs.append([0]*100)


            #target = [w2i[w] for w in ex[1] if w in w2i else 0]
            target = torch.Tensor(target).to(device)

            padd_resource = resource[-args.max_length:]
            padd_resource = np.pad(padd_resource, ((0, args.max_length -
                                   len(padd_resource)), (0, 0)), "constant",
                                   constant_values=(num_words))

            all_res = torch.Tensor(padd_resource).unsqueeze(0).repeat(20, 1,
                                   1).to(device)
            size_inp = all_res.size()
            x1 = all_res.reshape(size_inp[0], size_inp[1]*size_inp[2])
            x2 = all_temps.reshape(size_inp[0], size_inp[1]*size_inp[2])
            scores = saliency_model(x1, x2)
            best_template = all_temps[torch.argmax(scores)].squeeze(0)

            final_input = torch.cat((SOS_token, all_res[0], EOS_token,
                                     SOS_token, best_template,
                                     EOS_token)).unsqueeze(0)

            # resource = torch.cat((SOS_token, all_res[0], EOS_token))
            # template = torch.cat((SOS_token, best_template, EOS_token))
            # final_input = torch.cat((resource, template), dim=1).unsqueeze(0)

            encoder_hidden = rewrite_model.encoder.initHidden()
            input_length = final_input.size(1)
            target_length = target.size()

            encoder_outputs = torch.zeros(args.max_length*2 + 4,
                                          rewrite_model.encoder.hidden_size).to(device)
            loss = 0

            for ei in range(input_length):
                encoder_output, encoder_hidden = \
                    rewrite_model.encoder(final_input[:, ei].unsqueeze(0),
                                          encoder_hidden)
                encoder_outputs[ei] = encoder_output[0, 0]

            decoder_input = SOS_token.unsqueeze(0)
            decoder_hidden = encoder_hidden

            decoder_attentions = torch.zeros((args.max_length*2) + 4, (args.max_length*2) + 4).to(device)

            decoded_words = []
            for di in range(args.max_length):
                decoder_output, decoder_hidden, decoder_attention = \
                    rewrite_model.decoder(decoder_input, decoder_hidden, encoder_outputs)
                decoder_attentions[di] = decoder_attention.data
                #word = convert_to_word(decoder_output.cpu(), w2emb)
                decoder_output = F.softmax(decoder_output)
                _, max_ind = torch.max(decoder_output, 1)
                word = list(w2i.keys())[list(w2i.values()).index(max_ind)]

                if word == "EOS_token":
                    decoded_words.append("<EOS>")
                    break
                else:
                    decoded_words.append(word)

                if word in w2emb.keys():
                    decoder_input = torch.Tensor(w2emb[word]).unsqueeze(0).unsqueeze(0).to(device)
                else:
                    decoder_input = torch.Tensor([0]*100).unsqueeze(0).unsqueeze(0).to(device)
                #decoder_input = torch.Tensor(w2emb[word]).unsqueeze(0).unsqueeze(0)

            print(decoded_words)
            #showAttention(ex, decoded_words, decoder_attentions)


def showAttention(input_sentence, output_words, attentions):
    # Set up figure with colorbar
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(attentions.numpy(), cmap='bone')
    fig.colorbar(cax)

    print(input_sentence)
    # Set up axes
    ax.set_xticklabels([''] + input_sentence.split(' ') +
                       ['<EOS>'], rotation=90)
    ax.set_yticklabels([''] + output_words)

    # Show label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", help="path to file of the dataset.", default="../../data")
    parser.add_argument("--embeddings", help="path to file of the saved embeddings", default="../../embeddings/glove_50d.pkl")
    parser.add_argument("--word2vec", help="path to file of the word2vec embeddings.", default="../../embeddings/w2v_vectors.kv")
    parser.add_argument("--w2i", help="path to file of the saved w2i", default="../../embeddings/w2i.pkl")
    parser.add_argument("--w2emb", help="path to the file of the saved w2emb", default="../../embeddings/w2emb.pkl")
    parser.add_argument("--max_length", help="max length of sentences", default=110)
    parser.add_argument("--saved_train", help="where to save the training data", default="../../data/rewrite_train.pkl")
    parser.add_argument("--saved_test", help="where to save the test data", default="../../data/rewrite_test.pkl")
    parser.add_argument("--saved_model", help="where the model was saved")#, default="../../models/rewrite/model_encoder_10.pt")
    parser.add_argument("--evaluate", help="only evaluate, do not train", default=False, type=bool)

    args = parser.parse_args()
    run(args)

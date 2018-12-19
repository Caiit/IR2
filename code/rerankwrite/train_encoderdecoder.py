# torch.save(model.state_dict(), PATH)
# torch.save(model, PATH)

from encoder_decoder import DecoderRNN, EncoderRNN, CreateResponse
from saliency_model import SaliencyPrediction
from data_utils import load_data, load_pickle, get_context, embed_sentence, \
    clean_sentence, get_resources, get_templates
from retrieve import retrieve
from gensim.models import Word2Vec, KeyedVectors
import torch
import torch.nn as nn
import torch.optim as optim
from rouge import Rouge
import matplotlib.pyplot as plt
import argparse
import numpy as np
from tqdm import tqdm

def convert_to_word(word, w2emb):
    emb_dists = [torch.norm(word - torch.Tensor(embs)).item() for embs in list(w2emb.values())]
    index = np.argmin(emb_dists)
    real_w = list(w2emb.keys())[index]

    return real_w

def load_saliency_model():
    model = torch.load("../../models/rewrite/saliency.pt")
    model.eval()
    return model

# def loss_func(input, target):
#     b_size = input.shape[0]
#     total_sum = sum([torch.exp(torch.dot(input[i], target[i])) for i in range(b_size - 1)])
#     loss = sum([-torch.dot(input[i], target[i]) + torch.log(total_sum) for i in range(b_size - 1)])/input.shape[1]
#     return loss

def get_data(data, embeddings, w2i, gensim_model, args):
    all_examples = []
    for example in tqdm(data[:50]):
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
                embedded_utterances = [embed_sentence(utterance, embeddings, w2i) for utterance in
                                       last_utterances]
                context, embedded_context = get_context(last_utterances, embeddings, w2i)

                # Retrieve: Takes context and resources. Uses Word Mover's Distance
                # to obtain relevant resource candidates.
                similarities = retrieve(context, resources, gensim_model)

                padd_resource = embedded_resources[np.argmax(similarities)][-args.max_length:]
                padd_resource = np.pad(padd_resource, ((0, args.max_length - len(padd_resource)), (0, 0)), "constant",
                                    constant_values=(len(w2i)))

                exp.append(padd_resource)
                exp.append(embed_sentence(chat[i+1], embeddings, w2i))
                all_examples.append(tuple(exp))
    return all_examples

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Load data...")
    data_train = load_data(args.folder + "/train_data.json")
    data_test = load_data(args.folder + "/dev_data.json")
    embeddings = load_pickle(args.embeddings)
    w2i = load_pickle(args.w2i)
    w2emb = load_pickle(args.w2emb)
    templates_emb  = get_templates("../../data/templates.pkl")
    gensim_model = KeyedVectors.load(args.word2vec, mmap='r')

    print("Do the templates...")
    flattened_templates_emb = [y for x in templates_emb for y in x]
    cut_templates = [temp[-args.max_length:] for temp in flattened_templates_emb]
    flattened_templates_emb_padded = [np.pad(temp1, ((0, args.max_length - len(temp1)), (0, 0)),
                                            "constant", constant_values=(len(w2i))) for temp1 in cut_templates]
    templates_padd = torch.Tensor(flattened_templates_emb_padded)

    print("Now load the model...")
    emb_size = len(embeddings[0])
    hidden_size = 128
    model = CreateResponse(emb_size, 128, emb_size, 0.3, args.max_length, device).to(args.use_gpu)
    model_sal = load_saliency_model().to(args.use_gpu)
    loss_func = nn.MSELoss()
    #loss_func = nn.NLLLoss()
    #loss_func = nn.BCELoss()
    #optimizer = optim.Adam(model.parameters())
    encoder_optimizer = optim.SGD(model.encoder.parameters(), lr = 0.001, momentum=0.9)
    decoder_optimizer = optim.SGD(model.decoder.parameters(), lr=0.001, momentum=0.9)



    print("Go through training data...")
    # In the form of (start sent, resource, target)
    all_training_data = get_data(data_train, embeddings, w2i, gensim_model, args)
    all_test_data = get_data(data_test, embeddings, w2i, gensim_model, args)
    print(len(all_training_data))
    print(len(all_test_data))


    SOS_token = torch.Tensor([i for i in range(emb_size)]).unsqueeze(0).to(args.use_gpu)
    EOS_token = torch.Tensor([i+1 for i in range(emb_size)]).unsqueeze(0).to(args.use_gpu)
    w2emb["SOS_token"] = SOS_token.cpu()
    w2emb["EOS_token"] = EOS_token.cpu()

    model.train()
    for epoch in range(5):
        print("Epoch: " + str(epoch))
        np.random.shuffle(all_training_data)
        print("Now do the training...")
        total_loss = 0
        for ex in all_training_data:
            resource = ex[0]
            target = torch.Tensor(ex[1]).to(args.use_gpu)

            padd_resource = resource[-args.max_length:]
            padd_resource = np.pad(padd_resource, ((0, args.max_length - len(padd_resource)), (0, 0)), "constant",
                                constant_values=(len(w2i)))

            all_temps = torch.Tensor(flattened_templates_emb_padded).to(args.use_gpu)
            all_res = torch.Tensor(padd_resource).unsqueeze(0).repeat(20, 1, 1).to(args.use_gpu)
            size_inp = all_res.size()
            x1 = all_res.reshape(size_inp[0], size_inp[1]*size_inp[2])
            x2 = all_temps.reshape(size_inp[0], size_inp[1]*size_inp[2])
            scores = model_sal.forward(x1, x2)
            best_template = all_temps[torch.argmax(scores)].squeeze(0)

            final_input = torch.cat((SOS_token, all_res[0], EOS_token, SOS_token, best_template, EOS_token)).unsqueeze(0)
            encoder_hidden = model.encoder.initHidden()
            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()
            input_length = final_input.size(0)
            target_length = target.size(0)

            encoder_outputs = torch.zeros(args.max_length*2 + 4, model.encoder.hidden_size).to(args.use_gpu)
            loss = 0

            for ei in range(input_length):
                encoder_output, encoder_hidden = model.encoder(final_input[ei].unsqueeze(0), encoder_hidden)
                encoder_outputs[ei] = encoder_output[0, 0]

            decoder_input = SOS_token.unsqueeze(0)

            decoder_hidden = encoder_hidden

            # Teacher forcing: Feed the target as the next input
            for di in range(target_length):
                decoder_output, decoder_hidden = model.decoder(decoder_input, decoder_hidden)
                new_tar = target[di].unsqueeze(0)
                loss += loss_func(decoder_output, new_tar)
                decoder_input = new_tar.unsqueeze(0)

            total_loss += loss
            loss.backward()

            encoder_optimizer.step()
            decoder_optimizer.step()
        print("Total_loss: " + str(total_loss.item()))
        torch.save(model, args.model_save)

    model.eval()
    print("Now do the testing.. ")
    for ex in all_test_data:
        with torch.no_grad():
            resource = ex[0]
            target = torch.Tensor(ex[1]).to(args.use_gpu)

            padd_resource = resource[-args.max_length:]
            padd_resource = np.pad(padd_resource, ((0, args.max_length - len(padd_resource)), (0, 0)), "constant",
                                constant_values=(len(w2i)))

            all_temps = torch.Tensor(flattened_templates_emb_padded).to(args.use_gpu)
            all_res = torch.Tensor(padd_resource).unsqueeze(0).repeat(20, 1, 1).to(args.use_gpu)
            size_inp = all_res.size()
            x1 = all_res.reshape(size_inp[0], size_inp[1]*size_inp[2])
            x2 = all_temps.reshape(size_inp[0], size_inp[1]*size_inp[2])
            scores = model_sal.forward(x1, x2)
            best_template = all_temps[torch.argmax(scores)].squeeze(0)

            final_input = torch.cat((SOS_token, all_res[0], EOS_token, SOS_token, best_template, EOS_token)).unsqueeze(0)
            encoder_hidden = model.encoder.initHidden()
            input_length = final_input.size(0)
            target_length = target.size(0)

            encoder_outputs = torch.zeros(args.max_length*2 + 4, model.encoder.hidden_size)
            loss = 0

            for ei in range(input_length):
                encoder_output, encoder_hidden = model.encoder(final_input[ei].unsqueeze(0), encoder_hidden)
                encoder_outputs[ei] = encoder_output[0, 0]

            decoder_input = SOS_token.unsqueeze(0)

            decoder_hidden = encoder_hidden

            decoded_words = []
            for di in range(args.max_length):
                decoder_output, decoder_hidden = model.decoder(decoder_input, decoder_hidden)
                word = convert_to_word(decoder_output.cpu(), w2emb)
                if word == "EOS_token":
                    decoded_words.append("<EOS>")
                    break
                else:
                    decoded_words.append(word)
                decoder_input = decoder_output.unsqueeze(0)

            print(decoded_words)
    return




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", help="path to file of the dataset.", default="../../data")
    parser.add_argument("--embeddings", help="path to file of the saved embeddings", default="../../embeddings/glove_50d.pkl")
    parser.add_argument("--word2vec", help="path to file of the word2vec embeddings.", default="../../embeddings/w2v_vectors.kv")
    parser.add_argument("--w2i", help="path to file of the saved w2i", default="../../embeddings/w2i.pkl")
    parser.add_argument("--w2emb", help="path to the file of the saved w2emb", default="../../embeddings/w2emb.pkl")
    parser.add_argument("--max_length", help="max length of sentences", default=110)
    parser.add_argument("--use_gpu", help="whether to use gpu or not", default="cuda:0") # or use "cpu"
    parser.add_argument("--model_save", help="where to store the model", default="../../models/rewrite/model_encoder.pt")

    args = parser.parse_args()
    train(args)

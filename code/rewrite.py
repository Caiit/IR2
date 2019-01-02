from rerankwrite.encoder_decoder import DecoderRNN, EncoderRNN, CreateResponse
from rerankwrite.saliency_model import SaliencyPrediction
from data_utils import embed_sentence
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

def convert_to_word(word, w2emb):
    emb_dists = [torch.norm(word - torch.Tensor(embs)).item() for embs in list(w2emb.values())]
    index = np.argmin(emb_dists)
    real_w = list(w2emb.keys())[index]

    return real_w

def load_saliency_model(model_folder):
    model = torch.load(model_folder + "/saliency.pt")
    model.eval()
    return model

def load_encoder_decoder_model(model_folder):
    model = torch.load(model_folder + "/model_encoder_decoder.pt")
    model.eval()
    return model

class Rewrite():
    def __init__(self, model_folder, embeddings, w2i, SOS_token, EOS_token, templates, w2emb, device, max_length=110):
        #print("Please replace me with the real class")
        ## TODO: load model
        self.saliencymodel = load_saliency_model(model_folder)
        self.encoder_decoder = load_encoder_decoder_model(model_folder)
        self.embeddings = embeddings
        self.w2i = w2i
        self.embedding_size = len(embeddings[w2i.values()[0]])
        self.SOS = SOS_token
        self.EOS = EOS_token
        self.templates = templates
        self.max_length = max_length
        self.w2emb = w2emb
        self.device = device

    def rerank(self, resource, class_category):
        templates_best_class = self.templates[class_category]
        padd_resource = resource[-args.max_length:]
        padd_resource = np.pad(padd_resource, ((0, self.max_length - len(padd_resource)), (0, 0)), "constant",
                            constant_values=(len(self.w2i)))

        all_temps = templates_best_class.to(self.device)
        all_res = torch.Tensor(padd_resource).unsqueeze(0).repeat(5, 1, 1).to(self.device)
        scores = self.saliencymodel.forward(all_res, all_temps)
        temp = templates_best_class[torch.argmax(scores)]
        return padd_resource, temp

    def rewrite(self, best_response, best_template):
        # Inputs are both embeddings

        final_input = torch.cat((self.SOS_token, best_response, self.EOS_token,
                                 self.SOS_token, best_template,
                                 self.EOS_token)).unsqueeze(0)

        encoder_hidden = self.encoder_decoder.encoder.initHidden()
        input_length = final_input.size(1)

        encoder_outputs = \
            torch.zeros(self.max_length*2 + 4,
                        self.encoder_decoder.encoder.hidden_size).to(self.device)
        loss = 0

        for ei in range(input_length):
            encoder_output, encoder_hidden = \
                self.encoder_decoder.encoder(final_input[:, ei].unsqueeze(0),
                                      encoder_hidden)
            encoder_outputs[ei] = encoder_output[0, 0]

        decoder_input = self.SOS_token.unsqueeze(0)
        decoder_hidden = encoder_hidden

        decoder_attentions = torch.zeros((self.max_length*2) + 4, (self.max_length*2) + 4).to(device)

        decoded_words = []
        for di in range(self.max_length):
            decoder_output, decoder_hidden, decoder_attention = \
                self.encoder_decoder.decoder(decoder_input, decoder_hidden, encoder_outputs)

            decoder_output = F.softmax(decoder_output, 1)
            _, max_ind = torch.max(decoder_output, 1)
            word = list(self.w2i.keys())[list(self.w2i.values()).index(max_ind)]

            if word == "EOS_token":
                decoded_words.append("<EOS>")
                break
            else:
                decoded_words.append(word)

            if word in self.w2emb.keys():
                decoder_input = torch.Tensor(self.w2emb[word]).unsqueeze(0).unsqueeze(0).to(device)
            else:
                decoder_input = torch.Tensor([0]*100).unsqueeze(0).unsqueeze(0).to(device)
        return " ".join(decoded_words)

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

from rerankwrite.encoder_decoder import DecoderRNN, EncoderRNN, CreateResponse
from rerankwrite.saliency_model import SaliencyPrediction
from data_utils import embed_sentence
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

def convert_to_word(word, w2emb):
    """
    Converts an embedding back into a word.
    """

    emb_dists = [torch.norm(word - torch.Tensor(embs)).item() for
                 embs in list(w2emb.values())]
    index = np.argmin(emb_dists)
    real_w = list(w2emb.keys())[index]
    return real_w


def load_saliency_model(filename, device):
    """
    Loads the trained saliency model.
    """

    model = torch.load(filename, map_location=device)
    model.eval()
    return model


def load_encoder_decoder_model(model_folder, embedding_size, max_length,
                               device):
    """
    Loads the trained Rewrite (Encoder-Decoder) model.
    """

    model = CreateResponse(embedding_size, 128, embedding_size, 0.3, max_length,
                           device).to(device)
    checkpoint = torch.load(model_folder + "model_encoder.pt",
                            map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model


class Rewrite():
    def __init__(self, saved_saliency, saved_rewrite, embeddings, w2i,
                 SOS_token, EOS_token, templates, w2emb, device,
                 max_length=110):
        self.embedding_size = len(embeddings[list(w2i.values())[0]])
        self.saliencymodel = load_saliency_model(saved_saliency,
                                                 device).to(device)
        self.encoder_decoder = load_encoder_decoder_model(saved_rewrite,
                                                          self.embedding_size,
                                                          max_length,
                                                          device).to(device)
        self.embeddings = embeddings
        self.w2i = w2i
        self.SOS = SOS_token
        self.EOS = EOS_token
        self.templates = templates
        self.max_length = max_length
        self.w2emb = w2emb
        self.device = device


    def rerank(self, resource, class_category):
        """
        Reranks the templates according to the best resource that was found.
        """

        templates_best_class = self.templates[class_category]
        padd_resource = resource[-self.max_length:]
        padd_resource = np.pad(padd_resource, ((0, self.max_length -
                               len(padd_resource)), (0, 0)), "constant",
                               constant_values=(len(self.w2i)))

        all_temps = templates_best_class.to(self.device)
        all_res = torch.Tensor(padd_resource).unsqueeze(0).repeat(5, 1, 1).to(self.device)
        size_inp = all_res.size()
        x1 = all_res.reshape(size_inp[0], size_inp[1]*size_inp[2])
        x2 = all_temps.reshape(size_inp[0], size_inp[1]*size_inp[2])
        scores = self.saliencymodel(x1, x2)
        temp = templates_best_class[torch.argmax(scores)]
        return padd_resource, temp


    def rewrite(self, best_resource, best_template):
        """
        Rewrites the best resource and the best template into a single response.
        """

        best_resource = torch.Tensor(best_resource).to(self.device)
        best_template = best_template.to(self.device)

        input = torch.cat((self.SOS, best_resource, self.EOS,
                           self.SOS, best_template, self.EOS)).unsqueeze(0)
        encoder_hidden = self.encoder_decoder.encoder.initHidden()
        input_length = input.size(1)

        encoder_outputs = torch.zeros(self.max_length*2 + 4,
                                      self.encoder_decoder.encoder.hidden_size)
        encoder_outputs = encoder_outputs.to(self.device)

        for ei in range(input_length):
            encoder_output, encoder_hidden = \
                self.encoder_decoder.encoder(input[:, ei].unsqueeze(0),
                                             encoder_hidden)
            encoder_outputs[ei] = encoder_output[0, 0]

        decoder_input = self.SOS.unsqueeze(0)
        decoder_hidden = encoder_hidden
        decoded_words = []

        for di in range(self.max_length):
            decoder_output, decoder_hidden, decoder_attention = \
                self.encoder_decoder.decoder(decoder_input, decoder_hidden,
                                             encoder_outputs)
            decoder_output = F.softmax(decoder_output, 1)
            _, max_ind = torch.max(decoder_output, 1)
            word = list(self.w2i.keys())[list(self.w2i.values()).index(max_ind)]

            if word == "EOS_token":
                decoded_words.append("<EOS>")
                break
            else:
                decoded_words.append(word)

            if word in self.w2emb:
                decoder_input = torch.Tensor(self.w2emb[word]).unsqueeze(0)
                decoder_input = decoder_input.unsqueeze(0).to(self.device)
            else:
                decoder_input = torch.Tensor([0]*100).unsqueeze(0)
                decoder_input = decoder_input.unsqueeze(0).to(self.device)
        return " ".join(decoded_words)

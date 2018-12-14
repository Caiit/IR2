from code.rerankwrite.saliency_model import SaliencyPrediction
from code.rerankwrite.encoder_decoder import CreateResponse
from data_utils import embed_sentence
import torch
import torch.nn as nn
import numpy as np

class Rewrite():
    def __init__(self, model_folder, embeddings, w2i, SOS_token, EOS_token):
        #print("Please replace me with the real class")
        ## TODO: load model
        self.saliencymodel = load_saliency_model(model_folder)
        self.encoder_decoder = load_encoder_decoder_model(model_folder)
        self.embeddings = embeddings
        self.w2i = w2i
        self.embedding_size = len(embeddings[w2i.values()[0]])
        self.SOS = SOS_token
        self.EOS = EOS_token

    def load_saliency_model(self, model_folder):
        model = torch.load(model_folder + "/saliency.pt")
        model.eval()
        return model

    def load_encoder_decoder_model(self, model_folder):
        model = torch.load(model_folder + "/encoder_decoder.pt)
        model.eval()
        return model


    def rerank(self, templates, ranked_resources, ranked_classes):
        best_resource = ranked_resources[0]
        best_resource_class = ranked_classes[0]
        templates_best_class = templates[best_resource_class]
        embed_resource = embed_sentence(best_resource, self.embeddings, self.w2i)
        saliencies = []
        for template in templates_best_class:
            embed_template = embed_sentence(template, self.embeddings, self.w2i)
            curr_saliency = self.saliencymodel.forward(embed_resource, embed_template)
            saliencies.append(curr_saliency)
        best_template = templates_best_class[np.argmax(saliencies)]
        return embed_resource, embed_sentence(best_template, self.embeddings, self.w2i)

    def rewrite(self, best_response, best_template):
        # Inputs are both embeddings
        target_length = 50 # WAT ZULLEN WE DOEN?
        complete_sent = torch.cat(best_response, best_template)
        input_length = complete_sent.size(0)
        encoder_outputs = torch.zeros(max_length, self.encoder_decoder.encoder.hidden_size)

        for ei in range(input_length):
            encoder_output, encoder_hidden = self.encoder_decoder.encoder.forward(input_tensor[ei], encoder_hidden)
            encoder_outputs[ei] = encoder_output[0, 0]

        decoder_input = torch.tensor([[self.SOS]])
        decoder_hidden = encoder_hidden
        decoded_words = []

        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = self.encoder_decoder.decoder(decoder_input,
                decoder_hidden, encoder_outputs)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()  # detach from history as input

            if decoder_input.item() == self.EOS:
                decoded_words.append(self.EOS)
                break
            else:
                decoded_words.append(topi.item())

        return decoded_words

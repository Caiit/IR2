from code.rerankwrite.saliency_model import SaliencyPrediction
from code.rerankwrite.encoder_decoder import CreateResponse
from data_utils import embed_sentence
import torch
import torch.nn as nn
import numpy as np

class Rewrite():
    def __init__(self, model_folder, embeddings, w2i):
        #print("Please replace me with the real class")
        ## TODO: load model
        self.saliencymodel = load_saliency_model(model_folder)
        self.encoder_decoder = load_encoder_decoder_model(model_folder)
        self.embeddings = embeddings
        self.w2i = w2i
        self.embedding_size = len(embeddings[w2i.values()[0]])

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
        # DIT MOET NOG ANDERS MET FOR LOOP
        return self.encoder_decoder.forward(best_response, best_template)

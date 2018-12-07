import torch
import torch.nn as nn
from model import EncoderLSTM, AttnDecoderRNN
import json


def train_encoder_decoder(args):
    parameter_file = args.parameters
    params = json.loads(open(parameter_file).read())

    # Hier dus iets met de data
    # TEMPLATES
    # RESOURCES
    # ACTUAL RESPONSES
    input_size = # DO HERE SOMETHING WITH SIZE EMBEDDING
    out_size = params["out_size"]
    hidden_size = params["hidden_size"]
    batch_size = params["batch_size"]
    drop_out = params["drop_out"]
    num_layers = params["num_layers"]
    MAX_LENGTH = params["MAX_LENGTH"]


    encoder = EncoderLSTM(input_size, out_size, hidden_size, num_layers, drop_out, batch_size)
    decoder = AttnDecoderRNN(hidden_size, out_size, drop_out, MAX_LENGTH):

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=params["learning_rate"])
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=params["learning_rate"])

    # cross entropy
    # negative log likelihood
    criterion_encoder = nn.CrossEntropyLoss()
    criterion_decoder = nn.NLLLoss()

    # Now loop over each sentence:
    hidden_t = encoder.initHidden()
    hidden_r = encoder.initHidden()
    context, saliency = encoder.forward(self, template, resource, hidden_t, hidden_r)
    # NOW CALCULATE THE ACTUAL SALIENCY
    loss_encoder = encoder_optimizer(saliency, actual_saliency)
    loss_encoder.backward()
    loss_encoder.step()

    input = ["<SOS>"]
    hidden = decoder.initHidden()
    predicted_respons, out, hidden, attn_weights = decoder.forward(input, hidden, context, MAX_LENGTH))

    loss_decoder = decoder_optimizer(predicted_respons, respons)
    loss_decoder.backward()
    loss_decoder.step()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("data_folder", help="path to folder of the dataset.")
    parser.add_argument("parameters", help="file that contains the training parameters.")
    args = parser.parse_args()
    train_encoder_decoder(args)

import sys
sys.path.append("..")
import data_utils

templates = data_utils.get_templates("../../data/templates.pkl")
w2emb = data_utils.get_w2emb("../../embeddings/w2emb.pkl")

class_names = ["Comments", "Facts", "Plot", "Review"]
for i, resource_class in enumerate(templates):
    for template in resource_class:
        print(class_names[i])
        print(data_utils.convert_to_words(template, w2emb))

# IR2
Project for IR2 course.


To run the code, you firstly need to download pre-trained
[GloVe](https://nlp.stanford.edu/projects/glove/) embeddings. Right now, we are using 50D Wikipedia
embeddings due to memory limitations.

1. Run dataset.py and give as input: the the path to the data file, the path to the GloVe embeddings file, the path to where you would like to save the generated embeddings and the path to where you would like to save the generated word to index (w2i) file. Example:
```python dataset.py data/train_data.json glove/glove.6B.50d.txt data/embeddings.pkl data/w2i.pkl```

2. Install gensim using pip. Example: ```pip3 install gensim```

4. Run main.py and give as input: the path to the data file, the path to where you saved the pretrained GloVe embeddings, the path to where you saved the pretrained Word2Vec embeddings,
and the path to the word to index file generated with dataset.py. Also give the path to the folder containing the prediction model. Example:
```python main.py ../data/train_data.json ../data/glove_50d.pkl .../data/w2v_vectors.kv data/w2i.pkl  models/prediction/```


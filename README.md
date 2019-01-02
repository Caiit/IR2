# IR2
Project for IR2 course.


To run the code, you firstly need to download pre-trained
[GloVe](https://nlp.stanford.edu/projects/glove/) embeddings.

1. Run dataset.py to generate Glove embeddings from the training set, and to generate a word2index and word2embedding file. Additional arguments can be given to use a different pre-trained Glove embeddings file and to adjust the output accordingly.
```python dataset.py```

2. Run main.py to run the Response-Generating Retrieve, Rerank, Rewrite model.
```python main.py```

3. To separately train the rerank and rewrite module respectively, the following commands need to be run, where additional arguments can be given to change the training according to the settings that are desired.
```python train_saliency.py```
```python train_encoderdecoder.py```


## Dependencies
- Rouge: pip install rouge

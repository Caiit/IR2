# IR2
Project for IR2 course.


To run the code, you firstly need to download pre-trained
[GloVe](https://nlp.stanford.edu/projects/glove/) embeddings. Right now, we are using 50D Wikipedia embeddings due to memory limitations. 


1. Run dataset.py in 1 `code/` and give the the path to the data file, the path to the GloVe embeddings file, the path to where you would like to save the generated embeddings and the path to where you would like to save the generated word to index (w2i) file. Example: 
```python dataset.py data/train_data.json glove/glove.6B.50d.txt data/embeddings.pkl data/w2i.pkl```
2. Run train.py in `code/class_prediction` and give the path to the data folder and the path to the parameters config file (`parameters.json`). Example: 
```python train.py ../data/ parameters.json ```

# IR2
Project for IR2 course.

According to the paper, these parameters are used:

`Trained the HRED model using Adam (Kingma and Ba, 2014) optimizer with an
initial learning rate of 0.001 on a minibatch of size 16.
We used a dropout (Srivastava et al., 2014) with a rate of 0.25.
For word embeddings we use pre-trained GloVe (Pennington et al., 2014) embeddings of size 300.
For all the encoders and decoders in the model we used Gated Recurrent Unit (GRU) with 300 as the size of the hidden state.
We restricted our vocabulary size to 20,000 most frequent words.`

Train the model:

``` python train_hred.py --config_id 1 --data_dir ../data/hred/ --infer_data test --logs_dir logs
--checkpoint_dir checkpoints --rnn_unit gru --learning_rate 0.0001 --batch_size 16 --dropout 0.25
--num_layers 1 --word_emb_dim 300 --hidden_units 300 --eval_interval 1 --train=True --debug=False
```

Evaluate HRED:

``` python metrics/evaluate.py --config_id 1 --preds_path results/ ```

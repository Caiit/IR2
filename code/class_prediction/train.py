import argparse
import os
import sys
import json
import time
import logging
import dataset
import numpy as np
import tensorflow as tf
from text_cnn import TextCNN
from tensorflow.contrib import learn

logging.getLogger().setLevel(logging.INFO)

def train_cnn(args):
    """Step 0: load sentences, labels, and training parameters"""
    data_folder = args.data_folder
    max_length = args.max_length
    x_train, y_train = dataset.load_data_and_labels(data_folder + "/train_data.json", max_length, args.use_gensim, args.embeddings, args.w2i)
    x_test, y_test = dataset.load_data_and_labels(data_folder + "/test_data.json", max_length, args.use_gensim, args.embeddings, args.w2i)
    x_dev, y_dev = dataset.load_data_and_labels(data_folder + "/dev_data.json", max_length, args.use_gensim, args.embeddings, args.w2i)

    parameter_file = args.parameters
    params = json.loads(open(parameter_file).read())

    logging.info('x_train: {}, x_dev: {}, x_test: {}'.format(len(x_train), len(x_dev), len(x_test)))
    logging.info('y_train: {}, y_dev: {}, y_test: {}'.format(len(y_train), len(y_dev), len(y_test)))

    x_train = np.array(x_train)
    y_train = np.array(y_train)
    x_test = np.array(x_test)
    y_test = np.array(y_test)
    x_dev = np.array(x_dev)
    y_dev = np.array(y_dev)

    """Step 2: build a graph and cnn object"""
    graph = tf.Graph()
    with graph.as_default():
        session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            cnn = TextCNN(
                    sequence_length=x_train.shape[1],
                    num_classes=y_train.shape[1],
                    embedding_size=params['embedding_dim'],
                    filter_sizes=list(map(int, params['filter_sizes'].split(","))),
                    num_filters=params['num_filters'],
                    l2_reg_lambda=params['l2_reg_lambda'])

            global_step = tf.Variable(0, name="global_step", trainable=False)
            optimizer = tf.train.AdamOptimizer(1e-3)
            grads_and_vars = optimizer.compute_gradients(cnn.loss)
            train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

            timestamp = str(int(time.time()))
            out_dir = os.path.abspath(os.path.join(os.path.curdir, "trained_model_" + timestamp))

            checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
            checkpoint_prefix = os.path.join(checkpoint_dir, "model")
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            saver = tf.train.Saver()

            # One training step: train the model with one batch
            def train_step(x_batch, y_batch):
                feed_dict = {
                    cnn.input_x: x_batch,
                    cnn.input_y: y_batch,
                    cnn.dropout_keep_prob: params['dropout_keep_prob']}
                _, step, loss, acc = sess.run([train_op, global_step, cnn.loss, cnn.accuracy], feed_dict)

            # One evaluation step: evaluate the model with one batch
            def dev_step(x_batch, y_batch):
                feed_dict = {cnn.input_x: x_batch, cnn.input_y: y_batch, cnn.dropout_keep_prob: 1.0}
                step, loss, acc, num_correct, predictions, probs = sess.run([global_step, cnn.loss, cnn.accuracy, cnn.num_correct, cnn.predictions, cnn.probs], feed_dict)

                acc_per_class = []
                for i in range(y_train.shape[1]):
                    same_label = np.logical_and(np.equal(np.argmax(y_batch, 1), predictions), np.equal(predictions, i))
                    acc = 0
                    if (sum(np.equal(np.argmax(y_batch, 1), i)) > 0):
                        acc = sum(same_label) / sum(np.equal(np.argmax(y_batch, 1), i))
                    acc_per_class.append(acc)
                return num_correct, acc_per_class, loss

            sess.run(tf.global_variables_initializer())

            # Training starts here
            train_batches = dataset.batch_iter(list(zip(x_train, y_train)), params['batch_size'], params['num_epochs'])
            best_accuracy, best_at_step = 0, 0

            """Step 3: train the cnn model with x_train and y_train (batch by batch)"""
            for train_batch in train_batches:
                x_train_batch, y_train_batch = zip(*train_batch)
                train_step(x_train_batch, y_train_batch)
                current_step = tf.train.global_step(sess, global_step)

                """Step 3.1: evaluate the model with x_dev and y_dev (batch by batch)"""
                if current_step % params['evaluate_every'] == 0:
                    dev_batches = dataset.batch_iter(list(zip(x_dev, y_dev)), params['batch_size'], 1)
                    total_dev_correct = 0
                    n_batches = 0
                    acc_per_class = np.zeros(y_train.shape[1])
                    total_loss = 0
                    for dev_batch in dev_batches:
                        x_dev_batch, y_dev_batch = zip(*dev_batch)
                        num_dev_correct, acc, loss = dev_step(x_dev_batch, y_dev_batch)
                        total_dev_correct += num_dev_correct
                        acc_per_class += np.array(acc)
                        total_loss += loss
                        n_batches += 1
                    acc_per_class /= n_batches
                    avg_loss = total_loss / n_batches

                    dev_accuracy = float(total_dev_correct) / len(y_dev)
                    logging.critical('Accuracy on dev set: {}'.format(dev_accuracy))
                    logging.critical('Accuracy per class: {}'.format(acc_per_class))
                    logging.critical('Average loss: {}'.format(avg_loss))

                    """Step 3.2: save the model if it is the best based on accuracy on dev set"""
                    if dev_accuracy >= best_accuracy:
                        best_accuracy, best_at_step = dev_accuracy, current_step
                        path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                        logging.critical('Saved model at {} at step {}'.format(path, best_at_step))
                        logging.critical('Best accuracy is {} at step {}'.format(best_accuracy, best_at_step))

            """Step 4: predict x_test (batch by batch)"""
            test_batches = dataset.batch_iter(list(zip(x_test, y_test)), params['batch_size'], 1)
            total_test_correct = 0
            n_batches = 0
            acc_per_class = np.zeros(y_train.shape[1])
            total_loss = 0
            for test_batch in test_batches:
                x_test_batch, y_test_batch = zip(*test_batch)
                num_test_correct, acc, loss = dev_step(x_test_batch, y_test_batch)
                total_test_correct += num_test_correct
                acc_per_class += np.array(acc)
                total_loss += loss
                n_batches += 1
            acc_per_class /= n_batches
            avg_loss = total_loss / n_batches

            test_accuracy = float(total_test_correct) / len(y_test)
            logging.critical('Accuracy on test set is {} based on the best model {}'.format(test_accuracy, path))
            logging.critical('Accuracy per class: {}'.format(acc_per_class))
            logging.critical('Average loss: {}'.format(avg_loss))
            logging.critical('The training is complete')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("data_folder", help="path to folder of the dataset.", default="../../data")
    parser.add_argument("parameters", help="file that contains the training parameters.", default="parameters.json")
    parser.add_argument("embeddings", help="file where the embeddings are saved", default="../../data/embeddings.pkl")
    parser.add_argument("--use_gensim", help="whether gensim embeddings are used", type=bool, default=False)
    parser.add_argument("--w2i", help="path to w2i file", default="../../data/w2i.pkl")
    parser.add_argument("--max_length", help="max length of a sentence", default=110)
    args = parser.parse_args()
    train_cnn(args)

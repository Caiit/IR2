import json
import numpy as np

def load_data_and_labels(filename):
    """Load sentences and labels"""
    with open(filename, encoding="utf-8") as f:
        data = json.load(f)

    x_raw = []
    y_raw = []
    for example in data:
        chat = example["chat"]
        labels = example["labels"]
        for i in range(len(chat[3:])):
            x_raw.append(" ".join(chat[i - 3: i]))
            label_hot = np.zeros(5)
            label_hot[labels[i]] = 1
            y_raw.append(label_hot)
    return x_raw, y_raw


def batch_iter(data, batch_size, num_epochs, shuffle=True):
	"""Iterate the data batch by batch"""
	data = np.array(data)
	data_size = len(data)
	num_batches_per_epoch = int(data_size / batch_size) + 1

	for epoch in range(num_epochs):
		if shuffle:
			shuffle_indices = np.random.permutation(np.arange(data_size))
			shuffled_data = data[shuffle_indices]
		else:
			shuffled_data = data

		for batch_num in range(num_batches_per_epoch):
			start_index = batch_num * batch_size
			end_index = min((batch_num + 1) * batch_size, data_size)
			yield shuffled_data[start_index:end_index]

if __name__ == '__main__':
	input_file = '../../data/train_data.jsons'
	load_data_and_labels(input_file)

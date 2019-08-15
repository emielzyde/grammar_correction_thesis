from pathsetup import run_path_setup
run_path_setup()

import os
import gl
gl.isTrain = True

from model_config import model_argparse
config = model_argparse()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = config['device']

import tensorflow as tf
tf_config = tf.ConfigProto()
tf_config.gpu_options.allow_growth = True
sess = tf.Session(config=tf_config)

import numpy as np
import utils

from det_wae import DetWAEModel
from sklearn.model_selection import train_test_split

np.random.seed(1337)

if __name__ == '__main__':
    combined_data = utils.get_sentences(file_path = '/Users/emielzyde/Downloads/noisy_data.txt')
    input_data = utils.get_sentences(file_path = '/Users/emielzyde/Downloads/clean_data.txt')
    output_data = utils.get_sentences(file_path = '/Users/emielzyde/Downloads/noisy_data.txt')

    labels = []
    label_path = '/Users/emielzyde/Downloads/labels.txt'
    with open(label_path, 'r') as f:
        label_data = f.readlines()
    for item in label_data:
        item = item.rstrip()
        labels.append(int(item))

    print('[INFO] Number of sentences = {}'.format(len(combined_data)))

    combined_sentences = [s.strip() for s in combined_data]
    input_sentences = [s.strip() for s in input_data]
    output_sentences = [s.strip() for s in output_data]

    print('[INFO] Tokenizing input and output sequences')
    filters = '!"#$%&()*+/:;<=>@[\\]^`{|}~\t\n'
        
    word_index = utils.get_dict(combined_sentences, filters,config['num_tokens'], 100)
    input_sents = utils.tokenize_sequence(input_sentences, filters, config['num_tokens'], word_index)
    output_sents = utils.tokenize_sequence(output_sentences, filters, config['num_tokens'], word_index)

    print('[INFO] Split data into train-validation-test sets')
    input_train, input_val, output_train, output_val, label_train, label_val = train_test_split(input_sents, output_sents, labels, test_size = 0.05, random_state = 10)

    w2v = config['w2v_file']
    embeddings_matrix = utils.create_embedding_matrix(word_index,
                                                    config['embedding_size'],
                                                    w2v)

    # Re-calculate the vocab size based on the word_idx dictionary
    config['vocab_size'] = len(word_index)

    #----------------------------------------------------------------#

    model = DetWAEModel(config,
                        embeddings_matrix,
                        word_index)

    model.train(input_train, input_val, output_train, output_val, label_train, label_val)

    log_writer.close()
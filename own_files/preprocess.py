import os
import re
import unicodedata
import pickle
import tensorflow as tf
import numpy as np
#glove_path = '/Users/emielzyde/Desktop/Project/grammar_correction/glove.6B.300d.txt'
unknown_vector = np.random.rand(300)
start_vector = np.random.rand(300)
end_vector = np.random.rand(300)

MIN_COUNT = 0

class LookupTables():
    def __init__(self, dataset):
        self.dataset = dataset
        self.word2index = {}
        self.index2word = {}
        #self.vocab = set()
        self.vocab = dict()

        self.createTable()

    def createTable(self):
        for line in self.dataset:
            #self.vocab.update(line.split())
            split_line = line.split()
            for word in split_line:
                if word in self.vocab.keys():
                    self.vocab[word] += 1
                else:
                    self.vocab[word] = 1

        self.word2index['<PAD>'] = 0
        self.word2index['<START>'] = 1
        self.word2index['<UNK>'] = 2
        self.word2index['<EOS>'] = 3

        index = 4

        for word, count in self.vocab.items():
            if count > MIN_COUNT:
                self.word2index[word] = index
                index += 1
        #for index, word in enumerate(self.vocab):
        #    self.word2index[word] = index + 4

        for word, index in self.word2index.items():
            self.index2word[index] = word

def unicode_to_ascii(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')

def preprocess_sentence(w):
    w = unicode_to_ascii(w.lower().strip())
    #w = re.sub(r"([?.!,¿])", r" \1 ", w)
    #w = re.sub(r'[" "]+', " ", w)
    #w = re.sub(r"[^a-zA-Z?.!,¿]+", " ", w)
    w = w.rstrip().strip()
    return w

def preprocess_decoding(sentence):
    sentence = preprocess_sentence(sentence)
    return sentence

class Preprocessor():
    def __init__(self, data, num_examples, mode):
        self.num_examples = num_examples
        self.mode = mode
        self.data = data
        self.word_pairs = [[preprocess_sentence(w) for w in l.split('\t')] for l in data[:self.num_examples]]

        self.input_data = [line[1] for line in self.word_pairs]
        self.target_data = [line[0] for line in self.word_pairs]

        self.input_ids = []
        self.output_ids = []
        self.max_input_len = max([len(sentence.split()) for sentence in self.input_data])
        self.max_target_len = max([len(sentence.split()) for sentence in self.target_data])

        self.finalise_dataset()


    def finalise_dataset(self):

        #glove_file = open(glove_path)
        #embeddings_index = dict()
        #for line in glove_file:
        #    values = line.split()
        #    word = values[0]
        #    coefficients = np.asarray(values[1:], dtype='float32')
        #    embeddings_index[word] = coefficients

        input_table = LookupTables(self.input_data)
        output_table = LookupTables(self.target_data)

        self.input_ids = []
        #self.input_ids_glove = []
        self.target_ids = []
        #self.target_ids_glove = []

        for inputter in self.input_data:
            curr_list = []
            #glove_list = []
            for word in inputter.split():
                if word in input_table.word2index.keys():
                    curr_list.append(input_table.word2index[word])
                    #embedding_vector = embeddings_index.get(word)
                    #if embedding_vector is not None:
                    #    glove_list.append(embedding_vector)
                else:
                    curr_list.append(input_table.word2index['<UNK>'])
                    #glove_list.append(unknown_vector)
            self.input_ids.append(curr_list)
            #self.input_ids_glove.append(glove_list)

        for outputter in self.target_data:
            curr_list = []
            #glove_list = []
            for word in outputter.split():
                if word in output_table.word2index.keys():
                    curr_list.append(output_table.word2index[word])
                    #embedding_vector = embeddings_index.get(word)
                    #if embedding_vector is not None:
                    #    glove_list.append(embedding_vector)
                else:
                    curr_list.append(output_table.word2index['<UNK>'])
                    #glove_list.append(unknown_vector)
            self.target_ids.append(curr_list)
            #self.target_ids_glove.append(glove_list)

        #self.input_ids = [[input_table.word2index[word] for word in inputter.split()] for inputter in self.input_data]
        #self.target_ids = [[output_table.word2index[word] for word in outputter.split()] for outputter in self.target_data]

        if self.mode == 'TRAIN':
            for element in self.input_ids:
                element.append(input_table.word2index['<EOS>'])
                element.insert(0,input_table.word2index['<START>'])
            for element2 in self.target_ids:
                element2.append(output_table.word2index['<EOS>'])
                element2.insert(0, output_table.word2index['<START>'])

        max_length_input, max_length_target = self.max_length(self.input_ids), self.max_length(self.target_ids)

        target_lengths = []
        for i in range(len(self.target_ids)):
            curr_length = len(self.target_ids[i])
            target_lengths.append(curr_length)

        # Padding the input and output tensor to the maximum length
        # Will explore doing this here (can lead to unnecessary padding) and doing it for each batch separately
        input_dataset = tf.keras.preprocessing.sequence.pad_sequences(self.input_ids, maxlen=max_length_input, padding='post')
        target_dataset = tf.keras.preprocessing.sequence.pad_sequences(self.target_ids , maxlen=max_length_target, padding='post')

        return input_dataset, target_dataset, input_table, output_table, max_length_input, max_length_target, input_table.word2index, output_table.word2index, input_table.index2word, output_table.index2word, target_lengths

    def retrieve_output(self):
        input_table = LookupTables(self.input_data)
        output_table = LookupTables(self.target_data)

        return self.input_ids, self.target_ids, input_table.word2index, output_table.word2index, input_table.index2word, output_table.index2word

    def max_length(self, inputter):
        return max(len(t) for t in inputter)

if __name__ == '__main__':
    input_file = os.path.join('/Users/emielzyde/Downloads/fra-eng/fra.txt')
    with open(input_file, 'r') as f:
        lang_data = f.readlines()
    data_holder = Preprocessor(lang_data, 200, 'TRAIN')

    input_dataset, target_dataset, input_table, output_table, max_length_inp, max_length_tar, input_table.word2index, output_table.word2index, input_table.index2word, output_table.index2word = data_holder.finalise_dataset()
    pickle.dump((input_dataset, target_dataset, input_table, output_table, max_length_inp, max_length_tar, input_table.word2index, output_table.word2index, input_table.index2word, output_table.index2word), open('preprocess_outcome.pickle', 'wb'))


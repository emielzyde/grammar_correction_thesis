import tensorflow as tf
import tensorflow.contrib.eager as tfe
tf.enable_eager_execution()

from f_score import calculate_f_score
import os
from sklearn.model_selection import train_test_split
import postprocess
import preprocess
import pickle
import numpy as np

NUM_HIDDEN_UNITS = 256
EMBEDDING_DIM = 256
BATCH_SIZE = 64
LAYERS = 4
DROPOUT = 0.5
EPOCHS = 500
TEST_SPLIT = 0.01
LEARNING_RATE = 0.01
NUM_DATA = 1000

class Denoiser(tf.keras.Model):
    def __init__(self, vocab_size, input_length):
        super(Denoiser, self).__init__()
        self.denoiser_size = NUM_HIDDEN_UNITS
        self.vocab_size = vocab_size
        self.input_length = input_length
        self.LSTM_cell = tf.nn.rnn_cell.LSTMCell(num_units=self.denoiser_size)
        self.LSTM_cell2 = tf.nn.rnn_cell.LSTMCell(num_units = self.denoiser_size)
        self.LSTM_forw = tf.contrib.rnn.DropoutWrapper(self.LSTM_cell, input_keep_prob= DROPOUT, output_keep_prob= DROPOUT)
        self.LSTM_back = tf.contrib.rnn.DropoutWrapper(self.LSTM_cell2, input_keep_prob= DROPOUT, output_keep_prob= DROPOUT)
        self.embedding = tf.keras.layers.Embedding(self.vocab_size, EMBEDDING_DIM)

        self.parameters = {"W": tfe.Variable(tf.contrib.layers.xavier_initializer()([self.denoiser_size * 2,2],dtype=tf.float32)),
                           "b": tfe.Variable(tf.contrib.layers.xavier_initializer()([2], dtype=tf.float32))}

    def call(self, inputs, outputs, mask):
        embedded_inputs = self.embedding(inputs)
        (output_fw, output_bw), _ = tf.nn.bidirectional_dynamic_rnn(self.LSTM_forw, self.LSTM_back, inputs= embedded_inputs, dtype=tf.float32)
        output_layer = tf.concat([output_fw, output_bw], axis=2)
        #print(output_layer.shape)
        linear_output = tf.einsum('ijk,kl->ijl', output_layer, self.parameters["W"]) + self.parameters["b"]

        loss = tf.nn.softmax_cross_entropy_with_logits(logits = linear_output, labels = outputs)
        loss = loss * mask
        #print(loss.shape)
        loss = tf.reduce_mean(loss)
        #print(np.argmax(linear_output, axis = 2))
        #print(np.argmax(outputs, axis = 2 ))
        accuracy = np.mean(np.equal(np.argmax(outputs, axis = 2) * mask, np.argmax(linear_output, axis = 2) * mask))

        predictions = np.ndarray.flatten(np.array(np.argmax(linear_output, axis = 2) * mask))
        targets = np.ndarray.flatten(np.array(np.argmax(outputs, axis = 2) * mask))
        f_score_cal = calculate_f_score(predictions, targets)
        #print(loss.shape)
        #optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(loss)

        return loss, accuracy, f_score_cal

if __name__ == '__main__':
    input_file = os.path.join('/Users/emielzyde/Desktop/Project/grammar_correction/lang8_preprocess.pickle')
    with open(input_file, 'rb') as f:
        lang_data = pickle.load(f)
    new_data = [[preprocess.preprocess_sentence(w) for w in l.split('\t')] for l in lang_data[:NUM_DATA]]

    label_holder = []
    input_sentences = []
    for line in new_data:
        labels = postprocess.sentence_labeller(line[0], line[1])
        label_holder.append(labels)
        input_sentences.append(line[1])

        #label_holder = np.array(label_holder)
    #Pre-process the data
    data_holder = preprocess.Preprocessor(lang_data, NUM_DATA, 'TRAIN')
    _, target_dataset, _, output_table, _, max_length_tar, _, _, _, output_index2word, target_lengths = data_holder.finalise_dataset()

    train_targets, val_targets, train_labels, val_labels, train_lengths, val_lengths = train_test_split(target_dataset, label_holder, target_lengths, test_size = TEST_SPLIT)
    #Feeding the data in reverse order helps with training
    #input_dataset = np.flip(input_dataset)

    #Create a dataset
    padded_inputs = tf.keras.preprocessing.sequence.pad_sequences(train_targets, maxlen=max_length_tar, padding='post')
    label_holder = tf.keras.preprocessing.sequence.pad_sequences(train_labels, maxlen=max_length_tar, padding='post')
    padded_outputs = tf.keras.preprocessing.sequence.pad_sequences(train_labels,  maxlen=max_length_tar,padding='post')

    masker = np.zeros((len(train_lengths), max_length_tar))
    for i in range(len(train_lengths)):
        masker[i, 0:train_lengths[i]] = 1

    padded_outputs_new = np.zeros((padded_outputs.shape[0], padded_outputs.shape[1], 2))
    for i in range(padded_outputs.shape[0]):
        for j in range(padded_outputs.shape[1]):
            if padded_outputs[i, j] == 1:
                padded_outputs_new[i, j, 1] = 1
            elif padded_outputs[i, j] == 0:
                padded_outputs_new[i, j, 0] = 1

    #Val data
    val_padded_inputs = tf.keras.preprocessing.sequence.pad_sequences(val_targets, maxlen=max_length_tar, padding='post')
    val_label_holder = tf.keras.preprocessing.sequence.pad_sequences(val_labels, maxlen=max_length_tar, padding='post')
    val_padded_outputs = tf.keras.preprocessing.sequence.pad_sequences(val_labels,  maxlen=max_length_tar,padding='post')

    val_masker = np.zeros((len(val_lengths), max_length_tar))
    for i in range(len(val_lengths)):
        val_masker[i, 0:val_lengths[i]] = 1

    val_padded_outputs_new = np.zeros((val_padded_outputs.shape[0], val_padded_outputs.shape[1], 2))
    for i in range(val_padded_outputs.shape[0]):
        for j in range(val_padded_outputs.shape[1]):
            if val_padded_outputs[i, j] == 1:
                val_padded_outputs_new[i, j, 1] = 1
            elif val_padded_outputs[i, j] == 0:
                val_padded_outputs_new[i, j, 0] = 1


    number_batches = len(train_targets) // BATCH_SIZE
    #input_vocab_size = len(input_table.word2index)
    #target_vocab_size = len(output_table.word2index)
    padded_inputs = padded_inputs.astype(np.int64)
    padded_outputs_new = padded_outputs_new.astype(np.int64)
    masker = masker.astype(np.float32)
    print(padded_inputs.shape, padded_outputs_new.shape, masker.shape)

    dataset = tf.data.Dataset.from_tensor_slices((padded_inputs, padded_outputs_new, masker)).shuffle(len(train_targets))
    dataset = dataset.batch(BATCH_SIZE, drop_remainder = True)

    denoiser = Denoiser(len(output_index2word), max_length_tar)
    optimizer = tf.train.AdamOptimizer(learning_rate = LEARNING_RATE)
    #evidence = tf.placeholder(tf.float32, [BATCH_SIZE, max_length_tar])
    #output = tf.placeholder(tf.float32, [BATCH_SIZE, max_length_tar, 2])
    #mask = tf.placeholder(tf.float32, [BATCH_SIZE, max_length_tar])
    #loss, optimiser, linear_output = denoiser.run_model(evidence, output, mask)

    #init = tf.global_variables_initializer()

    #with tf.train.SingularMonitoredSession() as sess:
    #    sess.run(init)
        #padded_inputs = tf.keras.preprocessing.sequence.pad_sequences(target_dataset[:64], maxlen = max_length_tar, padding='post')
        #padded_outputs = tf.keras.preprocessing.sequence.pad_sequences(label_holder[:64], maxlen = max_length_tar, padding = 'post')
        #padded_outputs = np.reshape(padded_outputs, [-1])
        #masker = np.zeros((64, max_length_tar))
        #for i in range(64):
        #    masker[i, 0:target_lengths[:64][i]] = 1

        #padded_outputs_new = np.zeros((padded_outputs.shape[0],padded_outputs.shape[1],2))
        #for i in range(padded_outputs.shape[0]):
        #    for j in range(padded_outputs.shape[1]):
        #        if padded_outputs[i,j] == 1:
        #            padded_outputs_new[i,j, 1] =1
        #        elif padded_outputs[i,j] ==0:
        #            padded_outputs_new[i,j,0] = 1

    for epoch in range(EPOCHS):
        best_loss = 100
        for (batch, (input_data, target_data, mask_data)) in enumerate(dataset):
            loss = 0
            with tf.GradientTape() as tape:
                #loss_, optimizer_, linear_output_ = sess.run([loss, optimiser, linear_output], feed_dict={evidence:padded_inputs, output: padded_outputs_new, mask : masker})
                #loss_, optimizer_, linear_output_ = sess.run([loss, optimiser, linear_output], feed_dict={evidence: input_data, output: target_data, mask : mask_data})
                loss, accuracy, f_scorer = denoiser.call(input_data, target_data, mask_data)
                #print(loss)
                batch_loss = loss
                variables = denoiser.variables
                #print(variables)
                gradients = tape.gradient(loss, variables)
                optimizer.apply_gradients(zip(gradients, variables))

                if batch % 10 == 0:
                    val_loss, val_accuracy, val_f_scorer = denoiser.call(val_padded_inputs[:1000], val_padded_outputs_new[:1000], val_masker[:1000])
                    print('Epoch {}, Batch {}, Loss {:.4f}, Acc {:.4f}, F-score {:4f}, Val Loss {:.4f}, Val F-score {:4f}'.format(epoch + 1, batch, batch_loss.numpy(), accuracy, f_scorer, val_loss.numpy(), val_f_scorer))
        #print(linear_output)
        #print(optimizer)





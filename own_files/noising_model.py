import tensorflow as tf
tf.enable_eager_execution()
import numpy as np
import preprocess
import os
import beam_search
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pickle

BATCH_SIZE = 64
NUM_ENCODER_UNITS = 1024
NUM_DECODER_UNITS = 1024
EMBEDDING_DIM = 256
LAYERS = 4
DROPOUT = 0.5
EPOCHS = 500
TEST_SPLIT = 0.15

def initialize_cell(units):
    cell = tf.keras.layers.GRU(units, return_sequences = True, return_state = True)
    return cell

def loss_function(targets, predictions):
    #Padded parts of the input (which have a value of 0) will be masked.
    masked_inputs = np.equal(targets, 0)
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels= targets, logits=predictions) * (1-masked_inputs)
    accuracy = np.mean(np.equal(targets, np.argmax(predictions, axis =1)))

    #Calculate the mean of the loss across the batch size
    return tf.reduce_mean(loss), accuracy

class Encoder(tf.keras.Model):
    def __init__(self, vocab_size):
        super(Encoder, self).__init__()
        self.vocab_size = vocab_size
        self.batch_size = BATCH_SIZE
        self.encoder_size = NUM_ENCODER_UNITS
        #self.encoder_layers = LAYERS
        self.embedding = tf.keras.layers.Embedding(self.vocab_size, EMBEDDING_DIM)
        self.cell = initialize_cell(self.encoder_size)

    def call(self, input):
        embedded_inputs = self.embedding(input)
        output, state = self.cell(embedded_inputs)

        return output, state

class Decoder(tf.keras.Model):
    def __init__(self, vocab_size):
        super(Decoder, self).__init__()
        self.vocab_size = vocab_size
        self.batch_size = BATCH_SIZE
        self.decoder_size = NUM_DECODER_UNITS
        #self.decoder_layers = LAYERS
        self.embedding = tf.keras.layers.Embedding(self.vocab_size, EMBEDDING_DIM)
        self.cell = initialize_cell(self.decoder_size)
        self.fully_connected_layer = tf.keras.layers.Dense(self.vocab_size)

        self.attention_dense1 = tf.keras.layers.Dense(self.decoder_size)
        self.attention_dense2 = tf.keras.layers.Dense(self.decoder_size)
        self.attention_output_layer = tf.keras.layers.Dense(1)

    def call(self, input, hidden_state, encoder_output):
        #The hidden state is expanded so that it has the same dimension as the encoder output (to perform addition).
        expand_hidden = tf.expand_dims(hidden_state, 1)

        #The score for Baldanau's additive attention is clculated as per the standard formula.
        #The shape of score is [batch_size, max_length, 1]
        score = self.attention_output_layer(tf.nn.tanh(self.attention_dense1(expand_hidden) + self.attention_dense2(encoder_output)))

        #A softmax is applied across the length of the input (axis 1).
        attention_weights = tf.nn.softmax(score, axis=1)

        #The context vector sums the product of the attention weights with the encoder output.
        #The result is a tensor of size [batch_size, hidden_size
        context_vector = tf.reduce_sum(encoder_output * attention_weights, axis = 1)

        #The input is embedded now.
        embedded_input = self.embedding(input)

        #The input and context vector are concatenated and fed into the decoder.
        #The output has a shape [batch_size, 1, hidden_size]
        concatenated_data = tf.concat([tf.expand_dims(context_vector, 1), embedded_input], axis=-1)
        output, state = self.cell(concatenated_data)

        #The second dimension of the output above can be removed.
        output = tf.reshape(output, (output.shape[0], output.shape[2]))

        #Finally, the output is passed through the dense layer to get a 'distribution' over the vocabulary.
        predictions = self.fully_connected_layer(output)

        return predictions, state, attention_weights

def attention_plot(attention_weights, predicted_sentence, target_sentence):
    figure = plt.figure(figsize = (8,8))
    axis = figure.add_subplot(1,1,1)
    axis.matshow(attention_weights)

    axis.set_xticklabels([''] + target_sentence, rotation = 90)
    axis.set_yticklabels([''] + predicted_sentence)

    plt.show()

def decode_inference(sentence, encoder, decoder, input_word2index, output_index2word, max_input_length, max_target_length):

    attention_matrix = np.zeros((max_target_length, max_input_length))

    #First, the sentence is pre-processed
    sentence = preprocess.preprocess_decoding(sentence)
    sentence_inputs = [[input_word2index[word] for word in sentence.split()]]
    sentence_inputs = tf.keras.preprocessing.sequence.pad_sequences(sentence_inputs, maxlen=max_input_length, padding='post')

    decoded_text =  ''

    #The encoder and decoder are run.
    encoder_output, encoder_hidden_state = encoder(sentence_inputs)
    decoder_hidden = encoder_hidden_state
    # The first input to the decoder is the 'START' symbol.
    decoder_input = tf.expand_dims([output_table.word2index['<START>']], 0)

    for time_step in range(max_target_length):
        predictions, decoder_hidden, attention_weights = decoder(decoder_input, decoder_hidden, encoder_output)
        attention_weights = tf.reshape(attention_weights, (-1,))
        attention_matrix[time_step] = attention_weights.numpy()

        #The prediction is taken to be the argmax across the vocabulary (i.e. the most likely word)
        prediction_id = tf.argmax(predictions[0]).numpy()
        prediction_word = output_index2word[prediction_id]
        decoded_text += prediction_word + ' '

        #If the end-of-string (EOS) symbol is predicted, the process is stopped.
        if prediction_word == '<EOS>':
            return decoded_text, sentence, attention_matrix

        decoder_input = tf.expand_dims([prediction_id],0)

    return decoded_text, sentence, attention_matrix

if __name__ == '__main__':

    input_file = os.path.join('/Users/emielzyde/Desktop/Project/grammar_correction/lang8_preprocess.pickle')
    with open(input_file, 'rb') as f:
        #lang_data = f.readlines()
        lang_data = pickle.load(f)
        #lang_data = lang_data.readlines()
    #Pre-process the data
    data_holder = preprocess.Preprocessor(lang_data, 2000, 'TRAIN')
    input_dataset, target_dataset, input_table, output_table, max_length_inp, max_length_tar, input_word2index, output_word2index, input_index2word, output_index2word = data_holder.finalise_dataset()

    train_input_dataset, val_input_dataset, train_target_dataset, val_target_dataset = train_test_split(input_dataset, target_dataset, test_size = TEST_SPLIT)
    #Feeding the data in reverse order helps with training
    #input_dataset = np.flip(input_dataset)

    print('The vocabulary size is {}'.format(len(input_word2index)))

    #Create a dataset
    number_batches = len(train_input_dataset) // BATCH_SIZE
    input_vocab_size = len(input_table.word2index)
    target_vocab_size = len(output_table.word2index)
    dataset = tf.data.Dataset.from_tensor_slices((train_input_dataset, train_target_dataset)).shuffle(len(train_input_dataset))
    dataset = dataset.batch(BATCH_SIZE, drop_remainder= True)

    #validation_dataset = tf.data.Dataset.from_tensor_slices((val_input_dataset, val_target_dataset))
    #validation_dataset = validation_dataset.batch(BATCH_SIZE, drop_remainder=True)
    val_input_dataset = val_input_dataset[:64, :]
    val_target_dataset = val_target_dataset[:64, :]

    #Set up the encoder and decoder
    encoder = Encoder(input_vocab_size)
    decoder = Decoder(target_vocab_size)

    #Set up the optimizer
    optimizer = tf.train.AdamOptimizer(learning_rate= 0.001)

    #Set up checkpoints for saving the model
    save_directory = './checkpoints'
    save_prefix = os.path.join(save_directory, 'checkpoint')
    save_file = tf.train.Checkpoint(optimizer = optimizer, encoder = encoder, decoder = decoder)

    #Start the training process
    for epoch in range(EPOCHS):
        best_loss = 100
        for (batch, (input_data, target_data)) in enumerate(dataset):
            loss = 0
            with tf.GradientTape() as tape:
                encoder_output, encoder_hidden_state = encoder(input_data)
                decoder_hidden = encoder_hidden_state
                #The first input to the decoder is the 'START' symbol.
                decoder_input = tf.expand_dims([output_table.word2index['<START>']] * BATCH_SIZE, 1)

                for time_step in range(1, target_data.shape[1]):
                    #The decoder is run for the given time step.
                    predictions, dec_hidden, _ = decoder(decoder_input, decoder_hidden, encoder_output)
                    #The loss is calculated across the batch dimension.
                    loss_return, accuracy = loss_function(target_data[:, time_step], predictions)
                    loss += loss_return
                    #Teacher forcing is used here - the ground truth is given as input to the network.
                    decoder_input = tf.expand_dims(target_data[:, time_step], 1)

            batch_loss = loss /int(target_data.shape[1])
            variables = encoder.variables + decoder.variables
            gradients = tape.gradient(loss, variables)
            optimizer.apply_gradients(zip(gradients, variables))

            if batch % 10 == 0:

                val_loss = 0
                encoder_output, encoder_hidden_state = encoder(val_input_dataset)
                decoder_hidden = encoder_hidden_state
                #The first input to the decoder is the 'START' symbol.
                decoder_input = tf.expand_dims([output_table.word2index['<START>']] * BATCH_SIZE, 1)
                for time_step in range(1, val_target_dataset.shape[1]):
                    #The decoder is run for the given time step.
                    predictions, dec_hidden, _ = decoder(decoder_input, decoder_hidden, encoder_output)
                    #print(predictions.shape)
                    #print(dec_hidden.shape)
                    #The loss is calculated across the batch dimension.
                    loss_return, accuracy = loss_function(val_target_dataset[:, time_step], predictions)
                    val_loss += loss_return
                    #Teacher forcing is used here - the ground truth is given as input to the network.
                    decoder_input = tf.expand_dims(val_target_dataset[:, time_step], 1)
                val_loss = val_loss / int(target_data.shape[1])

                print('Epoch {}, Batch {}, Loss {:.4f}, Validation loss {:.4f}'.format(epoch + 1, batch, batch_loss.numpy(),
                                                                                val_loss.numpy()))

            #Save the model if the loss is lower than the previous best
            #This then allows us to recover the model for testing
            #if batch_loss < best_loss:
            #    save_file.save(save_prefix)
            #    print('Saved. New best loss is {:.4f}'.format(batch_loss.numpy()))
                #output_sent, sentence, attention_weights = decode_inference('Restez juste là où vous êtes.', encoder, decoder, input_word2index, output_index2word, max_length_inp, max_length_tar)
                #print(output_sent)
                #attention_matrix = attention_weights[:len(output_sent.split(" ")), :len(sentence.split(" "))]
                #attention_plot(attention_matrix, output_sent.split(" "), sentence.split(" "))

            #beam_output = beam_search.vanilla_beam_search_decoder('Je suis un homme', 10, encoder, decoder, input_word2index, output_word2index, output_index2word, max_length_inp)
            #print(beam_output)
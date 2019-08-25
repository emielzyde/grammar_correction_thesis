import numpy as np
import preprocess
import tensorflow as tf
tf.enable_eager_execution()
from operator import itemgetter

def vanilla_beam_search_decoder(input, beam_size, encoder, decoder, input_word2index, output_word2index, output_index2word, max_length_input):

    vocab_size = len(output_word2index)

    #Keep track of the finished beams (where an EOS token has been predicted) and the live ones
    completed_beams = 0
    completed_samples = []
    decoder_inputs = []
    decoder_hidden = []
    completed_scores = []

    live_samples = [[]] * beam_size
    live_scores = [0] * beam_size

    #First, the input sentence is pre-processed
    sentence = preprocess.preprocess_decoding(input)
    sentence_inputs = [[input_word2index[word] for word in sentence.split()]]
    sentence_inputs = tf.keras.preprocessing.sequence.pad_sequences(sentence_inputs, maxlen=max_length_input, padding='post')

    #The encoder is run
    encoder_output, encoder_hidden_state = encoder(sentence_inputs)

    #The initial hidden state and input for the decoder are set up.
    for i in range(beam_size):
        decoder_hidden.append(encoder_hidden_state)
        # The first input to the decoder is the 'START' (1) symbol.
        decoder_inputs.append(tf.expand_dims([output_word2index['<START>']], 0))

    while completed_beams < beam_size:
        #Set up a list to hold the possible sequences and their associated scores
        candidates = list()

        for beam in range(beam_size - completed_beams):
            sequence, score = live_samples[beam], live_scores[beam]

            #Get predictions from the decoder (prediction will be over the vocabulary size)
            predictions, decoder_hidden[beam], _ = decoder(decoder_inputs[beam], decoder_hidden[beam], encoder_output)
            #By taking the softmax of the predictions, we get normalised probabilities.
            probabilities = tf.nn.softmax(predictions)
            probabilities = tf.reshape(probabilities, [probabilities.shape[1]])

            for i in range(vocab_size):
                #For each candidate, I record the current beam, the prediction, the new score and the hidden decoder state
                candidate = [beam, i, score - np.log(probabilities[i]), decoder_hidden[beam]]
                #Add current candidate to the list
                candidates.append(candidate)

        #Sort the list of candidates by the 3rd dimension (which contains the scores
        ordered_candidates = sorted(candidates, key = itemgetter(2))

        #Return the top k candidates (where k is the beam_size)
        sequences = ordered_candidates[:beam_size-completed_beams]

        live_samples_copy = live_samples.copy()
        #Now, we add the new tokens to the respective lists
        for k in range(beam_size - completed_beams):
            j = 0
            l = 0
            #Extract relevant elements
            beam, predicted_id, score, decoder_hidden_state = sequences[k][0], sequences[k][1], sequences[k][2], sequences[k][3]
            curr_sequence = live_samples_copy[beam] + [predicted_id]  # add the new prediction to current sequence

            if predicted_id == 3 and len(curr_sequence) > 2: #if EOS token is predicted, remove the item from the live samples and add to the completed ones
                completed_samples.append(curr_sequence)
                completed_scores.append(score)
                live_samples.pop(l)
                live_scores.pop(l)
                completed_beams += 1

            else:
                live_samples[j] = curr_sequence
                live_scores[j] = score
                decoder_hidden[j] = decoder_hidden_state
                decoder_inputs[j] = tf.expand_dims([predicted_id], 0)
                j += 1
                l += 1

    outputs = []
    for sample in completed_samples:
        predicted_sent = ''
        for i in range(len(sample)):
            predicted_word = output_index2word[sample[i]]
            predicted_sent += predicted_word + " "

        outputs.append(predicted_sent)

    return outputs

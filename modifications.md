## seq2seq ##

helper.py (in seq2seq - seq2seq - contrib - seq2seq) 
1.	Added in functionality to implement the original method presented in the encoder-decoder section 

beam_search.py (in seq2seq - seq2seq - inference)
1.	Added in implementation of random noising method 

## sequence-labeler ##

experiment.py 
1. 	Added in functionality to label a specified dataset based on a pre-trained model (through get_labels and process_sentences_labelling functions)

labeler.py
1. 	Added in functionality to label a specified dataset based on a pre-trained model (though process_batch_labelling function). 3 different methods are included for various types of labelling.

evaluator.py
1.	Added in functionality to label a specified dataset based on a pre-trained model (through append_data_labelling function) 

## wronging ## 

tsvutils.py
1.	Added intermediate metrics to analyse the data (e.g. average number of incorrect tokens, percentage of incorrect tokens, corplus GLEU)
2.	Added in ability to specify the desired percentage of errors (rather than just the minimum and maximum number of errors) 

## probabilistic_nlg ##

det_wae.py (in snli - wae-det) 
1.	Added in the ability to add a one-hot encoded label to the latent space representation (which is needed for the conditional WAE)
2.	Modified the training, monitoring and sampling functions to be compatible with different input and output datasets and the addition of the one-hot encoded label 

train.py (in snli - wae-det)
1.	Modified code to allow for different input and output datasets and changed tokenisation process 

utils.py
1.	Added get_dict function which generates a word index from a specified dataset
2.	Changed tokenize_sequence function to label sequences according to word index from get_dict function. This means that the word index can be generated from a fixed dataset and used to label a range of datasets (according to this index), which was not previously possible.  	

## own_files ## 

beam_search.py - Own implementation of vanilla beam search 

combined_labels.py - Used to combine labels from self-training and heuristic labelling approaches (for 'Combined labels' section in report) 

consec_error_analysis.py - Used to analyse how well the error detection model labels consecutive incorrect tokens 

denoising_model.py - Initial implementation of the demonising model using a bi-directional LSTM architecture. This was later replaced by using the sequence-labeler repository. 

dropout_training.py - Used to implement consensus and majority vote on a range of self-trained models 

error_analysis.py - Used to analyse how the error detection models make errors (e.g. how often do they mislabel an incorrect token) 

f_score.py - Calculates precision, recall and F-scores

gan_analysis.py - Implements a variety of metrics to measure the quality of the generated GAN data (e.g. soft cosine similarity). Also used for the GPT-2 and WAE models 

gan_preprocess.py - Data pre-processing before data can be used to train GAN model

getROC.py - Calculates and plots an ROC curve and performs a threshold search 

gpt_reader.py - Pre-processes the GPT-2 generated samples

lang8_extractor.py - Extracts clean and noisy sentences from the Lang-8 dataset

noising_model.py - Initial implementation of encoder-decoder model. This was later replaced by using the seq2seq repository.

preprocess.py - Data pre-processing utilities used in denoising_model.py and noising_model.py 

queryGoogle.py - Used to query the Google Translate API. Before this file can be run, a console project and private key have to be set up (https://cloud.google.com/translate/docs/quickstart) 

sentence_analysis.py - Used to analyse the distribution of lengths and incorrect tokens in the FCE and CoNLL datasets 

xmlReader.py - Used to read the XML data from the FCE dataset, to extract relevant attributes from this data 








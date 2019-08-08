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

## Own files ## 

beam_search.py - Own implementation of vanilla beam search 

combined_labels.py - Used to combine labels from self-training and heuristic labelling approaches. 

denoising_model.py - Initial implementation of the demonising model using a bi-directional LSTM architecture. This was later replaced by using the sequence-labeler repository. 

dropout_training.py - Used to implement consensus and majority vote on a range of self-trained models 

f_score.py - Calculates precision, recall and F-scores

gan_analysis.py - Implements a variety of metrics to measure the quality of the generated GAN data

gan_preprocess.py - Data pre-processing before data can be used to train GAN model

getROC.py - Calculates and plots an ROC curve and performs a threshold search 

gpt_reader.py - Processes the GPT-2 generated samples

lang8_extractor.py - Extracts clean and noisy sentences from the Lang-8 dataset

noising_model.py - Initial implementation of encoder-decoder model. This was later replaced by using the seq2seq repository.

preprocess.py - Data pre-processing utilities used in denoising_model.py and noising_model.py 

queryGoogle.py - Used to query the Google Translate API. Before this file can be run, a console project and private key have to be set up (https://cloud.google.com/translate/docs/quickstart) 

xmlReader.py - Used to read the XML data from the FCE dataset, to extract relevant attributes from this data 








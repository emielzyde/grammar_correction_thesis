# Addressing the data sparsity bottleneck for grammatical error detection #

This repository contains the code for my thesis 'Addressing the data sparsity bottleneck for grammatical error detection'. 

This repository contains forks of several other repositories, including:
1.	seq2seq - an encoder-decoder framework used for machine translation (https://github.com/marekrei/sequence-labeler)
2.	sequence-labeler - a grammatical error detection model (https://github.com/google/seq2seq/)
3.	RelGAN - a GAN for text generation (https://github.com/weilinie/RelGAN)

All licences, terms and conditions from these repositories apply. 

Changes have been made to the code in these repositories. A detailed list of the changes implemented can be found in the modifications.md file. This file covers all of the major code changes which were made. 

## Data ## 

To run the experiments, you will have to acquire the FCE and CoNLL 2014 datasets. 

## Running encoder-decoder model ## 

To generate artificial data using the encoder-decoder model, follow these steps:
1.	First train the encoder-decoder model. This is done by running the s2s-training.sh script in the seq2seq repository.
2.	Once this model has been run, one should run one of the scripts in the seq2seq repository for generating artificial data (e.g. s2s-generation-fce-am.sh). Choose the script according to the type of data augmentation strategy that you want to use. 
3.	Having generated the data, run the tsvutils.py file in the utils folder. Change the file name to the name of the file containing the generated data. This function will output labelled data. 
4.	Run the experiment.py file in the sequence-labeler repository. Modify the file names and hyperparameters in the configuration file as needed. 

## Notebooks ##

Various notebooks which were used to execute the code in the Google Colaboratory environment are included in this folder.


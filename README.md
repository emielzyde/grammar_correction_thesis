# Addressing the Data Scarcity Bottleneck for Grammatical Error Detection #

This repository contains the code for my thesis 'Addressing the Data Scarcity Bottleneck for Grammatical Error Detection'. 

This repository contains forks of several other repositories, including:
1.	seq2seq - an encoder-decoder framework used for machine translation (https://github.com/google/seq2seq/) 
2.	sequence-labeler - a grammatical error detection model (https://github.com/marekrei/sequence-labeler)
3.	wronging - combines the above two repositories and adds in utility files (https://github.com/skasewa/wronging) 
4.	RelGAN - implements a GAN for text generation (https://github.com/weilinie/RelGAN)
5.	probabilistic_nlg - implements the Wasserstein auto encoder (https://github.com/HareeshBahuleyan/probabilistic_nlg) 
6.	gpt-2-simple - allows us to retrain GPT-2 on new data (https://github.com/minimaxir/gpt-2-simple)

All licences, terms and conditions from these repositories apply. 

Changes have been made to the code in these repositories. A detailed list of the changes implemented can be found in the modifications.md file. This file covers all of the major code changes which were made. 

The folder labelled 'own_files' contains all of my own code which was written for the project. The purpose of each file in this folder has been described in the modifications.md file. 

## Data ## 

To run the experiments, you will have to acquire the FCE and CoNLL 2014 datasets. 

## Running encoder-decoder model ## 

To generate artificial data using the encoder-decoder model, follow these steps:
1.	First train the encoder-decoder model. This is done by running the s2s-training.sh script in the seq2seq repository.
2.	Once this model has been run, one should run one of the scripts in the seq2seq repository for generating artificial data (e.g. s2s-generation-fce-am.sh). Choose the script according to the type of data augmentation strategy that you want to use. 
3.	Having generated the data, run the tsvutils.py file in the utils folder. Change the file name to the name of the file containing the generated data. This function will output labelled data. 
4.	Run the experiment.py file in the sequence-labeler repository. Modify the file names and hyperparameters in the configuration file as needed. 

## Notebooks ##

Various notebooks which were used to execute the code in the Google Colaboratory environment are included in this folder. References to the owners of the code are included in the notebooks. 
1.	Label_Runner - runs the full error detection model
2. 	Label_Generator - runs the label generation code (generates labels for a specified data-set from a specified pre-trained model using self-training approaches)
3.	RelGAN_Runner - runs the RelGAN code 
4. 	WAE_determinstic - runs the standard determinstic Wasserstein autoencoder 
5.	WAE_conditional - runs the conditional Wasserstein autoencoder 


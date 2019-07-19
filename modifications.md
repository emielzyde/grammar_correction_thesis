## seq2seq ##

helper.py (in seq2seq - seq2seq - contrib - seq2seq) 
1.	Added in functionality to implement the original method presented in the encoder-decoder section 

beam_search.py (in seq2seq - seq2seq - inference)
1.	Added in implementation of random noising method 

## sequence-labeler ##

tsvutils.py (in wronging - utils)
1.	Added intermediate metrics to analyse the data (e.g. average number of incorrect tokens, percentage of incorrect tokens, corplus GLEU)
2.	Added in ability to specify the desired percentage of errors (rather than just the minimum and maximum number of errors) 

## Own files ## 

queryGoogle.py
1.	Used to query the Google Translate API. Before this file can be run, a console project and private key have to be set up (https://cloud.google.com/translate/docs/quickstart) 

xmlReader.py
1.	Used to read the XML data from the FCE dataset, to extract relevant attributes from this data 

import os
import re
import operator

VOCAB_SIZE =10000000 #can restrict the size of the vocab with this if needed
MAX_SEQ_LEN = 50

def process_conll(path):
    input_file = os.path.join(path)
    with open(input_file, 'r') as f:
        data = f.readlines()

    #Remove newline characters
    data= [s.strip() for s in data]
    #Remove tabs
    data = [s.split("\t") for s in data]

    #Set up lists of the sentences and corrections
    sentences = []
    corrections = []

    new_sentence = ""
    new_correction = []
    for item in data:
        if len(item) == 1:
            sentences.append(new_sentence)
            corrections.append(new_correction)
            new_sentence = ""
            new_correction = []
        else:
            new_sentence = new_sentence + item[0] + " "
            if item[1] == 'c':
                new_correction.append(0)
            elif item[1] == 'i':
                new_correction.append(1)
            else:
                print("Only 2 types are possible. ")

    sentences_double = []
    for sentence in sentences:
        new_sent = sentence + "\t" + sentence + "\n"
        sentences_double.append(new_sent)

    return sentences, corrections, sentences_double

if __name__ == '__main__':

    vocab = dict()

    input_file = os.path.join('/Users/emielzyde/Downloads/Project Data/fce-public.train.original.tsv')
    with open(input_file, 'r') as f:
        data = f.readlines()

    #Remove newline characters
    data= [s.strip() for s in data]
    #Remove tabs
    data = [s.split("\t") for s in data]


    words = []
    #Set up lists of the sentences and corrections
    sentences = []
    corrections = []
    counter = 0
    incorrect_counter = 0
    sent_counter = 0

    new_sentence = ""
    new_correction = []
    for item in data:
        if len(item) == 1:
            sentences.append(new_sentence)
            corrections.append(new_correction)
            new_sentence = ""
            new_correction = []
            sent_counter += 1
        else:
            if item[0] not in words:
                words.append(item[0])


            if item[0] not in vocab:
                vocab[item[0]] = 1
            else:
                vocab[item[0]] += 1

            new_sentence = new_sentence + item[0] + " "
            if item[1] == 'c':
                new_correction.append(0)
            elif item[1] == 'i':
                new_correction.append(1)
                incorrect_counter += 1
            else:
                print("Only 2 types are possible. ")
            counter += 1
    #print(sentences)
    #print(corrections)

    print(len(words))

    vocab_size = 0
    new_vocab = dict()
    sorted_x = sorted(vocab.items(), key=operator.itemgetter(1))
    sorted_x.reverse()
    counter = 0
    for key, item in vocab.items():
        if counter < VOCAB_SIZE:
            new_vocab[key] = item
        counter += 1

    input_file = os.path.join('/Users/emielzyde/Downloads/Project Data/fce-public.train.original.tsv')
    with open(input_file, 'r') as f:
        data = f.readlines()

    #Remove newline characters
    data= [s.strip() for s in data]
    #Remove tabs
    data = [s.split("\t") for s in data]

    words = []
    print(len(new_vocab))
    #Set up lists of the sentences and corrections
    sentences = []
    corrections = []
    counter = 0
    incorrect_counter = 0
    sent_counter = 0
    len_counter = 0

    new_sentence = ""
    new_correction = []
    for item in data:
        if len(item) == 1:
            sentences.append(new_sentence)
            corrections.append(new_correction)
            new_sentence = ""
            new_correction = []
            sent_counter += 1
            len_counter = 0
        else:
            if item[0] not in words:
                words.append(item[0])
            item[0] = re.sub(r"([?.!,¿])", r" \1 ", item[0])
            item[0] = re.sub(r'[" "]+', " ", item[0])
            item[0] = re.sub(r"[^a-zA-Z?.!,¿]+", " ", item[0])

            if item[0] not in new_vocab:
                item[0] = 'UNK'

            if len_counter <= MAX_SEQ_LEN:
                new_sentence = new_sentence + item[0] + " "
                if item[1] == 'c':
                    new_correction.append(0)
                elif item[1] == 'i':
                    new_correction.append(1)
                    incorrect_counter += 1
                else:
                    print("Only 2 types are possible. ")
            counter += 1
            len_counter += 1

    #print(len(sentences))

    incorrect_sent = 0
    for correction_item in corrections:
        if 1 in correction_item:
            incorrect_sent += 1

    writer = open('/Users/emielzyde/Downloads/fce_train_new.txt', 'w')
    for i in range(len(sentences)):
        sentence = sentences[i]
        writer.write(sentences[i] + "\n")

    writer.close()
    print('Number of sentences: ', sent_counter)
    print('Number of incorrect sentences: ', incorrect_sent)
    print('Number of tokens: ', counter)
    print('Number of incorrect tokens: ', incorrect_counter)







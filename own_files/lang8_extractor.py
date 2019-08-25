import os
import pickle

if __name__ == '__main__':
    input_file = os.path.join('/Users/emielzyde/Downloads/lang-8-en-1.0/entries.test')
    with open(input_file, 'r') as f:
        data = f.readlines()
    splitter = []
    for i in range(len(data)):
        splitter.append(data[i].split('\t'))

    clean_sentences = []
    noisy_sentences = []

    incorrect_sentences_counter = 0
    correct_sentences_counter = 0
    multiple_corrections_counter = 0
    for line in splitter:
        if len(line) >= 4:
            if len(line) > 6:
                count = len(line) - 6
                for j in range(count):
                    noisy_sentences.append(line[6+j].rstrip('\n'))
                    clean_sentences.append(line[4].rstrip('\n'))
                multiple_corrections_counter += 1
            try:
                noisy_sentences.append(line[5].rstrip('\n'))
                clean_sentences.append(line[4].rstrip('\n'))
                incorrect_sentences_counter += 1
            except:
                noisy_sentences.append(line[4].rstrip('\n'))
                clean_sentences.append(line[4].rstrip('\n'))
                correct_sentences_counter += 1

    print('Number of correct sentences: {}'.format(correct_sentences_counter))
    print('Number of incorrect sentences: {}'.format(incorrect_sentences_counter))
    print('Number of sentences with multiple corrections: {}'.format(multiple_corrections_counter))
    print(len(clean_sentences))
    full_data = []
    file = open('lang8-correct.txt', 'w')

    for i in range(len(clean_sentences)):
        file.write(clean_sentences[i] + '\n')
        #full_data.append(clean_sentences[i] + '\t' + noisy_sentences[i] +'\n')
    file.close()
    #pickle.dump(full_data, open('lang8_preprocess.pickle', 'wb'))

    file = open('lang8-incorrect.txt', 'w')

    for i in range(len(noisy_sentences)):
        file.write(noisy_sentences[i] + '\n')
        #full_data.append(clean_sentences[i] + '\t' + noisy_sentences[i] +'\n')
    file.close()

    with open('/Users/emielzyde/Downloads/lang8-incorrect.txt','r') as f1:
        sents = f1.readlines()

    print(len(clean_sentences))
    print(len(noisy_sentences))






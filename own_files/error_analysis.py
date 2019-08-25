import os
from collections import defaultdict

import matplotlib.pyplot as plt
if __name__ == '__main__':
    vocab = dict()

    input_file = os.path.join('/Users/emielzyde/Downloads/Correct analysis/correct_test3.tsv')
    with open(input_file, 'r') as f:
        data = f.readlines()

    data= [s.strip() for s in data]
    data = [s.split("\t") for s in data]

    #Set up lists of the sentences and corrections
    incorrect_holder = []
    len_holder = []
    corrections = []
    counter = 0
    incorrect_counter = 0
    sent_counter = 0

    sentences = []
    new_counter = 0
    len_sent = 0
    new_correction = []
    new_sentence = ""

    error_matcher = 0
    error_match_list = []

    for item in data:
        if len(item) == 1:
            corrections.append(new_correction)
            len_holder.append(len_sent)
            sentences.append(new_sentence)
            incorrect_holder.append(new_counter/len_sent)
            if new_counter > 0:
                error_match_list.append(error_matcher/new_counter)
            sent_counter += 1
            len_sent = 0
            new_sentence = ""
            new_counter = 0
            error_matcher = 0

        else:
            new_sentence = new_sentence + item[0] + " "
            len_sent += 1
            if item[2] == 'c':
                new_correction.append(0)
            elif item[2] == 'i':
                new_correction.append(1)
                incorrect_counter += 1
                new_counter += 1
            else:
                print("Only 2 types are possible. ")

            if item[2] == 'i' and item[1] == 'c':
                error_matcher += 1
            elif item[2] == 'c' and item[1] == 'i':
                error_matcher += 1
            counter += 1

    print('Average incorrect: ', sum(incorrect_holder)/len(incorrect_holder))
    print('Average len: ', sum(len_holder)/len(len_holder))
    print('Average errors matched: ', sum(error_match_list)/len(error_match_list))
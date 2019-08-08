import Levenshtein
import numpy as np

#The PEN_GAP and PEN_MISMATCH parameters can be tuned if we want to punish one or the other mroe
PEN_MISMATCH = 1
PEN_GAP = 1
GAP = '<GAP>'

def levenshtein_calculator(source_sent, target_sent):

    #Split both sentences on the spaces
    source_sent = source_sent.split()
    target_sent = target_sent.split()

    source_len, target_len = len(source_sent), len(target_sent)

    #Set up the matrix to hold the values.
    distance_matrix = np.zeros([source_len +1, target_len+1])

    #If the sentences are the same, we are finished.
    if source_sent == target_sent:
        #print('The Levenshtein distance between the two sentences is 0.')
        return distance_matrix

    for i in range(source_len+1):
        distance_matrix[i][0] = i * PEN_GAP
    for j in range(target_len+ 1):
        distance_matrix[0][j] = j * PEN_GAP

    for i in range(source_len):
        for j in range(target_len):
            #If the tokens are equal, there is no penalty
            if source_sent[i] == target_sent[j]:
                pen_ij = 0
            #If they are different, we pay the mismatch penalty.
            else:
                pen_ij = PEN_MISMATCH
            #The new entry in the distance matrix is the lowest cost (equivalent to shortest path)
            distance_matrix[i+1][j+1] = min(distance_matrix[i][j+1] + PEN_GAP,
                                            distance_matrix[i+1][j] + PEN_GAP,
                                            distance_matrix[i][j] + pen_ij)

    #print('The Levenshtein distance between the two sentences is {}'.format(distance_matrix[-1][-1]))
    return distance_matrix

def sentence_labeller(source_sent, target_sent):

    #This code has been adapted from the code for the Kasewa et al. paper (essentially their post-processing routine
    #to label each token in a sentence as correct or incorrect).

    distance_matrix = levenshtein_calculator(source_sent, target_sent)

    #Split both sentences on the spaces
    source_sent = source_sent.split()
    target_sent = target_sent.split()

    source_len, target_len = len(source_sent) + 1 , len(target_sent) + 1

    #Start from the end of the source and target sentences
    i = source_len - 1
    j = target_len - 1

    #Set up lists to hold the modified sentences (where we add GAP tokens)
    source_sent_modified = []
    target_sent_modified = []

    while i > 0 and j > 0:

        if distance_matrix[i,j] == distance_matrix[i, j-1] + PEN_GAP: #Gap in source sentence
            target_sent_modified += [target_sent[j-1]] #add GAP token to source sentence
            source_sent_modified += [GAP]
            j -= 1

        elif distance_matrix[i,j] == distance_matrix[i-1,j] + PEN_GAP: #Gap in the target sentence
            target_sent_modified += [GAP] #add GAP token to target sentence
            source_sent_modified += [source_sent[i-1]]
            i -= 1

        else: #Tokens are equal or mismatched, but there is no gap
            target_sent_modified += [target_sent[j-1]]
            source_sent_modified += [source_sent[i-1]]
            i -= 1
            j -= 1

    #Currently, the modified source and target sentences are in reverse so, by reversing them, they will be in the correct order.
    source_sent_modified.reverse()
    target_sent_modified.reverse()

    #If the number of GAP tokens differ, the modified sentence lengths will be different. This is corrected by adding
    #additional GAP tokens.
    if i > 0: #more GAP tokens in target sentence
        target_sent_modified = [GAP] * i + target_sent_modified
        source_sent_modified = source_sent[0:i] + source_sent_modified

    elif j > 0: #more GAP tokens in source sentence
        target_sent_modified = target_sent[0:j] + target_sent_modified
        source_sent_modified = [GAP] * j + source_sent_modified

    target_sent_modified.reverse()
    target_sent_modified_copy = target_sent_modified.copy()

    #Removes any trailing GAPs from the target sentence.
    for i, token in enumerate(target_sent_modified_copy):
        if token == GAP:
            del target_sent_modified[0]
            continue
        else:
            break
    target_sent_modified.reverse()

    #Here, the string of correct/incorrect labels is constructed as per the criteria in the paper.
    evaluation_list = []
    for i, target_token in enumerate(target_sent_modified):
        source_token = source_sent_modified[i]
        if target_token == GAP:
            continue
        #If the tokens are different, label it 'incorrect'
        if source_token != target_token:
            evaluation_list.append(0)
        # If the token is the last word in the sentence but it is not aligned with the last word in the source sentence, label it 'incorrect'
        elif i == len(target_sent_modified) - 1 and len(source_sent_modified) > len(target_sent_modified):
            evaluation_list.append(0)
        #If the tokens are the same but there was a preceding gap, label it 'incorrect'
        elif i > 0 and target_sent_modified[i-1] == GAP:
            evaluation_list.append(0)
        #Otherwise, label it correct
        else:
            evaluation_list.append(1)

    return evaluation_list

if __name__ == '__main__':
    string1 = 'This is not the best'
    string2 = 'This is the best'

    a = levenshtein_calculator(string1, string2)
    print(sentence_labeller(string1, string2, a))

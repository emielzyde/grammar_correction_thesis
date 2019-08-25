import os
import random
import nltk
nltk.download('stopwords')
from nltk.tokenize import word_tokenize
import gensim
from gensim.matutils import softcossim
from gensim import corpora
import gensim.downloader as api

print('Loading model...')
#model = gensim.models.KeyedVectors.load_word2vec_format('/Users/emielzyde/Downloads/GoogleNews-vectors-negative300.bin',
                                                        #binary=True)
fasttext_model300 = api.load('fasttext-wiki-news-subwords-300')
print('Model loaded!')

def overlap_measure(data, threshold):
    repetition_counter = 0
    data = [s.strip() for s in data]
    data = [s.lower() for s in data]
    sample_splitter = [s.split(" ") for s in data]
    print(len(sample_splitter))

    for i in range(len(sample_splitter)):
        if i % 10 == 0:
            print(i)
        for j in range(len(sample_splitter)):
            if i == j:
                continue
            overlap_counter = 0
            source_sentence = sample_splitter[i]
            target_sentence = sample_splitter[j]
            len_sent = len(source_sentence)
            for item in source_sentence:
                for item2 in target_sentence:
                    if item == item2:
                        overlap_counter += 1
                        break

            if overlap_counter/len_sent > threshold:
                repetition_counter += 1
                break

    print('Done')
    print('Repetition counter: ', repetition_counter)
    print('Repetition percentage: ', repetition_counter/len(sample_splitter))

def soft_cosine_similarity(data_source, data_target):
    data_source = [s.strip() for s in data_source]
    data_source = [s.lower() for s in data_source]
    data_source = [s.split() for s in data_source]

    data_target = [s.strip() for s in data_target]
    data_target = [s.lower() for s in data_target]
    data_target = [s.split() for s in data_target]

    random.shuffle(data_source)

    overall_data = data_source + data_target
    assert len(overall_data) == len(data_source) + len(data_target), 'Lengths should be equal'

    #dictionary = corpora.Dictionary(data_source)
    dictionary = corpora.Dictionary(overall_data)
    print('Making similarity matrix')
    similarity_matrix = fasttext_model300.similarity_matrix(dictionary, tfidf = None, threshold = 0.0, exponent = 2.0, nonzero_limit = 100)
    overlap_list = []
    same_counter = 0

    target_bow_list = []
    print('Processing target data')
    for k in range(len(data_target)):
        if k% 100 == 0:
            print(k)
        target_sent = data_target[k]
        target_bow = dictionary.doc2bow(target_sent)
        target_bow_list.append(target_bow)

    for i in range(1000):
        print('Iteration ', i)
        max_overlap = -100
        overlapper = 0
        source_sentence = data_source[i]
        source_bow = dictionary.doc2bow(source_sentence)
        for j in range(len(data_target)):
            #if i == j:
            #    continue
            target_bow = target_bow_list[j]
            distance = softcossim(source_bow, target_bow, similarity_matrix)
            if distance == 1:
                same_counter += 1
            if distance > max_overlap:
                max_overlap = distance
                overlapper = j
        overlap_list.append(max_overlap)
        print('Source: ', data_source[i])
        print('Closest sentence: ', data_target[overlapper])
        print('Overlap: ', max_overlap)
        print('Perfect matches: ', same_counter)

    avg_overlap = sum(overlap_list)/len(overlap_list)
    print(overlap_list)
    print('Average distance: ', avg_overlap)

def word_mover_distance(data_source, data_target):

    data_source = [s.strip() for s in data_source]
    data_source = [s.lower() for s in data_source]

    data_target = [s.strip() for s in data_target]
    data_target = [s.lower() for s in data_target]

    random.shuffle(data_source)
    overlap_list = []

    for i in range(200):
        print('Iteration ', i)
        min_overlap = 10000
        overlapper = 10000
        source_sentence = data_source[i].split()
        for j in range(len(data_target)):
            if j == i:
                continue
            target_sentence = data_target[j].split()
            distance = model.wmdistance(source_sentence, target_sentence)
            if distance < min_overlap:
                min_overlap = distance
                overlapper = j
        overlap_list.append(min_overlap)
        print('Source: ', data_source[i])
        print('Closest sentence: ', data_target[overlapper])
        print('Overlap:', overlapper)

    avg_overlap = sum(overlap_list)/len(overlap_list)
    print(overlap_list)
    print('Average distance: ', avg_overlap)

if __name__ == '__main__':
    input_file1 = '/Users/emielzyde/Downloads/wae_noisy_samples_new.txt'
    input_file2 = '/Users/emielzyde/Downloads/wronging/data/train/fce/targets.txt'

    with open(input_file1, 'r') as f:
        samples_source = f.readlines()

    with open(input_file2, 'r') as f:
        samples_target = f.readlines()
        
    #overlap_measure(samples, 0.75)
    #word_mover_distance(samples_source, samples_target)
    soft_cosine_similarity(samples_source, samples_target)
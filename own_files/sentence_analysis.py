import matplotlib.pyplot as plt
import numpy as np
import statistics

tableau20 = [(31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),
             (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),
             (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),
             (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),
             (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]

for i in range(len(tableau20)):
    r, g, b = tableau20[i]
    tableau20[i] = (r / 255., g / 255., b / 255.)

def error_analysis():
    input_file = '/Users/emielzyde/Downloads/Project Data/nucle.test1.original.tsv'
    with open(input_file, 'r') as f:
        data = f.readlines()

    data = [s.strip() for s in data]
    data = [s.split("\t") for s in data]

    input_file2 = '/Users/emielzyde/Downloads/Project Data/nucle.test0.original.tsv'
    with open(input_file2, 'r') as f:
        data2 = f.readlines()

    data2 = [s.strip() for s in data2]
    data2 = [s.split("\t") for s in data2]
    data_new = data + data2



    incorrect_list = []
    in_list = []
    incorrect_count = 0
    len_sent = 0

    over_50 = 0
    over_25 = 0
    wrong_counter = 0
    counter = 0
    incorrect_token_counter = 0
    token_counter = 0

    for item in data_new:
        if len(item) == 1:
            counter += 1
            if incorrect_count > 0:
                incorrect_list.append(incorrect_count/len_sent)
                in_list.append(incorrect_count)
                wrong_counter +=1

            if incorrect_count/len_sent > 0.5:
                over_50 += 1
            if incorrect_count/len_sent > 0.25:
                over_25 += 1
            incorrect_count = 0
            len_sent = 0

        else:
            token_counter += 1
            len_sent += 1
            if item[1] == 'i':
                incorrect_count += 1
                incorrect_token_counter += 1

    print('Number of sentences: ', counter)
    print('Percentage of incorrect sentences: ', wrong_counter/counter)
    print('Percentage of incorrect tokens: ', incorrect_token_counter/token_counter)
    print('Over 50%: ', over_50/len(incorrect_list))
    print('Over 25%: ', over_25/len(incorrect_list))

    print('Median error percentage: ', statistics.median(incorrect_list))
    print('Mean error percentage: ', statistics.mean(incorrect_list))

    print()
    print('Median number of errors: ', statistics.median(in_list))
    print('Mean number of errors: ', statistics.mean(in_list))

    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    # Common sizes: (10, 7.5) and (12, 9)
    plt.figure(figsize=(10, 6))

    ax = plt.subplot(111)
    ax.spines["top"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)

    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

    plt.setp((ax), xticks=[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1])

    #ticker1 = [0.05,0.1,0.15,0.2,0.25,0.3,0.35]
    ticker1 = [0.02,0.04,0.06,0.08,0.1,0.12]
    ranger = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
    for y in ticker1:
        ax.plot(ranger, [y] * len(ranger), "--", lw=0.5, color="black",
                 alpha=0.3)

    ax.hist(incorrect_list, weights=np.zeros_like(incorrect_list) + 1. / np.array(incorrect_list).size, bins = 50, color = tableau20[0], edgecolor = 'black')
    #ax.hist(x, n_bins, density=True, histtype='step',
    #        cumulative=True, label='Empirical')
    #ax.hist(incorrect_list, cumulative = True, bins = 100)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    plt.xlabel("Percentage of tokens labelled as incorrect", fontsize=16)
    plt.ylabel("Relative frequency", fontsize=16)

    plt.show()

def len_analysis():
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    input_file = '/Users/emielzyde/Downloads/fce_test.txt'
    with open(input_file, 'r') as f:
        data = f.readlines()
    data= [s.strip() for s in data]
    data= [s.split() for s in data]

    len_list = []

    max_len = 0
    for sent in data:
        length = len(sent)
        len_list.append(length)
        if length > max_len:
            max_len = length

    median_len = statistics.median(len_list)
    mean_len = statistics.mean(len_list)
    len_list.sort(reverse = True)
    print(median_len)
    print(mean_len)



    # Common sizes: (10, 7.5) and (12, 9)
    plt.figure(figsize=(10, 6))
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    ax1 = plt.subplot(111)
    ax1.spines["top"].set_visible(False)
    ax1.spines["bottom"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.spines["left"].set_visible(False)

    ax1.get_xaxis().tick_bottom()
    ax1.get_yaxis().tick_left()

    ticker1 = [0.02,0.04,0.06,0.08,0.1]
    for y in ticker1:
        ax1.plot(range(1, max_len+1), [y] * len(range(1, max_len+1)), "--", lw=0.5, color="black",
                 alpha=0.3)

    ax1.hist(len_list, weights=np.zeros_like(len_list) + 1. / np.array(len_list).size, bins = 50, color = tableau20[0], edgecolor = 'black')
    ax1.set_ylabel("Relative frequency", fontsize=13)
    ax1.set_xlabel("Sentence length", fontsize=13)

    #ranger = [0,0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09]

    #ax1.plot([median_len] * len(ranger),ranger, "--", lw=1, color= tableau20[0], alpha=0.3)

    #ax2 = plt.subplot(122)
    #ax2.spines["top"].set_visible(False)
    #ax2.spines["right"].set_visible(False)
    #ax2.get_xaxis().tick_bottom()
    #ax2.get_yaxis().tick_left()
    #ax2.set_xlim([0, 100])
    #ax2.hist(len_list, weights=np.zeros_like(len_list) + 1. / np.array(len_list).size, bins = 200, color = tableau20[0])
    #ax2.set_xlabel("Sentence length", fontsize=13)

    plt.show()

if __name__ == '__main__':
    error_analysis()



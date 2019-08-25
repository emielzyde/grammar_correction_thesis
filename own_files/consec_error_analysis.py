import os
import collections
from collections import defaultdict
import matplotlib.pyplot as plt

if __name__ == '__main__':
    input_file = os.path.join('/Users/emielzyde/Downloads/Wrong analysis/wrong_test1.tsv')
    input_file_correct = os.path.join('/Users/emielzyde/Downloads/Correct analysis/correct_test1.tsv')

    with open(input_file, 'r') as f:
        data = f.readlines()
    data= [s.strip() for s in data]
    data = [s.split("\t") for s in data]

    with open(input_file_correct, 'r') as f:
        data_correct = f.readlines()
    data_correct = [s.strip() for s in data_correct]
    data_correct = [s.split("\t") for s in data_correct]

    consecutive_dict = defaultdict(list)

    consecutive_error_holder = []
    consec_errors_matched = []
    consecutive_count = 0
    consec_error_count = 0
    new_sentence = ""
    consecutive = True

    for item in data:
        if consecutive is False and consecutive_count > 0:
            consecutive_error_holder.append(consecutive_count)
            consec_errors_matched.append(consec_error_count/consecutive_count)

            if consecutive_count in consecutive_dict:
                consecutive_dict[consecutive_count].append(consec_error_count)
            else:
                consecutive_dict[consecutive_count] = []
                consecutive_dict[consecutive_count].append(consec_error_count)
            consecutive = True
            consecutive_count = 0
            consec_error_count = 0

        if len(item) == 1:
            new_sentence = ""
            if consecutive_count > 0:
                consecutive_error_holder.append(consecutive_count)
                consec_errors_matched.append(consec_error_count / consecutive_count)

                if consecutive_count in consecutive_dict:
                    consecutive_dict[consecutive_count].append(consec_error_count)
                else:
                    consecutive_dict[consecutive_count] = []
                    consecutive_dict[consecutive_count].append(consec_error_count)
            consecutive = True
            consecutive_count = 0
            consec_error_count = 0

        else:
            new_sentence += " " + item[0]
            if item[2] == 'c':
                consecutive = False
            elif item[2] == 'i':
                consecutive = True
                consecutive_count += 1
                if item[1] == 'c':
                    consec_error_count += 1
            else:
                print("Only 2 types are possible. ")

    #Correct data
    consecutive_error_holder = []
    consec_errors_matched = []
    consecutive_count = 0
    consec_error_count = 0
    new_sentence = ""
    consecutive = True

    for item in data_correct:
        if consecutive is False and consecutive_count > 0:
            consecutive_error_holder.append(consecutive_count)
            consec_errors_matched.append(consec_error_count/consecutive_count)

            if consecutive_count in consecutive_dict:
                consecutive_dict[consecutive_count].append(consec_error_count)
            else:
                consecutive_dict[consecutive_count] = []
                consecutive_dict[consecutive_count].append(consec_error_count)
            consecutive = True
            consecutive_count = 0
            consec_error_count = 0

        if len(item) == 1:
            new_sentence = ""
            if consecutive_count > 0:
                consecutive_error_holder.append(consecutive_count)
                consec_errors_matched.append(consec_error_count / consecutive_count)

                if consecutive_count in consecutive_dict:
                    consecutive_dict[consecutive_count].append(consec_error_count)
                else:
                    consecutive_dict[consecutive_count] = []
                    consecutive_dict[consecutive_count].append(consec_error_count)
            consecutive = True
            consecutive_count = 0
            consec_error_count = 0

        else:
            new_sentence += " " + item[0]
            if item[2] == 'c':
                consecutive = False
            elif item[2] == 'i':
                consecutive = True
                consecutive_count += 1
                if item[1] == 'c':
                    print('Error')
                    consec_error_count += 1
            else:
                print("Only 2 types are possible. ")

    consecutive_dict = collections.OrderedDict(sorted(consecutive_dict.items()))
    keys = []
    vals = []
    new_vals = []
    for key in consecutive_dict:
        print(key)
        lister = consecutive_dict[key]
        print(len(lister))
        print((sum(lister)/len(lister))/key)
        keys.append(key)
        vals.append((sum(lister)/len(lister))/key)

        if key == 1:
            new_vals.append((sum(lister)/len(lister))/key)
    print('Val: ', sum(new_vals)/len(new_vals))


    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    tableau20 = [(31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),
                 (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),
                 (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),
                 (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),
                 (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]

    # Scale the RGB values to the [0, 1] range, which is the format matplotlib accepts.
    for i in range(len(tableau20)):
        r, g, b = tableau20[i]
        tableau20[i] = (r / 255., g / 255., b / 255.)

    plt.figure(figsize=(6.5, 4))
    ax = plt.subplot(111)
    ax.spines["top"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)


    y_plotter = [0.6, 0.65,0.7,0.75,0.8,0.85,0.9,0.95,1]
    plt.yticks(y_plotter, fontsize=8)
    for y in y_plotter:
        plt.plot(range(0, keys[-1] + 1), [y] * len(range(0, keys[-1]+1)), "--", lw=0.5, color="black", alpha=0.3)

    plt.xlabel('Number of consecutive incorrect tokens', fontsize = 10)
    plt.ylabel('Proportion of errors', fontsize = 10)

    plt.plot(keys, vals, color = tableau20[0])

    plt.show()
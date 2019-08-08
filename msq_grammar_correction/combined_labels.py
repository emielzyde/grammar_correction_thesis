import os

if __name__ == '__main__':

    input_file_1 = os.path.join('/Users/emielzyde/Desktop/Datasets/beam_ORIGINAL.tsv')
    with open(input_file_1, 'r') as f:
        data_original = f.readlines()
    #Remove newline characters
    data_original = [s.strip() for s in data_original]
    #Remove tabs
    data_original = [s.split("\t") for s in data_original]

    input_file_2 = os.path.join('/Users/emielzyde/Downloads/beam_consensus_25.tsv')
    with open(input_file_2, 'r') as f:
        data_self = f.readlines()
    #Remove newline characters
    data_self = [s.strip() for s in data_self]
    #Remove tabs
    data_self = [s.split("\t") for s in data_self]

    writer = open('/Users/emielzyde/Downloads/beam_consensus_selftrain_25.tsv', 'w')

    correct_changer = 0
    incorrect_changer = 0
    incorrect_count = 0
    for item in range(len(data_original)):
        if len(data_original[item]) == 1:
            writer.write("\n")
        else:
            assert data_original[item][0] == data_self[item][0], 'Tokens should be the same'
            item_original = data_original[item][1]
            item_self = data_self[item][1]

            if item_self == 'c':
                writer.write(data_original[item][0] + '\t' + 'c' + '\n')
                if item_original != 'c':
                    correct_changer += 1
            elif item_self == 'i':
                writer.write(data_original[item][0] + '\t' + 'i' + '\n')
                incorrect_count += 1
                if item_original != 'i':
                    incorrect_changer += 1
            elif item_self == 'u':
                 writer.write(data_original[item][0] + '\t' + item_original + '\n')
                 if item_original == 'i':
                     incorrect_count += 1
            else:
                print("Incorrect type.")

    print('Changed tokens: ', correct_changer + incorrect_changer)
    print('Incorrect changed: ', incorrect_changer)
    print('Correct changed: ', correct_changer)
    print('Incorrect tokens: ', incorrect_count)

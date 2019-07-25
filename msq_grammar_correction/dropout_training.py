import os

def process_files_consensus(path_list, num_paths = 2):
    data_lister = []
    for i in range(len(path_list)):
        input_file = os.path.join(path_list[i])
        with open(input_file, 'r') as f:
            data = f.readlines()
        # Remove newline characters
        data = [s.strip() for s in data]
        # Remove tabs
        data = [s.split("\t") for s in data]
        data_lister.append(data)

    #Set up lists of the sentences and corrections
    writer = open('/Users/emielzyde/Downloads/dropout_writer.txt', 'w')
    wait_till_space = False
    data_sourcer = data_lister[0]

    sentence_corrections = []
    correction_holder = []
    sentence_holder = []

    for i in range(len(data_sourcer)):

        for j in range(len(data_lister)):
            lister = data_lister[j]
            item = lister[i]
            if len(item) == 1:
                if wait_till_space:
                   do_nothing = 1
                else:
                    for k in range(len(sentence_corrections)):
                        writer.write(sentence_holder[k] + "\t" + sentence_corrections[k] + "\n")
                    writer.write("\n")

                correction_holder = []
                sentence_corrections = []
                sentence_holder = []
                wait_till_space = False
                break
            elif wait_till_space:
                break
            else:
                if j == 0:
                    sentence_holder.append(item[0])
                if item[1] == 'c':
                    correction_holder.append(0)
                elif item[1] == 'i':
                    correction_holder.append(1)
                else:
                    print("Only 2 types are possible. ")

        #Consensus vote
        if sum(correction_holder) != num_paths and sum(correction_holder) != 0:
            wait_till_space = True
        elif not wait_till_space:
            if sum(correction_holder) == 0 and len(correction_holder) > 0:
                sentence_corrections.append('c')
            elif sum(correction_holder) == num_paths:
                sentence_corrections.append('i')
            correction_holder = []

    writer.close()

def process_files(path_list):
    data_lister = []
    for i in range(len(path_list)):
        input_file = os.path.join(path_list[i])
        with open(input_file, 'r') as f:
            data = f.readlines()
        # Remove newline characters
        data = [s.strip() for s in data]
        # Remove tabs
        data = [s.split("\t") for s in data]
        data_lister.append(data)

    #Set up lists of the sentences and corrections
    writer = open('/Users/emielzyde/Downloads/dropout_writer.txt', 'w')
    writer_ind = True
    data_sourcer = data_lister[0]
    for i in range(len(data_sourcer)):
        correction_holder = []
        sentence_holder = []
        for lister in data_lister:
            item = lister[i]
            if len(item) == 1:
                writer.write("\n")
                writer_ind = False
                break
            else:
                sentence_holder.append(item[0])
                if item[1] == 'c':
                    correction_holder.append(0)
                elif item[1] == 'i':
                    correction_holder.append(1)
                else:
                    print("Only 2 types are possible. ")

        #Majority vote
        if writer_ind:
            if sum(correction_holder)/len(correction_holder) > 0.5:
                writer.write(item[0] + "\t" + "i" + "\n")
            else:
                writer.write(item[0] + "\t" + "c" + "\n")
        writer_ind = True

    writer.close()
if __name__ == '__main__':

    path_list = ['/Users/emielzyde/Desktop/Datasets/japanese_normal.tsv', '/Users/emielzyde/Desktop/Datasets/japanese_normal_2.tsv']
    process_files_consensus(path_list)


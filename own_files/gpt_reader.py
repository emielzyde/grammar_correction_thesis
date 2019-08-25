import os
import random

if __name__ == '__main__':
    input_file = os.path.join('/Users/emielzyde/Downloads/conll_gpt_generated.txt')
    with open(input_file, 'r') as f:
        data = f.readlines()

    removal_counter = 0
    new_data = []
    for line in data:
        if line == '====================\n':
            removal_counter += 1
        else:
            if line != "\n" and len(line.split()) > 8:
                new_data.append(line)

    random.shuffle(new_data)
    print(removal_counter)
    writer = open('/Users/emielzyde/Downloads/conll_gpt_generated_full.txt','w')
    write_counter = 0
    for line in new_data:
        if write_counter < 60000:
            writer.write(line)
            write_counter += 1
        else:
            break

    writer.close()




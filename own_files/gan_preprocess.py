import os
import re

if __name__ == '__main__':
    input_file = os.path.join('/Users/emielzyde/Downloads/wronging/data/dev/fce/sources.txt')
    with open(input_file, 'r') as f:
        data = f.readlines()

    new_data = []
    for sent in data:
        sent = re.sub(r'[^\w\s\?\.\,]', '', sent.strip().lower()) 
        sent = re.sub(r'(([a-z]*)\d+.?\d*\%?)', ' NUM ', sent.strip()) 

        if len(sent.strip()) > 50:
            sent_new = sent.split()
            sent_new = sent_new[:50]
            senter = ""
            for word in sent_new:
                senter += word + " "
            sent = senter
        new_data.append(sent)

    writer = open('/Users/emielzyde/Downloads/fce_test_new.txt', 'w')
    for i in range(len(new_data)):
        writer.write(new_data[i] + "\n")
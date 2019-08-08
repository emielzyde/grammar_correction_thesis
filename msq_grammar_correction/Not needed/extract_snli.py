import json

if __name__ == '__main__':
    output_file = open('/Users/emielzyde/Downloads/snli_out.txt', 'w')
    counter = 0
    with open('/Users/emielzyde/Downloads/snli_1.0/snli_1.0_train.json', 'r') as json_file:
        for line in json_file:
            data = json.loads(line)
            output_file.write(data['sentence1'] + '\n')
            output_file.write(data['sentence2'] + '\n')
            counter += 1

    print(counter)
    output_file.close()
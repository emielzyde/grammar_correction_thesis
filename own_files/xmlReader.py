import glob, os
from xml.etree import ElementTree as ET

if __name__ == '__main__':
    counter1 = 0
    counter2 = 0
    languages = dict()
    for root, dirs, files in os.walk("/Users/emielzyde/Downloads/fce-released-dataset/dataset"):
        for file in files:
            if file.endswith(".xml"):
                counter1 += 1
                input_file = os.path.join(root, file)

                with open(input_file, 'r') as f:
                    data = f.readlines()
                stringer = ""
                for line in data:
                    stringer += line
                root = ET.fromstring(stringer)
                levels = root.findall('.//personnel')
                for level in levels:
                    lang = level.find('language').text
                    if lang in languages:
                        languages[lang] += 1
                        counter2 += 1
                    else:
                        languages[lang] = 0
                        counter2+= 1

    sorted_x = sorted(languages.items(), key=lambda kv: kv[1])
    print(sorted_x)
    assert counter1 == counter2, 'Counts should be equal'

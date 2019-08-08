import os

if __name__ == '__main__':
    from google.cloud import translate

    # Instantiates a client
    translate_client = translate.Client()

    input_file = os.path.join('/Users/emielzyde/Desktop/msq-grammar-correction/data/train/fce/targets.txt')
    with open(input_file, 'r') as f:
        data = f.readlines()

    data= [s.strip() for s in data]
    # The text to translate
    text = u'Hello, world!'
    # The target language
    target = 'ja'
    print('Target: ', target)

    translated_texts = []
    # Translates some text into Russian
    print('Translating')
    for i in range(len(data)):
        if i%1000 == 0:
            print('Item ', i)
        translation = translate_client.translate(data[i], target_language=target)
        translated_texts.append(translation['translatedText'])

    print('Translating back')
    round_translated_texts = []
    for i in range(len(translated_texts)):
        if i%1000 == 0:
            print('Item ', i)
        translation = translate_client.translate(translated_texts[i], target_language = 'en')
        round_translated_texts.append(translation['translatedText'])

    writer = open('/Users/emielzyde/Downloads/round_translation_japanese.txt', 'w')
    for i in range(len(data)):
        #print(u'Text: {}'.format(data[i]))
        #print(u'Translation: {}'.format(round_translated_texts[i]))
        writer.write(round_translated_texts[i] + '\n')
    writer.close()

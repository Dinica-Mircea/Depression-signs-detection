from googletrans import Translator


def translateFromEnToRo(text):
    translator = Translator()
    return translator.translate(text=text, src='en', dest='ro').text


def translateDataset(pathToEnDataset, pathToRoDataset):
    fileInput = open(pathToEnDataset, 'r')
    lines = fileInput.readlines()
    translatedLines = []
    for line in lines[1:]:
        text = line[:line.__len__() - 3]
        label = line[line.__len__() - 3:]
        translatedText = translateFromEnToRo(text)
        translatedLines.append(translatedText + label)

    with open(pathToRoDataset, 'w',encoding='utf-8') as fileOutput:
        for line in translatedLines:
            fileOutput.write(line)


if __name__ == '__main__':
    pathToEnDataset = 'dataset/raw/English/depression_dataset_reddit_cleaned.csv'
    pathToRoDataset = 'dataset/raw/Romanian/depression_dataset_reddit_cleaned.csv'
    #translateDataset(pathToEnDataset, pathToRoDataset)

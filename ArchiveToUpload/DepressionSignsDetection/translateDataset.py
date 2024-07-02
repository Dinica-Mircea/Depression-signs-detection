from googletrans import Translator


def translateFromEnToRo(text):
    translator = Translator()
    return translator.translate(text=text, src='en', dest='ro').text


def translateDataset(pathToEnglishDatasetInput, pathToRomanianDatasetOutput):
    fileInput = open(pathToEnglishDatasetInput, 'r')
    lines = fileInput.readlines()
    translatedLines = []
    for line in lines[1:]:
        text = line[:line.__len__() - 3]
        label = line[line.__len__() - 3:]
        translatedText = translateFromEnToRo(text)
        translatedLines.append(translatedText + label)

    with open(pathToRomanianDatasetOutput, 'w', encoding='utf-8') as fileOutput:
        for line in translatedLines:
            fileOutput.write(line)


def changeSeparator(pathToRoDataset, pathToChangedRoDataset):
    fileInput = open(pathToRoDataset, 'r', encoding='utf-8')
    lines = fileInput.readlines()
    changedText = []
    for line in lines:
        text = line[:line.__len__() - 3]
        text = text.replace(",", "")
        label = line[line.__len__() - 2:]
        changedText.append(text + "," + label)

    with open(pathToChangedRoDataset, 'w', encoding='utf-8') as fileOutput:
        for line in changedText:
            fileOutput.write(line)


def removeDoubleQoute(path, pathToChanged):
    fileInput = open(path, 'r', encoding='utf-8')
    lines = fileInput.readlines()
    changedText = []
    for line in lines:
        text = line.replace('"', '')
        changedText.append(text)

    with open(pathToChanged, 'w', encoding='utf-8') as fileOutput:
        for line in changedText:
            fileOutput.write(line)


if __name__ == '__main__':
    pathToEnDataset = 'dataset/raw/English/depression_dataset_reddit_cleaned.csv'
    pathToRoDataset = 'dataset/raw/Romanian/depression_dataset_reddit_cleaned.csv'
    pathToChangedRoDataset = 'dataset/raw/Romanian/depression_dataset_reddit_cleaned_changed.csv'
    pathToChangedProcessedRo = 'dataset/processedWithLIWC/Romanian/depression_dataset_reddit_cleaned_changed.csv'
    pathToProcessedRo = 'dataset/processedWithLIWC/Romanian/depression_dataset_reddit_cleaned.csv'
    removeDoubleQoute(pathToProcessedRo, pathToChangedProcessedRo)
    # translateDataset(pathToEnDataset, pathToRoDataset)

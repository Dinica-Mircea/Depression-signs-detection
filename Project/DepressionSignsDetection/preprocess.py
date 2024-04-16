from collections import Counter

import liwc

from transformers import AutoTokenizer


from tokenizers import Tokenizer


def tokenizeForEnglishLanguage(text):
    tokenizer = Tokenizer.from_pretrained("bert-base-cased")
    return tokenizer.encode(text).tokens


def tokenizeForRomanianLanguage(text):
    # tokenizer only accepts this kind of special characters
    text = text.replace("ţ", "ț").replace("ş", "ș").replace("Ţ", "Ț").replace("Ş", "Ș")
    # load tokenizer and model
    tokenizerBertBaseRomanian = AutoTokenizer.from_pretrained("dumitrescustefan/bert-base-romanian-cased-v1")
    # get tokens
    return tokenizerBertBaseRomanian.tokenize(text, add_special_tokens=True)


def getLiwcCategories(tokens, parse_liwc, category_names_liwc):
    categories = dict()
    existentCategories = Counter(category for token in tokens for category in parse_liwc(token))
    for category in category_names_liwc:
        if existentCategories.get(category) is not None:
            categories.update({category: existentCategories.get(category)})
        else:
            categories.update({category: 0})

    return categories


def processDataset(pathToDataset, tokenizeFunction, pathToLiwcDictionary):
    parse, category_names = liwc.load_token_parser(pathToLiwcDictionary)
    fileInput = open(pathToDataset, 'r')
    lines = fileInput.readlines()

    labels = ['clean_text']
    labels.extend(category_names)
    labels.append('is_depression')
    processed = [labels]

    for idx, line in enumerate(lines[1:]):
        text = line[:line.__len__() - 3]
        label = line[line.__len__() - 2:line.__len__() - 1]
        tokens = tokenizeFunction(text)
        categories = getLiwcCategories(tokens, parse, category_names)
        newEntry = [text.replace(';', '')]
        newEntry.extend(categories.values())
        newEntry.append(int(label))
        processed.append(newEntry)
        if categories.values().__len__() != category_names.__len__():
            raise Exception("Categories not calculated correctly!")
        if idx % 100 == 0:
            print("At step ", idx)

    pathToProcessedDataset = pathToDataset.replace('raw', 'processed')
    with open(pathToProcessedDataset, 'w', encoding='utf-8') as fileOutput:
        print("Writing to file")
        for entry in processed:
            for value in entry:
                fileOutput.write(f"{value};")
            fileOutput.write("\n")


if __name__ == '__main__':
    pathToEnLiwcDictionary = 'liwc/Dictionary/free/LIWC2007_English100131.dic'
    pathToEnDataset = 'dataset/raw/English/depression_dataset_reddit_cleaned.csv'
    processDataset(pathToEnDataset, tokenizeForEnglishLanguage, pathToEnLiwcDictionary)

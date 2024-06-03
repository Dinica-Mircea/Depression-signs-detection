import json
import os
import pickle
import subprocess
from googletrans import Translator
from googletrans.models import Detected



def processWithLiwc(inputText, liwcDictionary):
    print(inputText)
    print(liwcDictionary)
    cmd_to_execute = ["LIWC-22-cli",
                      "--mode", "wc",
                      "--dictionary", liwcDictionary,
                      "--input", "console",
                      "--console-text", inputText,
                      "--output", "console"]
    results = subprocess.check_output(cmd_to_execute, shell=True).strip().splitlines()
    print(results)
    categories = json.loads(results[1])
    categories.pop('Emoji', None)
    print(categories)
    x = []
    for key, value in categories.items():
        x.append(value)
    return x


def getModel(pathToModel, root):
    root=os.path.dirname(os.path.abspath(__file__))
    pathToModel = os.path.join(root, pathToModel)
    pathToModel=pathToModel.replace("\\","\\\\")
    print(pathToModel)
    if not os.path.exists(pathToModel):
        print("File not found")
    else:
        print("File found")
    if os.access(pathToModel, os.R_OK):
        print("File is accessible for reading")
    else:
        print("File is not accessible for reading")
    rf = None
    try:
        with open(pathToModel, 'rb') as f:
            rf = pickle.load(f)
    except Exception as e:
        print("help")
        print(e.__str__())

    if rf is not None:
        return rf
    else:
        print("Couldn't load model")


def classify(inputText, root):
    modelPathEnglish = "models\English\model.cpickle"
    modelPathRo = "models\Romanian\model.cpickle"
    translator = Translator()
    result = translator.detect(inputText)
    language = Detected.__getattribute__(result, 'lang')
    print(language)
    if language == 'ro':
        rf = getModel(modelPathRo, root)
        liwcDictionary = "liwc/Dictionary/Romanian2015/LIWC2015 Dictionary - Romanian.dicx"
    elif language == 'en':
        rf = getModel(modelPathEnglish, root)
        liwcDictionary = "LIWC22"
    else:
        inputText = translator.translate(inputText, dest='en').text
        rf = getModel(modelPathEnglish, root)
        liwcDictionary = "LIWC22"

    inputCategories = processWithLiwc(inputText, liwcDictionary)
    print(rf.predict([inputCategories]))
    return int(rf.predict([inputCategories]))


def findInputsForThesis():
    modelPathEnglish = "models\English\model.cpickle"
    modelPathRo = "models\Romanian\model.cpickle"
    root = r"C:\Users\Mircea\Desktop\Licenta\Depression-signs-detection\Project\DepressionSignsDetection"
    pathToEnDataset = 'dataset/raw/English/depression_dataset_reddit_cleaned.csv'
    pathToRoDataset = 'dataset/raw/Romanian/depression_dataset_reddit_cleaned_changed.csv'

    rfRo = getModel(modelPathRo, root)
    rfEn = getModel(modelPathEnglish, root)
    xEn = []
    yEn = []
    fileInput = open(pathToEnDataset, 'r')
    lines = fileInput.readlines()
    for line in lines[1:]:
        text = line[:line.__len__() - 3]
        label = line[line.__len__() - 3:]
        xEn.append(text)
        yEn.append(label)

    xRo = []
    yRo = []
    fileInput = open(pathToRoDataset, 'r', encoding='utf-8')

    lines = fileInput.readlines()
    for line in lines[1:]:
        text = line[:line.__len__() - 3]
        label = line[line.__len__() - 3:]
        xRo.append(text)
        yRo.append(label)

    for i in range(0,len(xRo)):
        textRo=xRo.__getitem__(i)
        textEn=xEn.__getitem__(i)
        outputRo=classify(textRo,root)
        outputEn=classify(textEn,root)
        trueOutput=yEn.__getitem__(i).replace("\n","").replace(",","")
        if outputRo==outputEn and outputRo==int(trueOutput):
            print("Text corect si pentru engleza si pentru romana")
            print(textRo)
            print(textEn)

        if outputRo!=outputEn and outputEn==int(trueOutput):
            print("Text corect pentru engleza dar nu pentru romana")
            print(textRo)
            print(textEn)

        if outputRo==outputEn and outputRo!=int(trueOutput):
            print("Text incorect si pentru engleza si pentru romana")
            print(textRo)
            print(textEn)




if __name__ == '__main__':
    inputStringTest = "This is some text that I would like to analyze. After it has finished, I will say \"Thank you, " \
                      "LIWC!\" "
    inputStringDepressed = "The days blend into a monotonous gray. " \
                           "I wake up to the echo of my own thoughts, whispering doubts that linger like morning fog." \
                           " There's a weight on my chest, heavy with unspoken words and unshed tears. The world outside moves" \
                           " in vivid colors, but I watch through a filter that drains the vibrancy from everything I see. " \
                           "It's as if I'm behind a glass wall, able to see life but unable to feel its warmth. Every " \
                           "small" \
                           " task feels like a mountain, and the summit is always cloaked in clouds of disinterest and fatigue. " \
                           "I long for a break in the clouds, a moment of sunlight, but it seems so far away."
    inputStringDepressedRo = "Zilele se contopesc într-un gri monoton. Mă trezesc în ecoul propriilor gânduri, " \
                             "care îmi șoptesc îndoieli persistente ca o ceață de dimineață. O greutate apasă pe " \
                             "pieptul meu, încărcat cu cuvinte nespuse și lacrimi nevărsate. Lumea din afară se mișcă " \
                             "în culori vii, dar eu privesc printr-un filtru care îi scurge de vigoare tot ce văd. E " \
                             "ca și cum aș fi în spatele unui perete de sticlă, capabil să văd viața, dar incapabil " \
                             "să simt căldura ei. Fiecare sarcină mică pare un munte, iar vârful este întotdeauna " \
                             "învăluit în nori de dezinteres și oboseală. Tânjesc după o răbufnire a norilor, " \
                             "un moment de soare, dar pare atât de departe. "
    inputStringDepressedDe = "Gedanken, die Zweifel flüstern, die wie Morgennebel verweilen. Eine Schwere liegt auf " \
                             "meiner Brust, beladen mit unausgesprochenen Worten und ungeweinten Tränen. Die Welt da " \
                             "draußen bewegt sich in lebendigen Farben, doch ich betrachte sie durch einen Filter, " \
                             "der alles, was ich sehe, seiner Lebhaftigkeit beraubt. Es ist, als stünde ich hinter " \
                             "einer Glaswand, fähig das Leben zu sehen, jedoch unfähig, seine Wärme zu fühlen. Jede " \
                             "kleine Aufgabe erscheint wie ein Berg, und der Gipfel ist stets in Wolken aus " \
                             "Desinteresse und Erschöpfung gehüllt. Ich sehne mich nach einer Lücke in den Wolken, " \
                             "einem Moment des Sonnenlichts, doch er scheint so fern. "
    root = r"C:\Users\Mircea\Desktop\Licenta\Depression-signs-detection\Project\DepressionSignsDetection"
    #classify(inputStringDepressedRo, root)
    findInputsForThesis()

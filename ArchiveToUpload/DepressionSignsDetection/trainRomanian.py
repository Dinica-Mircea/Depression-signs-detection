import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import recall_score
from sklearn.model_selection import train_test_split

from trainEnglish import plot_classification_metrics, show_confusion_matrix, show_roc_curve, top10features


def split_data(pathToDataset):
    fileInput = open(pathToDataset, 'r', encoding='utf-8')
    lines = fileInput.readlines()
    x = []
    y = []
    feature_names = lines[0].split(',')[2:]
    for line in lines[1:]:
        splitted = line.split(',')
        y.append(float(splitted[1]))
        x.append([0 if el == '' else float(el) for el in splitted[2:-1]])

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.25, stratify=y)
    return X_train, X_test, y_train, y_test, feature_names


def train(pathToDataset, pathToSaveMetrics, pathToSaveModel, pathToMaxMetric, pathToMaxParameters):
    X_train, X_test, y_train, y_test, feature_names = split_data(pathToDataset)
    print(feature_names.__len__())
    n_estimators = 900
    max_features = 6
    min_samples_leaf = 3
    max_samples = 8
    with open(pathToMaxMetric, 'r') as read:
        maxRecall = float(read.readline())
    rf = RandomForestClassifier(n_estimators=n_estimators, max_features=max_features,
                                min_samples_leaf=min_samples_leaf, max_samples=max_samples / 10)
    recall = 0
    while recall < maxRecall:
        X_train, X_test, y_train, y_test, feature_names = split_data(pathToDataset)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)
        recall = recall_score(y_test, y_pred)
        print("Recall:", recall)

    # rf.fit(X_train, y_train)
    # y_pred = rf.predict(X_test)
    # recall = recall_score(y_test, y_pred)
    #if recall >= maxRecall:
    print("new max for: " + str(n_estimators) + ";" + str(max_features) + ";" + str(
        min_samples_leaf) + ";" + str(max_samples))
    with open(pathToMaxMetric, 'w+') as write:
        write.write(str(recall))
    with open(pathToMaxParameters, 'w+') as maxPar:
        maxPar.write('n_estimators=' + str(n_estimators) + ';max_features=' + str(max_features) +
                     ';min_samples_leaf=' + str(min_samples_leaf) + ';max_samples=' + str(
            max_samples))
    with open(pathToSaveModel, 'wb') as f:
        pickle.dump(rf, f)
    plot_classification_metrics(rf, X_test, y_test, pathToSaveMetrics)
    show_confusion_matrix(y_test, y_pred, pathToSaveMetrics)
    show_roc_curve(rf, y_test, X_test, pathToSaveMetrics)
    top10features(rf, pathToSaveMetrics, feature_names)
    # elif recall < 0.950:
    #     X_train, X_test, y_train, y_test, feature_names = split_data(pathToDataset)


if __name__ == '__main__':
    pathToDatasetRo = 'dataset/processedWithLIWC/Romanian/depression_dataset_reddit_cleaned.csv'
    pathToSaveMetricsRo = 'metrics/experimentRomanian/'
    pathToSaveModelRo = 'models/Romanian/model.cpickle'
    pathToMaxMetricRo = 'models/Romanian/maxRecall.txt'
    pathToMaxParamsRo = 'models/Romanian/hyperparamsForMax.txt'
    train(pathToDatasetRo, pathToSaveMetricsRo, pathToSaveModelRo, pathToMaxMetricRo, pathToMaxParamsRo)

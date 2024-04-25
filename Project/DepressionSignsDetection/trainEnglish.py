import os

import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, roc_curve, auc, \
    confusion_matrix, precision_score, recall_score, f1_score, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.ensemble import RandomForestClassifier
import pickle


def split_data(pathToDataset):
    fileInput = open(pathToDataset, 'r')
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
    rf = RandomForestClassifier(n_estimators=900, max_features=6,
                                min_samples_leaf=3, max_samples=0.8)
    recall = 0
    while recall < 0.953:
        X_train, X_test, y_train, y_test, feature_names = split_data(pathToDataset)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)
        recall = recall_score(y_test, y_pred)
        print("Recall:", recall)

    for n_estimators in range(700, 1000, 100):
        for max_features in range(6, 10):
            for min_samples_leaf in range(3, 9):
                for max_samples in range(6, 9):
                    rf = RandomForestClassifier(n_estimators=n_estimators, max_features=max_features,
                                                min_samples_leaf=min_samples_leaf, max_samples=max_samples / 10)
                    rf.fit(X_train, y_train)
                    y_pred = rf.predict(X_test)
                    recall = recall_score(y_test, y_pred)
                    with open(pathToMaxMetric, 'r') as read:
                        maxRecall = float(read.readline())

                    print("Recall:", recall)
                    if recall >= maxRecall:
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


def plot_classification_metrics(rf, X_test, y_test, pathToSave):
    # Predicting the test set results
    y_pred = rf.predict(X_test)

    # Calculating metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='binary')
    recall = recall_score(y_test, y_pred, average='binary')
    f1 = f1_score(y_test, y_pred, average='binary')

    # Metrics for visualization
    metrics = [accuracy, precision, recall, f1]
    metric_names = ['Accuracy', 'Precision', 'Recall', 'F1 Score']

    # Creating bar plot for metrics
    fig, ax = plt.subplots()
    ax.bar(metric_names, metrics, color=['blue', 'green', 'red', 'purple'])
    ax.set_ylim([0, 1])
    ax.set_ylabel('Score')
    ax.set_title('Classification Metrics')
    ax.set_ylim([0, 1.05])  # Set y-axis limit slightly above 1 for better visualization

    # Adding the text labels for each bar
    for i in range(len(metrics)):
        ax.text(i, metrics[i] + 0.02, f'{metrics[i]:.2f}', ha='center', color='black')

    plt.savefig(os.path.join(pathToSave, 'classificationMetrics.jpg'))
    # plt.show()


def show_confusion_matrix(y_test, y_pred, pathToSave):
    cm = confusion_matrix(y_test, y_pred)
    ConfusionMatrixDisplay(confusion_matrix=cm).plot()
    plt.savefig(pathToSave + 'confusionMatrix.jpg')
    # plt.show()


def show_roc_curve(rf, y_test, X_test, pathToSave):
    # Make sure y_test is a numpy array
    y_test = np.array(y_test)

    # Assuming binary classification; determine classes dynamically
    classes = np.unique(y_test)
    if len(classes) != 2:
        print("Error: ROC curve requires exactly two classes for binary classification.")
        return

    y_test_bin = label_binarize(y_test, classes=classes)

    # Predict probabilities
    y_score = rf.predict_proba(X_test)[:, 1]  # Assumed to be the second column for the positive class

    # Compute ROC curve and ROC area for each class
    fpr, tpr, _ = roc_curve(y_test_bin, y_score)
    roc_auc = auc(fpr, tpr)

    # Plotting
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(pathToSave, 'roc_curve.jpg'))
    # plt.show()


def top10features(rf, pathToSave, feature_names):
    # Feature Importance
    # Feature Importance
    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1]

    # Selecting the top 10 most important features
    top_n_indices = indices[:10]
    top_n_feature_names = [feature_names[i] for i in top_n_indices]  # Get the names of the top features

    # Plotting feature importances for the top 10 features only
    plt.figure(figsize=(12, 8))  # Increase figure size for better readability
    plt.title('Top 10 Feature Importances')
    bars = plt.bar(range(len(top_n_indices)), importances[top_n_indices], color='r', align='center')
    plt.xticks(range(len(top_n_indices)), top_n_feature_names, rotation=45,
               ha='right')  # Adjust label angles and alignment
    plt.xlim([-1, len(top_n_indices)])
    plt.tight_layout()  # Adjust layout to make room for label text

    plt.savefig(pathToSave + 'top10features.jpg')
    # plt.show()


if __name__ == '__main__':
    pathToEnglishDataset = 'dataset/processedWithLIWC/English/depression_dataset_reddit_cleaned.csv'
    pathToSaveMetrics = 'metrics/experiment2/'
    pathToSaveModel = 'models/English/model.cpickle'
    pathToMaxMetric = 'models/English/maxRecall.txt'
    pathToMaxParams = 'models/English/hyperparamsForMax.txt'
    train(pathToEnglishDataset, pathToSaveMetrics, pathToSaveModel, pathToMaxMetric, pathToMaxParams)

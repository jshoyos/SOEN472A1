import sys
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn import datasets as ds
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

BBC_PATH = "./DATA/BBC/"

def Write_to_file(txt):
    try:
        with open(r"bbc-performance.txt", "a") as result_file:
            result_file.write(txt)
    except Exception as e:
        print("ERROR SOMETHING HAPPENED WHEN WRITING TO FILE. EXCEPTION WAS THROWN")
        print(e)
        sys.exit()

def Task1(try_numb, x_train, x_test, y_train, y_test, smoothing=None):
    vectorizer = CountVectorizer(stop_words='english')
    count_matrix = vectorizer.fit_transform(data["data"])
    count_array = count_matrix.toarray()

    classification = MultinomialNB(alpha=smoothing) if (smoothing != None) else MultinomialNB()
    classification.fit(vectorizer.transform(x_train), y_train)

    y_pred = classification.predict(vectorizer.transform(x_test))

    Write_to_file("a)\n******************** MultinomialNB default values try " +
                        str(try_numb) + " ********************\n")
    Write_to_file("b)\n" +
                        str(confusion_matrix(y_test, y_pred)) + "\n")
    Write_to_file(
        "c)\n" + str(classification_report(y_test, y_pred)) + "\n")
    Write_to_file("d)\naccuracy: " +
                        str(accuracy_score(y_test, y_pred)) + "\n")
    Write_to_file(
        "macro average F1: " + str(f1_score(y_test, y_pred, average='macro')) + "\n")
    Write_to_file(
        "weighted average F1: " + str(f1_score(y_test, y_pred, average='weighted')) + "\n")
    Write_to_file("e)\n")
    for key, value in classes_dic.items():
        prob = value/total_files
        Write_to_file("Probability of class " +  str(key) + ": " + str(prob) + "\n")
    vocabulary_size = len(vectorizer.get_feature_names_out())
    Write_to_file("f)\nSize of vocabulary: " + str(vocabulary_size) + "\n")

    Write_to_file("g)\nNumber of word-tokens in each class:\n")
    word_tokens = classification.feature_count_
    total_count_word_tokens = 0
    for i in range(0,len(word_tokens)):
        tokens = np.sum(word_tokens[i])
        Write_to_file(str(data.target_names[i]) + ": " + str(tokens) + "\n")

    total_count_word_tokens = np.sum(word_tokens)
    Write_to_file("h)\nNumber of word-tokens in corpus: " + str(total_count_word_tokens) + "\n")

    Write_to_file("i)\n")
    for i in range(0,len(word_tokens)):
        zeros_in_class = np.count_nonzero(word_tokens[i] == 0)
        Write_to_file(f"Number of zeros in {data.target_names[i]}: {zeros_in_class}\n With percentage of {zeros_in_class*100/np.sum(word_tokens[i])}\n")
    
    ones_in_class = np.count_nonzero(word_tokens == 1)
    Write_to_file(f"j)\nNumber of ones in corpus: {ones_in_class}\n With percentage of {ones_in_class * 100/np.sum(word_tokens[i])}\n")

data = ds.load_files(BBC_PATH, encoding="latin1")

labels, counts = np.unique(data.target, return_counts=True)
classes_dic = dict(zip(np.array(data.target_names)[labels], counts))

plt.title("Class distribution of the BBC dataset")
plt.xlabel("classes")
plt.ylabel("Number of files")
plt.bar([1, 2, 3, 4, 5], counts)

plt.savefig("BBC-distribution.pdf")

x_train, x_test, y_train, y_test = train_test_split(
    data.data, data.target, train_size=0.8, test_size=0.2)
total_files = np.sum(counts)

Task1(1, x_train, x_test, y_train, y_test)

Task1(2, x_train, x_test, y_train, y_test)

Task1(3, x_train, x_test, y_train, y_test, smoothing=0.0001)

Task1(4, x_train, x_test, y_train, y_test, smoothing=0.9)
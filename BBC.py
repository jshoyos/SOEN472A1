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

def Task1(try_numb, x_train, x_test, y_train, y_test, smoothing=None):

    classification = MultinomialNB(alpha=smoothing) if (smoothing != None) else MultinomialNB()
    classification.fit(vectorizer.transform(x_train), y_train)

    y_pred = classification.predict(vectorizer.transform(x_test))

    Write_to_file("a)\n******************** MultinomialNB default values try " +
                        str(try_numb) + "********************\n")
    Write_to_file("b)\n" +
                        str(confusion_matrix(y_test, y_pred)) + "\n")
    Write_to_file(
        "c)\n" + str(classification_report(y_test, y_pred)) + "\n")
    Write_to_file("d)\naccuracy: " +
                        str(accuracy_score(y_test, y_pred)) + "\n")
    Write_to_file(
        "macro average F1:" + str(f1_score(y_test, y_pred, average='macro')) + "\n")
    Write_to_file(
        "weighted average F1:" + str(f1_score(y_test, y_pred, average='weighted')) + "\n")


data = ds.load_files(BBC_PATH, encoding="latin1")

labels, counts = np.unique(data.target, return_counts=True)
classes_dic = dict(zip(np.array(data.target_names)[labels], counts))
print(classes_dic)

plt.title("Class distribution of the BBC dataset")
plt.xlabel("classes")
plt.ylabel("Number of files")
plt.bar([1, 2, 3, 4, 5], counts)

plt.savefig("BBC-distribution.pdf")


vectorizer = CountVectorizer(stop_words='english')
count_matrix = vectorizer.fit_transform(data["data"])
count_array = count_matrix.toarray()
df = pd.DataFrame(data=count_array, columns=vectorizer.get_feature_names_out())

print(df)

x_train, x_test, y_train, y_test = train_test_split(
    data.data, data.target, train_size=0.2, test_size=0.8)

total_files = np.sum(counts)
Write_to_file("e)\n")
for key, value in classes_dic.items():
    prob = value/total_files
    Write_to_file("Probability of class " +  str(key) + ": " + str(prob) + "\n")
Task1(1, x_train, x_test, y_train, y_test)
Task1(2, x_train, x_test, y_train, y_test)
Task1(3, x_train, x_test, y_train, y_test, smoothing=0.0001)
Task1(4, x_train, x_test, y_train, y_test, smoothing=0.9)

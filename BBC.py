import collections
import os
import numpy as np
from matplotlib import pyplot as plt
from sklearn import datasets as ds
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
from sklearn.model_selection import train_test_split

BBC_PATH = "./DATA/BBC/"

folders = ([name for name in os.listdir(BBC_PATH)])
results = []

for folder in folders:
    folder_path = os.path.join(BBC_PATH,folder)
    if (os.path.isdir(folder_path)):
        content = os.listdir(folder_path)
        results.append( len(content))

plt.title("Class distribution of the BBC dataset")
plt.xlabel("classes")
plt.ylabel("Number of files")
plt.bar([1,2,3,4,5], results)

plt.savefig("BBC-distribution.pdf")

data = ds.load_files(BBC_PATH, encoding = "latin1")

count_vect = CountVectorizer(stop_words='english')
count_matrix = count_vect.fit_transform(data["data"])
count_array = count_matrix.toarray()
df = pd.DataFrame(data=count_array, columns=count_vect.get_feature_names())

print(df)

#x_train, x_test, y_train, y_test = train_test_split(data, train_size= 0.2, test_size= 0.8)





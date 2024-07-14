import pandas as pd


import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from skmultilearn.adapt import MLkNN
from sklearn.metrics import hamming_loss, accuracy_score
from scipy.sparse import csr_matrix
from sklearn.metrics import multilabel_confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

raw = pd.read_csv('dataset_25k_combined.csv')
#dfa = pd.DataFrame(raw, columns = ["y","post"])
df = pd.DataFrame(raw, columns = ["annotate","post"])
dfa = df.sample(frac=1).reset_index(drop=True)

dfa = dfa.sample(frac=1).reset_index(drop=True)
dfa = dfa.sample(frac=1).reset_index(drop=True)

#annotation
dfa['xAURI'] = d.apply(lambda x: x['annotate'].count("AURI"), axis=1)
dfa['xPN'] = d.apply(lambda x: x['annotate'].count("PN"), axis=1)
dfa['xTB'] = d.apply(lambda x: x['annotate'].count("TB"), axis=1)
dfa['xCOVID'] = d.apply(lambda x: x['annotate'].count("COVID"), axis=1)

X = dfa["post"]
y = np.asarray(dfa[dfa.columns[2:]])

vetorizar = TfidfVectorizer(max_features=3000, max_df=0.85)
vetorizar.fit(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=3)

X_train_tfidf = vetorizar.transform(X_train)
X_test_tfidf = vetorizar.transform(X_test)

num_epochs = 2  # Number of epochs
k_neighbors = 9  # Number of nearest neighbors
s = 1.5

#mlknn = MLkNN(k=k_neighbors)
mlknn_classifier = MLkNN(k=k_neighbors, s = s)

for epoch in range(num_epochs):

    mlknn_classifier.fit(X_train_tfidf, y_train)

    predicted = mlknn_classifier.predict(X_test_tfidf)

    #print(accuracy_score(y_test, predicted))
    #print(hamming_loss(y_test, predicted))

    # Evaluate performance metrics
    accuracy = accuracy_score(y_test, predicted)
    hamming_loss_value = hamming_loss(y_test, predicted)

    # Print training progress
    print(f'Epoch {epoch + 1}/{num_epochs}, Accuracy: {accuracy:f}, Hamming Loss: {hamming_loss_value:f}')



from sklearn.compose import ColumnTransformer

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import OneHotEncoder

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn import metrics

random_seed = 3

def multinomialNaiveBayes(dataset, labels):
    X_train, X_test, Y_train, Y_test = train_test_split(dataset, labels, test_size=0.2, random_state=random_seed)
    
    nb = MultinomialNB()
    nb.fit(X_train, Y_train)
    y_pred = nb.predict(X_test)
    y_true = Y_test

    print("Accuracy: ", accuracy_score(y_pred, y_true))
    
    print()
    print(classification_report(y_true, y_pred, target_names=["Negativo", "Positivo"]))
    kCrossValidation2(nb, dataset, labels)

def randomForest(dataset, labels):
    X_train, X_test, Y_train, Y_test = train_test_split(dataset, labels, test_size=0.2, random_state=random_seed)
    
    rf = RandomForestClassifier(n_estimators = 200, criterion = 'entropy')
    rf.fit(X_train, Y_train)
    y_pred=rf.predict(X_test)
    
    print("Accuracy:",metrics.accuracy_score(Y_test, y_pred))
    
    # feature selection
    sel = SelectFromModel(rf, prefit=True)
    selected_feat = dataset.columns[(sel.get_support())]
    print(len(selected_feat))

    truncated_df = dataset.loc[:, sel.get_support()]
    return truncated_df
'''
def kCrossValidation(model, x, y):
    vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5)
    x_mod = vectorizer.fit_transform(x)
    k = []
    acc = []
    dev = []

    #nota, non viene effettuato lo shuffle dei fold, quindi sono sempre gli stessi, sono già istanziati
    for i in range(10, 16):
        scores = cross_val_score(model, x_mod, y, cv=i)
        print("K cross validation, k= ", i)
        k.append(i)
        print("Average scores: ", scores.mean())
        acc.append(scores.mean())
        print("Standard Deviation of scores: ", scores.std())
        dev.append(scores.std())
        
        print("\n\n")
'''
def kCrossValidation2(model, dataset, labels):
    #vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5)
    #x_mod = vectorizer.fit_transform(x)
    k = []
    acc = []
    dev = []

    #nota, non viene effettuato lo shuffle dei fold, quindi sono sempre gli stessi, sono già istanziati
    for i in range(10, 16):
        scores = cross_val_score(model, dataset, labels, cv=i)
        print("K cross validation, k= ", i)
        k.append(i)
        print("Average scores: ", scores.mean())
        acc.append(scores.mean())
        print("Standard Deviation of scores: ", scores.std())
        dev.append(scores.std())
        print("\n\n")
    
def createDataframeWithTfIdf(dataframe):
    reviewText = dataframe['reviewText'].to_numpy()
    vectorizer = TfidfVectorizer(sublinear_tf=True)
    tfidf_vector = vectorizer.fit_transform(reviewText)
    tfidf_df = pd.DataFrame(tfidf_vector.toarray(), index=reviewText, columns=vectorizer.get_feature_names())
    
    tfidf_df['numberPositiveReview'] = dataframe['numberPositiveReview'].to_numpy()
    tfidf_df['numberNegativeReview'] = dataframe['numberNegativeReview'].to_numpy()
    tfidf_df.reset_index(drop=True, inplace=True)
    sentiment = dataframe['sentiment']
    return tfidf_df, sentiment
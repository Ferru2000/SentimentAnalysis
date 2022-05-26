from sklearn.compose import ColumnTransformer

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import OneHotEncoder

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectFromModel
from sklearn import metrics

from sklearn.neighbors import KNeighborsClassifier

import math

random_seed = 3

def randomForest_parameterTuning(dataset, labels):
    n_estimators_list = [40,60,80,100]
    criterion_list = ['entropy']
    max_depth_list = [35,55]
    max_depth_list.append(None)
    min_samples_split_list = [15, 30]
    min_samples_leaf_list = [5, 15, 30]
    max_features_list = ['log2']

    params_grid = {
        'n_estimators': n_estimators_list,
        'criterion': criterion_list,
        'max_depth': max_depth_list,
        'min_samples_split': min_samples_split_list,
        'min_samples_leaf': min_samples_leaf_list,
        'max_features': max_features_list
    }
    X_train, X_test, Y_train, Y_test = train_test_split(dataset, labels, test_size=0.2, random_state=random_seed)
    model_rf2 = GridSearchCV(estimator=RandomForestClassifier(class_weight='balanced'),
                            param_grid=params_grid,
                            cv=3,
                            scoring=my_roc_auc_score,
                            return_train_score=True,
                            verbose=2)

    model_rf2.fit(X_train, Y_train)
    return model_rf2
    

def randomForest_featureSelection(dataset, labels):
    X_train, X_test, Y_train, Y_test = train_test_split(dataset, labels, test_size=0.2, random_state=random_seed)
    
    rf = RandomForestClassifier(criterion = 'entropy', random_state=random_seed)
    rf.fit(X_train, Y_train)
    y_pred=rf.predict(X_test)
    
    #print("Accuracy:",metrics.accuracy_score(Y_test, y_pred))
    
    # feature selection
    sel = SelectFromModel(rf, prefit=True)
    selected_feat = dataset.columns[(sel.get_support())]
    print("Numero feature selezionate: " + str(len(selected_feat)))

    truncated_df = dataset.loc[:, sel.get_support()]
    return truncated_df

def randomForest(dataset, labels, best_params):
    X_train, X_test, Y_train, Y_test = train_test_split(dataset, labels, test_size=0.2, random_state=random_seed)
    
    # prendo i parametri migliori trovati dal Gridsearch
    n_estimators_found = best_params.get('n_estimators')
    max_depth_found = best_params.get('max_depth')
    max_features_found = best_params.get('max_features')
    min_samples_leaf_found = best_params.get('min_samples_leaf')
    min_samples_split_found = best_params.get('min_samples_split')
    criterion_found = best_params.get('criterion')
    
    rf = RandomForestClassifier(n_estimators = n_estimators_found, 
                                criterion = criterion_found,
                                class_weight= 'balanced',
                                max_depth= max_depth_found,
                                max_features= max_features_found,
                                min_samples_leaf= min_samples_leaf_found,
                                min_samples_split= min_samples_split_found)
    rf.fit(X_train, Y_train)
    y_pred=rf.predict(X_test)
    
    print("Accuracy:",metrics.accuracy_score(y_pred, Y_test))
    print(classification_report(Y_test, y_pred, target_names=["Negativo", "Positivo"]))
    kCrossValidation(rf, dataset, labels)
    
    return rf
    
    
def multinomialNaiveBayes(dataset, labels):
    X_train, X_test, Y_train, Y_test = train_test_split(dataset, labels, test_size=0.2, random_state=random_seed)
    
    nb = MultinomialNB()
    nb.fit(X_train, Y_train)
    y_pred = nb.predict(X_test)
    y_true = Y_test

    print("Accuracy: ", accuracy_score(y_pred, y_true))
    
    print()
    print(classification_report(y_true, y_pred, target_names=["Negativo", "Positivo"]))
    kCrossValidation(nb, dataset, labels)

def findBestK(dataset, labels):
    X_train, X_test, Y_train, Y_test = train_test_split(dataset, labels, test_size=0.2, random_state=random_seed)
    error = []
    
    acc_train = []
    acc_test = []
    max_acc_test = 0
    max_acc_train = 0
    best_k = 0
    
    root = int(math.sqrt(X_train.shape[0]))
    k = [9,11,13,15,17,19]
    k.append(root)
    
    for i in k:
        print(i)
        knn = KNeighborsClassifier(n_neighbors=i)
        knn.fit(X_train, Y_train)
        pred_i = knn.predict(X_test)
        error.append(np.mean(pred_i != Y_test))
        
        print("train: ", knn.score(X_train, Y_train))
        print("test: ", knn.score(X_test, Y_test))
        acc_train.append(knn.score(X_train, Y_train))
        acc_test.append(knn.score(X_test, Y_test))
        if knn.score(X_train, Y_train) >= max_acc_train and knn.score(X_test, Y_test) >= max_acc_test:
            best_k = i
            max_acc_test = knn.score(X_test, Y_test)
            max_acc_train = knn.score(X_train, Y_train)
        
    print(error)
    
    x=[9,11,13,15,17,19,97]
    values = range(len(x))

    plt.figure(figsize=(12, 6))
    plt.xticks(values,x)
    plt.plot(values, error, color='red', linestyle='dashed', marker='o',
             markerfacecolor='blue', markersize=10)
    plt.title('Error Rate K Value')
    plt.xlabel('K Value')
    plt.ylabel('Mean Error')

    return best_k

def knn(dataset, labels):  #KNN
    #features_train, features_test, labels_train, labels_test = processing(description, continent)
    X_train, X_test, Y_train, Y_test = train_test_split(dataset, labels, test_size=0.2, random_state=random_seed)
    
    #bestK = maxK(features_train, features_test, labels_train, labels_test)
    k = findBestK(dataset, labels)
    knc = KNeighborsClassifier(n_neighbors=k, weights="distance")
    knc.fit(X_train, Y_train)

    y_pred = knc.predict(X_test)
    y_true = Y_test

    print(accuracy_score(y_true, y_pred))
    #print(confusion_matrix(y_true, y_pred))
    print(classification_report(y_true, y_pred, target_names=["Negativo", "Positivo",]))

    kCrossValidation(knc, dataset, labels)

def my_roc_auc_score(model, truncated_df, labels):
    return metrics.roc_auc_score(labels, model.predict(truncated_df))

def kCrossValidation(model, dataset, labels):
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
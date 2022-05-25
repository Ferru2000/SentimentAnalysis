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
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.feature_selection import SelectFromModel
from sklearn import metrics

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
    
    rf = RandomForestClassifier(criterion = 'entropy')
    rf.fit(X_train, Y_train)
    y_pred=rf.predict(X_test)
    
    #print("Accuracy:",metrics.accuracy_score(Y_test, y_pred))
    
    # feature selection
    sel = SelectFromModel(rf, prefit=True)
    selected_feat = dataset.columns[(sel.get_support())]
    print(len(selected_feat))

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
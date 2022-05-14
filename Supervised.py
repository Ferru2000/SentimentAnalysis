from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
#from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score


def processingTfIdf(reviewText, sentiment):
    X_train, X_test, Y_train, Y_test = train_test_split(reviewText, sentiment, test_size=0.3)
    
    # Tf-idf
    vectorizer = TfidfVectorizer(sublinear_tf=True)
    X_train_transformed = vectorizer.fit_transform(X_train)
    X_test_transformed = vectorizer.transform(X_test)

    return X_train_transformed, X_test_transformed, Y_train, Y_test

#valori: rs=8  0.8313888888888888
def multinb(textReview, sentiment):
    X_train, X_test, Y_train, Y_test = processingTfIdf(textReview, sentiment)

    nb = MultinomialNB()
    nb.fit(X_train, Y_train)
    y_pred = nb.predict(X_test)
    y_true = Y_test

    print("Accuracy: ", accuracy_score(y_true, y_pred))
    print()
    #print(confusion_matrix(y_true, y_pred))
    print(classification_report(y_true, y_pred, target_names=["Negativo", "Positivo"]))
    kCrossValidation(nb, textReview, sentiment)
    
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
    
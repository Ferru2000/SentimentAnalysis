from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

def processingTfIdf(reviewText, sentiment):
    X_train, X_test, Y_train, Y_test = train_test_split(reviewText, sentiment, test_size=0.3, random_state=2)
    
    # Tf-idf
    vectorizer = TfidfVectorizer(sublinear_tf=True)
    X_train_transformed = vectorizer.fit_transform(X_train)
    X_test_transformed = vectorizer.transform(X_test)

    return X_train_transformed, X_test_transformed, Y_train, Y_test
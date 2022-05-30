from bisect import bisect_left

'''
   Questa funzione effettua il cleaning del dataset
   rimuovendo le colonne irrilevanti ed eliminando righe con valori mancanti
'''
def dataset_cleaning(dataset):  
    #eliminazione colonne irrilevanti
    clean_dataset = dataset.drop(columns={'Unnamed: 0', 'Unnamed: 0.1', 'asin', 'helpful',
                                          'reviewTime', 'reviewerID', 'reviewerName', 'unixReviewTime'}, inplace=False) 
    
    #eliminazione righe che hanno valori mancanti
    clean_dataset.dropna(axis='index', how='any', inplace=True)
    
    return clean_dataset

'''
   Questa funzione aggiunge una colonna al dataset indicante il sentimento della review
   basata sulla colonna reviewText ed associa un numero compreso in {-1, 0, 1} in base alla colonna rating.
   Rating 1,2 -> -1 (negativo)
          3   ->  0 (neutro)
          4,5 -> +1 (positivo)
'''
def addSentimentColumn(dataset):
    #creazione della colonna sentiment
    if not 'sentiment' in dataset:
        dataset.insert(3, 'sentiment', 0)
    
    dataset.loc[dataset["rating"] == 1, "sentiment"] = 0
    dataset.loc[dataset["rating"] == 2, "sentiment"] = 0
    dataset.loc[dataset["rating"] == 3, "sentiment"] = 0
    dataset.loc[dataset["rating"] == 4, "sentiment"] = 1
    dataset.loc[dataset["rating"] == 5, "sentiment"] = 1
    return dataset;

def countWords(list_word, text):
    splitted_text = text.split()
    count = 0
    
    for word in splitted_text:
        i = bisect_left(list_word, word)
        if i != len(list_word) and list_word[i] == word:
            count = count + 1
    
    if len(splitted_text) == 0:
        return 0
    
    count = count / len(splitted_text)
    count = round(count, 2)
    return count

def addNumberPositiveReview(dataset):
    #creazione della colonna numberPositiveReview
    if not 'numberPositiveReview' in dataset:
        dataset.insert(4, 'numberPositiveReview', 0)
    
    return dataset
    
def addNumberNegativeReview(dataset):
    #creazione della colonna numberNegativeReview
    if not 'numberNegativeReview' in dataset:
        dataset.insert(5, 'numberNegativeReview', 0)
    
    return dataset

def addNumberPositiveSummary(dataset):
    #creazione della colonna numberPositiveSummary
    if not 'numberPositiveSummary' in dataset:
        dataset.insert(6, 'numberPositiveSummary', 0)
    
    return dataset
    
def addNumberNegativeSummary(dataset):
    #creazione della colonna numberNegativeSummary
    if not 'numberNegativeSummary' in dataset:
        dataset.insert(7, 'numberNegativeSummary', 0)
    
    return dataset

def binarySearch(list_word, word):
    for word in list_word:
        i = bisect_left(list_word, word)
        if i != len(list_word) and list_word[i] == word:
            return 1
        else:
            return 0



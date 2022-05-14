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
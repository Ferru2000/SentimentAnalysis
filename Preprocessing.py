def dataset_cleaning(dataset):  
    #dataset cleaning dei duplicati, mantenendo la prima copia
    #clean_dataset = dataset.drop_duplicates(keep='first', inplace=False)

    #dataset cleaning dei valori mancanti, eliminando le righe in cui mancano i valori
    #clean_dataset.dropna(axis='index', how='any', inplace=True)

    #rimozione colonne irrilevanti
    clean_dataset = dataset.drop(columns={'Unnamed: 0', 'Unnamed: 0.1', 'asin', 'helpful',
                                          'reviewTime', 'reviewerID', 'reviewerName', 'unixReviewTime'}, inplace=False) 
    return clean_dataset
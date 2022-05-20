import string
import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer

'''
    Questa funzione converte tutte le lettere maiuscole in minuscole presenti
    nella colonna reviewText
'''
def text_lowerer(dataframe, column):
    # lowering delle lettere nella colonna reviewText
    dataframe[column] = dataframe[column].str.lower()
    return dataframe
    
'''
    Questa funzione elimina tutti i segni di punteggiatura
'''
def remove_punctuation(text):
    PUNCTUATION = string.punctuation
    return text.translate(str.maketrans('', '', PUNCTUATION))
    

'''
    Questa funzione elimina tutte le stopwords
'''
def remove_stopwords(text):
    STOPWORDS = set(stopwords.words('english'))
    return " ".join([word for word in str(text).split() if word not in STOPWORDS])


'''
    Questa funzione lemmatizza tutte le parole
'''
def lemmatize_words(text):
    lemmatizer = WordNetLemmatizer()
    wordnet_map = {"N":wordnet.NOUN, "V":wordnet.VERB, "J":wordnet.ADJ, "R":wordnet.ADV}
    
    pos_tagged_text = nltk.pos_tag(text.split())
    return " ".join([lemmatizer.lemmatize(word, wordnet_map.get(pos[0], wordnet.NOUN)) for word, pos in pos_tagged_text])

     
     
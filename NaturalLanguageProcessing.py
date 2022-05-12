import string
import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer

'''
    Questa funzione effettua il cleaning dei dati (colonna reviewText), eliminando i
    simboli di punteggiatura, le stopword e tokenizza le parole
'''
def text_cleaning(text):
    # rimozione dei simboli di punteggiatura
    cleaned_text = [char for char in text if char not in string.punctuation]
    
    # join delle singole lettere dopo la rimozione della punteggiatura
    cleaned_text = ''.join(cleaned_text)
    
    # lower del testo
    
    # setting delle stopword in inglese e rimozione nel return
    stopword = set(stopwords.words('english'))
    return [word for word in cleaned_text.split() if word.lower() not in stopword]


def text_lowerer(dataframe):
    # lowering delle lettere nella colonna reviewText
    dataframe["reviewText"] = dataframe["reviewText"].str.lower()
    return dataframe
    

def remove_punctuation(text):
    PUNCTUATION = string.punctuation
    return text.translate(str.maketrans('', '', PUNCTUATION))
    

def remove_stopwords(text):
    STOPWORDS = set(stopwords.words('english'))
    return " ".join([word for word in str(text).split() if word not in STOPWORDS])


def lemmatize_words(text):
    lemmatizer = WordNetLemmatizer()
    wordnet_map = {"N":wordnet.NOUN, "V":wordnet.VERB, "J":wordnet.ADJ, "R":wordnet.ADV}
    
    pos_tagged_text = nltk.pos_tag(text.split())
    return " ".join([lemmatizer.lemmatize(word, wordnet_map.get(pos[0], wordnet.NOUN)) for word, pos in pos_tagged_text])

     
     
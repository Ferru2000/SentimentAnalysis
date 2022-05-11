import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#istogramma della commestibilità
def sentiment_distribution(dataframe):    
    plt.figure(figsize=(12, 5))
    plt.title("Histogram of sentiment distribution")
    sns.countplot(x="sentiment", data = dataframe, palette=['#FF0000',"#BCBDBC", "#0DFF06"])
    plt.savefig('img/sentiment_distribution.png')
    plt.show()
    
def rating_distribution(dataframe):    
    plt.figure(figsize=(12, 5))
    plt.title("Histogram of sentiment distribution")
    sns.countplot(x="rating", data = dataframe, palette=sns.color_palette())
    plt.savefig('img/sentiment_distribution.png')
    plt.show()
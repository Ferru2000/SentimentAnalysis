import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# istogramma della distribuzione del sentiment
def sentiment_distribution(dataframe):    
    plt.figure(figsize=(12, 5))
    plt.title("Istogramma della distribuzione dei sentiment")
    sns.countplot(x="sentiment", data = dataframe, palette=['#FF0000',"#BCBDBC", "#0DFF06"])
    plt.savefig('img/sentiment_distribution.png')
    plt.show()

# istogramma della distribuzione del rating
def rating_distribution(dataframe):    
    plt.figure(figsize=(12, 5))
    plt.title("Istogramma della distribuzione dei rating")
    sns.countplot(x="rating", data = dataframe, palette=sns.color_palette())
    plt.savefig('img/rating_distribution.png')
    plt.show()
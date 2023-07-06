import string
import nltk
import pandas as pd
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag

from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer

def preprocess(df):
    # Remove special characters
    df['Text2'] = df['Text'].replace('\n',' ')
    df['Text2'] = df['Text2'].replace('\r',' ')

    # Remove punctuation signs and lowercase all
    df['Text2'] = df['Text2'].str.lower()
    df['Text2'] = df['Text2'].str.translate(str.maketrans('', '', string.punctuation))


    # Remove stop words
    stop_words = stopwords.words("english")
    lemmatizer = WordNetLemmatizer()

    def fwpt(each):
        tag = pos_tag([each])[0][1][0].upper()
        hash_tag = {"N": wordnet.NOUN,"R": wordnet.ADV, "V": wordnet.VERB,"J": wordnet.ADJ}
        return hash_tag.get(tag, wordnet.NOUN)


    def lematize(text):
        tokens = nltk.word_tokenize(text)
        ax = ""
        for each in tokens:
            if each not in stop_words:
                ax += lemmatizer.lemmatize(each, fwpt(each)) + " "
        return ax

    df['Text2'] = df['Text2'].apply(lematize)

def create_and_fit(clf, x, y, vector):
    best_clf = clf
    pipeline = Pipeline([('vectorize', vector), ('model', best_clf)])
    return pipeline.fit(x, y)

def classify_text(input_text):
    full_df = pd.read_csv('news_df.csv')
    preprocess(full_df)
    X_train, X_test, y_train, y_test = train_test_split(full_df['Text2'],
                                                        full_df['Class'],
                                                        test_size=0.2,
                                                        random_state=9)
    vector = TfidfVectorizer(stop_words='english',
                         ngram_range = (1,2),
                         min_df = 3,
                         max_df = 1.0,
                         max_features = 10000)
    X = pd.concat([X_train,
               X_test])
    Y = pd.concat([y_train,
                y_test])
    classifier = create_and_fit(MultinomialNB(), X, Y, vector)
    out = classifier.predict([input_text])[0]
    probs = classifier.predict_proba([input_text])
    return out,probs
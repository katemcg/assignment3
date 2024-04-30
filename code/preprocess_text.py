from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import pad_sequences
import numpy as np
import re
import string
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import nltk
from nltk.corpus import stopwords
#nltk.download('stopwords')

stop_words = stopwords.words('english') + ['u', 'im', 'c']
stemmer = nltk.SnowballStemmer("english")

def remove_stopwords(text):
    text = ' '.join(word for word in text.split(' ') if word not in stop_words)
    return text

def stemm_text(text):
    text = ' '.join(stemmer.stem(word) for word in text.split(' '))
    return text

def clean_text(text):
    '''Make text lowercase, remove text in square brackets,remove links,remove punctuation
    and remove words containing numbers.'''
    text = str(text).lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    return text

def preprocessor(texts, maxlen=40):
    """
    Preprocesses a list of texts using pre-defined functions to clean text, remove stopwords and stem words, tokenises using tf.keras Tokenizer and pads sequences to a fixed length of 40.

    Args:
    texts: List of strings representing input texts.
    maxlen: Maximum length of sequences after padding, set to 40 by default.

    Returns:
    padded_sequences: Numpy array of padded sequences.
    tokenizer: Tokenizer object fitted on the input texts.
    """
    # Clean puntuation, urls, and so on
    text = text.apply(clean_text)
    # Remove stopwords
    text = text.apply(remove_stopwords)
    # Stem all the words in the sentence
    text = text.apply(stemm_text)

    # Initialize Tokenizer
    tokenizer = Tokenizer()

    # Fit tokenizer on input texts
    tokenizer.fit_on_texts(texts)

    # Convert texts to sequences of integers
    sequences = tokenizer.texts_to_sequences(texts)

    # Pad sequences to a fixed length
    padded_sequences = pad_sequences(sequences, maxlen)

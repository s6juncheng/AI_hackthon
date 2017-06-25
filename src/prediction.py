from newspaper import Article
import re
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
import pickle
from keras.preprocessing.sequence import pad_sequences

# Get text
def get_text_title(url):
    article = Article(url)
    article.download()
    article.parse()
    text = article.text
    title = article.title
    return title, text

# Process: clean text
def rm_punctuation(tokenized):
    regex = re.compile('[%s]' % re.escape(string.punctuation)) 
    new_tokenized = []
    for token in tokenized:
        new_token = regex.sub(u'', token)
        if not new_token == u'':
            new_tokenized.append(new_token)
    return new_tokenized

def rm_stopwords(words):
    new_words = []
    for word in words:
        if not word in stopwords.words('english'):
            new_words.append(word)
    return new_words

def stemming(words):
    snowball = SnowballStemmer('english')
    new_words = []
    for word in words:
        new_words.append(snowball.stem(word))
    return new_words

def clean_text(text):
    # tokenize
    tokenized = word_tokenize(text)
    punc_rmd = rm_punctuation(tokenized)
    stw_rmd = rm_stopwords(punc_rmd)
    stmd = stemming(stw_rmd)
    new_text = " ".join(stmd)
    return new_text

# Process: word to index
def word2indx(text, trained_tokenizer, max_len=2000):
    if isinstance(text, str):
        # single string text
        text = [text]
    tokenizer = trained_tokenizer
    sequences = tokenizer.texts_to_sequences(text)
    data = pad_sequences(sequences, maxlen=max_len)
    return data

# Prediction
def predict_from_index(model, index, verbose = True):
    prediction = model.predict(index).flatten()
    prob_true = float(prediction[0] * 100)
    prob_false = float(prediction[1] * 100)
    if verbose:
        if prob_true >= 80.0:
            quote = "Likely a true news!" 
        elif (prob_true >= 60.0) & (prob_true < 80.0):
            quote = "Maybe a true news"
        elif (prob_true >= 40.0) & (prob_true < 60.0):
            quote = "Mabye a fake news"
        elif (prob_true > 20) & (prob_true < 40.0):
            quote = "Mabye a fake news"
        elif prob_true <= 20.0:
            quote = "Attention! Likely a fake news!"
        else:
            quote = "We are %0.3f percent confident that this is a true news" % prob_true
    return prob_true, quote

def predict_news(model, url, trained_tokenizer, max_len=2000):
    title, text = get_text_title(url)
    cleaned = clean_text(text)
    index = word2indx(cleaned, trained_tokenizer, max_len)
    prob_true, quote = predict_from_index(model, index)
    return prob_true, quote
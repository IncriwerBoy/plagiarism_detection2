import re
from string import punctuation
import pickle
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity



# with open('artifacts\model_category.pkl' , 'rb') as model:
#     rf_category = pickle.load(model)

# with open('artifacts\model_class.pkl' , 'rb') as model:
#     xgb_class = pickle.load(model)


# def read_text(file_name):
#     try:
#         with open(file_name, 'r', encoding='utf-8') as file:
#             return file.read()
#     except:
#         with open(file_name, 'r', encoding='latin1') as file:
#             return file.read()


def preprocess_text(text):
    text = text.lower()
    text = text.replace('\n', ' ')
    text = text.replace('\t', ' ')
    if "\'" in text:
        text = text.replace("\'", "'")
    if "\"" in text:
        text = text.replace("\"", "'")
        #remove punc
    for char in punctuation:
        text = text.replace(char, '')
    #remove numbers
    text = re.sub(r'\d+', '', text)
    return text


def ngram(n, a_text, o_text):
    
    ngram_array = CountVectorizer(analyzer='word', ngram_range=(n,n)).fit_transform([a_text, o_text]).toarray()
    return ngram_array


def containment(ngram_matrix):
    intersection = np.amin(ngram_matrix, axis=0)
    intersection = np.sum(intersection)
    total_ngrams = np.sum(ngram_matrix[0])
    return intersection / total_ngrams

def jaccard_similarity(ngram_matrix):
    intersection = np.amin(ngram_matrix, axis=0)
    intersection = np.sum(intersection)
    union = np.amax(ngram_matrix, axis=0)
    union = np.sum(union)
    return intersection / union


def tfidf_similarity(a_text, o_text):
    
    tfidf_array = TfidfVectorizer(analyzer='word', ngram_range=(2,2)).fit_transform([a_text, o_text])
    similarity  = cosine_similarity(tfidf_array[0], tfidf_array[1])[0][0]

    return similarity

def lcs(ansfile, origfile):
    a_text = ansfile.split()
    o_text = origfile.split()
    wordCount_a = len(a_text)
    wordCount_o = len(o_text)
    lcs_matrix = np.zeros((wordCount_a + 1, wordCount_o + 1), dtype=int)
    for a_idx, a_word in enumerate(a_text, 1):
        for o_idx, o_word in enumerate(o_text, 1):
            if a_word == o_word:
                lcs_matrix[a_idx][o_idx] = lcs_matrix[a_idx - 1][o_idx - 1] + 1
            else:
                lcs_matrix[a_idx][o_idx] = max(lcs_matrix[a_idx - 1][o_idx], lcs_matrix[a_idx][o_idx - 1])
    lcs_val = lcs_matrix[wordCount_a][wordCount_o]
    return lcs_val / wordCount_a


class Pipeline():
    def __init__(self, ansfile, origfile):
        self.ansfile = ansfile
        self.origfile = origfile

    def pipeline(self):
        a_text = preprocess_text(self.ansfile)
        o_text = preprocess_text(self.origfile)

        ngram_array1 = ngram(1, a_text, o_text)
        ngram_array5 = ngram(5, a_text, o_text)

        containment_1 = containment(ngram_array1)
        jaccard_5 = jaccard_similarity(ngram_array5)
        lcs_val = lcs(a_text, o_text)
        tfidf_2 = tfidf_similarity(a_text, o_text)

        features = np.array([lcs_val, tfidf_2, containment_1, jaccard_5]).reshape(1,-1)
        return features

    def plag_detection(self, xgb_class):
        features = self.pipeline()
        prediction = xgb_class.predict(features)[0]
        return prediction
    
    def plagType_detection(self, pred1, rf_category):
        features = self.pipeline()
        pred1 = np.array([[pred1]])
        features2 = np.concatenate((features, pred1), axis = 1)
        prediction2 = rf_category.predict(features2)
        return prediction2
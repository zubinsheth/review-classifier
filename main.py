import pandas as pd
import numpy as np
import nltk as nltk
import csv
import matplotlib.pyplot as plt
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict, OrderedDict
from nltk.corpus import wordnet as wn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import model_selection, naive_bayes, svm
from sklearn.metrics import accuracy_score
from wordcloud import WordCloud

## initializers, instances, global variables
np.random.seed(500)
word_Lemmatized = WordNetLemmatizer()

# add more custom stopwords
stopwords = stopwords.words('english')
my_file = open("stopwords.txt", "r")
moreStopWords = my_file.read().split(",")
my_file.close()
stopwords.extend(moreStopWords)

## data loading
datafile = "reviews.csv"  #can use test.csv for 1 review test
Corpus = pd.read_csv(datafile, error_bad_lines=False)


## data pre-processing
# Step - a : Remove blank rows if any.
Corpus['text'].dropna(inplace=True)
# Step - b : Change all the text to lower case. This is required as python interprets 'dog' and 'DOG' differently
Corpus['text'] = Corpus['text'].str.lower()
# Step - c : Tokenization : In this each entry in the corpus will be broken into set of words
Corpus['text']= [word_tokenize(entry) for entry in Corpus['text']]
# Step - d : Remove Stop words, Non-Numeric and perfom Word Stemming/Lemmenting.

# WordNetLemmatizer requires Pos tags to understand if the word is noun or verb or adjective etc. By default it is set to Noun
tag_map = defaultdict(lambda : wn.NOUN)
tag_map['J'] = wn.ADJ
tag_map['V'] = wn.VERB
tag_map['R'] = wn.ADV

for index,entry in enumerate(Corpus['text']):
    # Declaring Empty List to store the words that follow the rules for this step
    Final_words = []
    # pos_tag function below will provide the 'tag' i.e if the word is Noun(N) or Verb(V) or something else.
    for word, tag in pos_tag(entry):
        # Below condition is to check for Stop words and consider only alphabets
        if word not in stopwords and word.isalpha():
            word_Final = word_Lemmatized.lemmatize(word,tag_map[tag[0]])
            Final_words.append(word_Final)
    # The final processed set of words for each iteration will be stored in 'text_final'
    Corpus.loc[index,'text_final'] = str(Final_words)

# prepare Train and Test Data sets
Train_X, Test_X, Train_Y, Test_Y = model_selection.train_test_split(Corpus['text_final'],Corpus['label'],test_size=0.3)

# transform Categorical data of string type in the data set into numerical values which the model can understand
Encoder = LabelEncoder()
Train_Y = Encoder.fit_transform(Train_Y)
Test_Y = Encoder.fit_transform(Test_Y)

# TF-IDF are word frequency scores that try to highlight words that are more interesting.
# TF-IDF build a vocabulary of words which it has learned from the corpus data and 
# it will assign a unique integer number to each of these words

tfidf_param = {
    "sublinear_tf": False,  #False for smaller data size  #Apply sublinear tf scaling, to reduce the range of tf with 1 + log(tf)
    "ngram_range" : (1, 2),   #the min and max size of tokenized terms
    "max_features": 5000,    #the top 500 weighted features
    "stop_words": stopwords,
}

Tfidf_vect = TfidfVectorizer(**tfidf_param)
Tfidf_vect.fit(Corpus['text_final'])
Train_X_Tfidf = Tfidf_vect.transform(Train_X)
Test_X_Tfidf = Tfidf_vect.transform(Test_X)

# get the first vector out (for the first document)
first_vector_tfidfvectorizer=Train_X_Tfidf[0]

# place tf-idf values in a pandas data frame to see its score
df = pd.DataFrame(first_vector_tfidfvectorizer.T.todense(), index=Tfidf_vect.get_feature_names(), columns=["tfidf"])
df.sort_values(by=["tfidf"], ascending=False, inplace=True)

freqs_dict = dict([(word, Train_X_Tfidf.getcol(idx).sum()) for word, idx in Tfidf_vect.vocabulary_.items()])
w = WordCloud(width=800,height=600,mode='RGBA',background_color='white',max_words=500).fit_words(freqs_dict)
plt.figure(figsize=(12,9))
plt.imshow(w)
plt.axis("off")
#plt.show()


## ML Start

# NB: fit the training dataset on the Naive Bayes Classifier Algorithm
Naive = naive_bayes.MultinomialNB()
Naive.fit(Train_X_Tfidf,Train_Y)
# predict the labels on validation dataset
predictions_NB = Naive.predict(Test_X_Tfidf)
# Use accuracy_score function to get the accuracy
print("Naive Bayes Accuracy Score -> ",accuracy_score(predictions_NB, Test_Y)*100)


# SVM: fit the training dataset on the Support Vector Machine Classifier
SVM = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
SVM.fit(Train_X_Tfidf,Train_Y)
# predict the labels on validation dataset
predictions_SVM = SVM.predict(Test_X_Tfidf)
# Use accuracy_score function to get the accuracy
print("SVM Accuracy Score -> ",accuracy_score(predictions_SVM, Test_Y)*100)


## Test with new data
data = np.array(["Love the product. Great buy at low price", "Hated it. Battery is dead. Charger is slow."]) 
xNew = pd.Series(data)
xNew_Tfidf = Tfidf_vect.transform(xNew)
yNew = SVM.predict(xNew_Tfidf)
print("X=%s, Predicted=%s" % (xNew, yNew))

## end
print("DONE!!!")
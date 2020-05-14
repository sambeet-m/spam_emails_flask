from nltk.classify.util import apply_features
from nltk.tokenize import RegexpTokenizer
from nltk import NaiveBayesClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import os

from flask import current_app
import pickle, re
import collections

tokenizer = RegexpTokenizer(r'\w+')
class SpamClassifier:

    def load_model(self, model_name):
        model_file = os.path.join(current_app.config['ML_MODEL_UPLOAD_FOLDER'], model_name+'.pk')
        model_word_features_file = os.path.join(current_app.config['ML_MODEL_UPLOAD_FOLDER'],model_name +'_word_features.pk')
        with open(model_file, 'rb') as mfp:
            self.classifier = pickle.load(mfp)
        with open(model_word_features_file, 'rb') as mwfp:
            self.word_features = pickle.load(mwfp)


    def extract_tokens(self, text, target):
        """returns array of tuples where each tuple is defined by (tokenized_text, label)
         parameters:
                text: array of texts
                target: array of target labels

        NOTE: consider only those words which have all alphabets and atleast 3 characters.
        """
        corpus=[]
        for index,element in enumerate(text):
          tokenized_element = tokenizer.tokenize(element)
          tokenized_element_with_atleast3alapha = []
          for word in tokenized_element:
            if word.isalpha() and len(word) >= 3:
                #print("------------->", word,len(word),word.isalpha())
                tokenized_element_with_atleast3alapha.append(word)
        #print(tokenized_element_with_atleast3alapha)
          corpus.append((tokenized_element_with_atleast3alapha, target[index]))
        return corpus

    def get_features(self, corpus):
        """
        returns a Set of unique words in complete corpus.
        parameters:- corpus: tokenized corpus along with target labels

        Return Type is a set
        """
        cummulative_word_list = []
        total_words = 0
        for tuple_element in corpus:
            cummulative_word_list.extend(tuple_element[0])
            total_words = total_words + len(tuple_element[0])
        #print(len(set(cummulative_word_list)), total_words)
        return set(cummulative_word_list)

    def extract_features(self, document):
        """
        maps each input text into feature vector
        parameters:- document: string

        Return type : A dictionary with keys being the train data set word features.
                      The values correspond to True or False
        """
        features={}
        doc_words = set(document)
        #iterate through the word_features to find if the doc_words contains it or not
        for word in self.word_features:
            features[word] = (word in doc_words)
        #feature set
        return features

    def train(self, text, labels):
        """
        Returns trained model and set of unique words in training data
        """
        #call extract_tokens
        self.corpus=self.extract_tokens(text,labels)

		    #call get_features
        self.word_features=self.get_features(self.corpus)

		    #Extracting training set
        train_set = apply_features(self.extract_features, self.corpus)

		    #Now train the NaiveBayesClassifier with train_set
        self.classifier = NaiveBayesClassifier.train(train_set)

        return self.classifier, self.word_features


    def predict(self, text):
        """
        Returns prediction labels of given input text.
        """
        if isinstance(text, (list)):
            pred = []
            for sentence in list(text):
                pred.append(self.classifier.classify(self.extract_features(sentence.split())))
            return pred
        if isinstance(text, (collections.OrderedDict)):
            pred = collections.OrderedDict()
            for label, sentence in text.items():
                pred[label] = self.classifier.classify(self.extract_features(sentence.split()))
            return pred
        return self.classifier.classify(self.extract_features(text.split()))


if __name__ == '__main__':

    print('Done')
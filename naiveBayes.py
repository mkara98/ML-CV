from sklearn.model_selection import train_test_split
import re
import nltk
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
import sys
import numpy as np
np.set_printoptions(threshold=sys.maxsize)

file1 = open('all_sentiment_shuffled.txt', encoding="utf8")
lines = file1.readlines()


def cleanWords(sentence):
    string = ""
    tokens = [word.lower() for word in sentence]  # lower case
    words = [word for word in tokens if word.isalpha()]  # remove not alphabetic
    stop_words = stopwords.words()
    words = [w for w in words if not w in stop_words]  # remove stopwords
    for w in words:
        string += w + " "
    return string


class NaiveBayes:
    def __init__(self):
        pass

    def fit(self, bowTrain, y_train):
        totalSample, bowLength = bowTrain.shape

        self.classes = np.unique(y_train)
        self.category = np.zeros(2)
        self.bowVector = np.zeros((2, bowLength))

        for idx, c in enumerate(self.classes):
            trainArray = np.array([bowTrain[i] for i in range(len(y_train)) if y_train[i] == c])
            wordsTotal = np.array([np.sum(trainArray[:, i]) for i in range(bowLength)])
            self.category[idx] = trainArray.shape[0] / totalSample
            self.bowVector[idx, :] = wordsTotal + 1 / (
                    np.sum(wordsTotal) + 1)

    def predict(self, testData):

        pred = []
        for test in testData:
            pred.append(self.testBayes(test))
        return pred

    def testBayes(self, test):

        predicts = []
        for idx, c in enumerate(self.classes):
            totalProb = np.log(self.category[idx])
            testProb = self.calculate(self.bowVector[idx, :], test)
            finalProb = np.sum(testProb) + totalProb
            predicts.append(finalProb)

        return self.classes[np.argmax(predicts)]

    def calculate(self, probOfWord, test):
        return np.log(probOfWord) * test

    def accuracy(self, y_pred, y_test):
        total_acc = 0
        wrongClassify = []
        for i in range(len(y_pred)):
            if y_pred[i] == y_test[i]:
                total_acc += 1
            else:
                wrongClassify.append(i)

        return (total_acc / len(y_pred)) * 100, wrongClassify


categoryLabel = []
allSentences = []


def specificKeywords(contentString, keyword, categoryType):
    try:
        contentString.index(keyword)
        keywords.append(categoryTypew)
    except:
        pass


def countWord(sentences, category, word):
    pos = 0
    neg = 0
    for i in range(len(sentences)):
        try:
            sentences[i].index(word)
            if category[i] == "pos":
                pos += 1
            else:
                neg += 1
        except:
            pass
    print(word, "is used in ", pos, "positive comment")
    print(word, "is used in ", neg, "negative comment")


for line in lines:
    content = line.split(" ")
    topic = content[0]
    category = content[1]
    identifier = content[2]
    tokens = cleanWords(content[3:])
    categoryLabel.append(category)
    allSentences.append(tokens)

np.save("All", allSentences)
np.save("category", categoryLabel)
allSentences = np.load("All.npy", allow_pickle=True)
categoryLabel = np.load("Category.npy", allow_pickle=True)
countWord(allSentences, categoryLabel, "great")
vectorizer = CountVectorizer()
bow = vectorizer.fit_transform(allSentences).toarray()
trainBow, testBow, Y_train, Y_test = train_test_split(bow, categoryLabel, test_size=0.20)
bayes = NaiveBayes()
bayes.fit(trainBow, Y_train)
pred = bayes.predict(testBow)
score, wrong = bayes.accuracy(pred, Y_test)
print(score)
for i in range(10):
    index = 0
    for j in range(len(bow)):
        if bow[j].all() == testBow[wrong[i]].all():
            index = i
    print(lines[index])

import nltk
import pickle
import math
from math import log

#importing the SPAM files corpus
#Sources :https://canvas.ucdavis.edu/courses/205264/pages/nltk-demo
spam = nltk.corpus.PlaintextCorpusReader('Messages.txt','SPAM.*')
list_spam = spam.fileids()
spam_count = len(list_spam)
spam_words = spam.words()
for tokens in spam_words:
    print(tokens)

#finding the frequency distribution of all the words in the SPAM corpus
spam_fd = nltk.FreqDist(spam_words)
spam_fd

# Importing the HAM files corpus
#https://canvas.ucdavis.edu/courses/205264/pages/nltk-demo
ham = nltk.corpus.PlaintextCorpusReader('Messages.txt','HAM.*')
list_ham = ham.fileids()
ham_count = len(list_ham)
ham_words = ham.words()
for token in ham_words:
    print(token)

#finding the frequency distribution of all the words in the HAM corpus
ham_fd = nltk.FreqDist(ham_words)
ham_fd

# Creating dictionary the model with our (reading in the files and their frequency distributions)
model = model = {
 'ham_count': ham_count,
 'spam_count': spam_count,
 'ham_fd': ham_fd,
 'spam_fd': spam_fd
}

# Opening a file f and dumping the spam.nb using pickle
#https://python.swaroopch.com/io.html
pickle.dump(model, open('spam.nb', 'wb'))

#Loading the file in using pikle load fuction.

with open('spam.nb','rb') as f:
    m = pickle.load(f)   #m gives the dictionary model


#importing the spam_test corpus
test = nltk.corpus.PlaintextCorpusReader('Messages.txt','TEST.*')

lis_test = test.fileids()
for files in lis_test:
    print(files)
    for files in lis_test:
        test_word = test.words(files)
        print(test_word)

#calculating the priors for calculating the ham_score
ham_sc = math.log(model['ham_count'])/(model['spam_count']+ model['ham_count'])

#calculating the priors for calculating the spam_score
spam_sc =  math.log(model['spam_count'])/(model['spam_count']+ model['ham_count'])

#number types of model add one smoothing
num_types = (model['spam_fd']+m['spam_fd']).B()

#intialization of ham_sc & spam_sc
ham_score = ham_sc
spam_score = spam_sc

#putting everything in that word loop for every file
for w in test_word:
    cond_ham = model['ham_fd'][w] / model['ham_fd'].N()
    ham_score = ham_score + math.log(cond_ham)
    print(ham_score)
    cond_spam = (1+model['spam_fd'][w]) / (model['spam_fd'].N() + num_types)
    print(cond_spam)
    spam_score = spam_score + math.log(cond_spam)
    print(spam_score)

# printing the final statement
if spam_score < ham_score:
    print(str(files)+' '+'HAM')
else:
    print(str(files+' '+'SPAM'))

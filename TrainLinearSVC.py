from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, recall_score

import pandas as pd
import numpy as np
import pickle

#extra words to exclude
extra = ['inc', 'lp', '&', 'llc', 'sc', ',', '-']
stop = extra

'''targets'''
fileloc = 'TrainingData.csv'
pickleloc = '\\text_class\\pickleJar\\'

#set dtype to read file faster
acct_trans = pd.read_csv(fileloc,
                    header = 0,
                    usecols = ['Data',
                               'LabelsForData',
                    dtype = {'Data': str,
                             'LabelsForData': str}])


# large C value will choose smaller-margin hyperplane ---> strives to label data more finely
# small C value will choose larger-margin hyperplane ---> strives to label data more broadly
text_clf = Pipeline([
    ('vect', TfidfVectorizer(ngram_range = (1,3), stop_words = stop, analyzer = 'word')),
    ('clf', LinearSVC(C=0.3))
])


#create train/test data
d_train, d_test, l_train, l_test = train_test_split(acct_trans['Data'], #data
                                                    acct_trans['LabelsForData']) #labels

#fit model
features = text_clf.fit(d_train, l_train)
#predict values from model
predicted = text_clf.predict(d_test)

#how often the model is right against labels
accuracy = np.round(accuracy_score(l_test, predicted) * 100, decimals = 2)
#quality of labels
recall = np.round(recall_score(l_test, predicted) * 100, decimals = 2)
#data being trained vs total data set
pct_target = np.round(np.mean(acct_trans[target[no]]) * 100, decimals = 2)
print('% of data related to',target[no],': ', pct_target,'%')

print(f'Accuracy {accuracy}%')
print(f'Precision of guess {recall}%')

'''Save model'''
#pickle saves pipeline, wb = write in bytes for >py2
#to save
with open(pickleloc+task+target[no][:2]+'.pickle', 'wb') as save_classifier:
    pickle.dump(text_clf, save_classifier)

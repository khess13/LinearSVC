import pandas as pd
import pickle
import datetime as dt

'''targets'''
fileloc = '.csv'
exportloc = '\\text_class\\'
pickleloc = '\\text_class\\pickleJar\\'

datestamp = dt.datetime.now().strftime('%m-%d-%Y')

#read file, dtype specs for faster read
print(f'Processing {ITRE_lab}')
acct_trans = pd.read_csv(fileloc,
                    header = 0,
                    encoding = 'ISO-8859-1', #b/c excel update
                    usecols = ['Data',
                               'Data1',
                               'Data2'],
                      dtype = {'Data': str,
                               'Data1':str,
                               'Data2':str})

#opening pickeled classifier
print(f'Loading classifier')
text_clf = pickle.load(open(pickleloc + '.pickle', 'rb'))
new_data = acct_trans['Data'].copy()

print(f'Predicting labels')
#use loaded classifier to predict labels
predicted = text_clf.predict(new_data)

label = ['No','Yes']
res_data = []
res_lis = []
print('Building list of results')
for data, label in zip(new_data, predicted):
    res_data.append(data)
    #indexing prediction from list, 0 = No, 1 = Yes
    res_lis.append(IT_label[label])


print('Rebuilding dataframe with predicted labels')
#append to df
res_df = pd.DataFrame({'Result' : res_lis})
acct_trans.update(res_df)

with pd.ExcelWriter(exportloc+datestamp+'.xlsx') as writer:
    #write dfs to excel file
    acct_trans.to_excel(writer, sheet_name = 'LabeledData', index = False)
print('Excel exported!')

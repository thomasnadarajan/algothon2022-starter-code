import numpy as np
import pandas as pd
import math

df1 = pd.read_csv('prices.txt', delim_whitespace=True, header = None)

pool = []
indexes = []
trades_stack = []
used_starts = set()
while len(used_starts) < len(df1.columns):
    if len(pool) < 28:
        pool = []
    for i in range(len(df1.columns)):
        if len(pool) == 28:
            indexes.append(np.array(pool))
            pool = []
        store = df1.iloc[:, i].values
        '''
        new_store = list()
        for j in range(1, len(store)):
            new_store.append(((store[j] - store[j-1])/store[j-1]) * 100)
        '''
        if pool == [] and i not in used_starts:
            #pool.append(store)
            pool.append(i)
            used_starts.add(i)
        elif i in used_starts:
            continue
        if len(pool) < 28 and i not in pool:
            found = False
            for data in pool:
                correl = np.corrcoef(df1.iloc[:, data].values, store)[0][1]
                if (correl < 0.8 and correl >= 0) or (correl > -0.8 and correl < 0):
                    found = True
            if not found:
                #pool.append(store)
                pool.append(i)

#print(len(indexes))
'''
for index in indexes:
    print("new index")
    #print(index)
    seen = set()
    avg_correlation = 0
    for i in range(len(index)):
        for j in range(len(index)):
            if i != j:
                if (i, j) not in seen and (j, i) not in seen:
                    #print(np.corrcoef(df1.iloc[:,index[i]], df1.iloc[:,index[j]])[0][1])
                    avg_correlation += np.corrcoef(df1.iloc[:,index[i]], df1.iloc[:,index[j]])[0][1]
                    seen.add((i, j))
    avg_correlation /= len(seen)
    print(index)
    print(avg_correlation)
'''
#index_array = np.array(model_data).sum(axis=0) / len(model_data)
correlations = []
'''
for instrument in model_data:
    correlations.append(np.corrcoef(instrument, index_array))
    print(correlations[-1])
'''
def differenced(prcSoFar, currentPos, unregressed):
    global model_data
    global trades_stack
    for i in range(prcSoFar.shape[0]):
        #if i in unregressed:
        store = prcSoFar[i,:]
        if len(trades_stack[i]) > 0:
            marked = []
            for trade in trades_stack[i]:
                if len(store) == trade[1] + 4:
                    currentPos[i] += trade[0] * -1
                    marked.append(trade)
            for mark in marked:
                trades_stack[i].remove(mark)
        
    return currentPos
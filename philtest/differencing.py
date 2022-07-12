import numpy as np
import pandas as pd
import scipy.stats as stats

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


normalised_indexes = []
for index in indexes:
    temp_store = []
    for i in range(len(index)):
        store = df1.iloc[:, index[i]].values
        new_store = list()
        for j in range(1, len(store)):
            new_store.append(((store[j] - store[j-1])/store[j-1]))
        #print(new_store)
        temp_store.append(np.array(new_store))
    #print(temp_store)
    temp_store = np.array(temp_store)
    temp_store = temp_store.sum(axis=0) / len(index)
    normalised_indexes.append(temp_store)
regressions = {}
for ind in range(len(indexes)):
    for i in range(len(indexes[ind])):
        store = df1.iloc[:, indexes[ind][i]].values
        new_store = list()
        for j in range(1, len(store)):
            new_store.append(((store[j] - store[j-1])/store[j-1]))
        if indexes[ind][i] in regressions:
            regressions[indexes[ind][i]][ind] = stats.linregress(new_store, normalised_indexes[ind]).slope
        else:
            regressions[indexes[ind][i]] = {}
            regressions[indexes[ind][i]][ind] = stats.linregress(new_store, normalised_indexes[ind]).slope
        
    
def differenced(prcSoFar, currentPos):
    global regressions
    global trades_stack
    for i in range(prcSoFar.shape[0]):
        '''
        if len(trades_stack[i]) > 0:
            marked = []
            for trade in trades_stack[i]:
                if len(store) == trade[1] + 4:
                    currentPos[i] += trade[0] * -1
                    marked.append(trade)
            for mark in marked:
                trades_stack[i].remove(mark)
        '''
        store_a = prcSoFar[i,:]
        if len(store_a) > 1:
            for j in range(len(indexes)):
                if i in indexes[j]:
                    store_b = normalised_indexes[j]
                    scalar = regressions[i][j]
                    if store_b[len(store_b) - 1] - store_b[len(store_b) - 2] / store_a[len(store_a) - 1]- store_a[len(store_a) - 2] > 1.15 * scalar:
                        # volume test 1
                        '''
                        trade_a = round((1000/scalar)/store_b[len(store_b) - 1], 0)
                        trade_b = round(-1000/store_b[len(store_b) - 1], 0)
                        '''
                        # volume test 2
                        
                        
                        #trade_b = -1000 * (store_b[len(store_b) - 1] - store_b[len(store_b) - 2] / store_a[len(store_a) - 1] - store_a[len(store_a) - 2]) / (1.2 * scalar)
                        trade_b = -1000
                        trade_a = (trade_b * -1) / scalar
                        #trade_b = round(trade_b/store_b[len(store_b) - 1], 0)
                        trade_a = round(trade_a/store_a[len(store_a) - 1], 0)
                        for k in indexes[j]:
                            if k != i:
                                prices_k = prcSoFar[k,:]
                                currentPos[k] = round((trade_b / (len(indexes[j]) - 1)) / prices_k[-1], 0)
                    elif store_b[len(store_b) - 1] - store_b[len(store_b) - 2] / store_a[len(store_a) - 1] - store_a[len(store_a) - 2] < 0.85 * scalar:
                        # volume test 1
                        '''
                        trade_a = round((-1000/scalar)/store_b[len(store_b) - 1], 0)
                        trade_b = round(1000/store_b[len(store_b) - 1], 0)
                        '''
                        # volume test 2
                        #trade_b = 10 * (store_b[len(store_b) - 1] - store_b[len(store_b) - 2] / store_a[len(store_a) - 1] - store_a[len(store_a) - 2]) / (1.2 * scalar)
                        trade_b = 1000
                        trade_a = (trade_b * -1) / scalar
                        #trade_b = round(trade_b/store_b[len(store_b) - 1], 0)
                        trade_a = round(trade_a/store_a[len(store_a) - 1], 0)
                        for k in indexes[j]:
                            if k != i:
                                prices_k = prcSoFar[k,:]
                                currentPos[k] = round((trade_b / (len(indexes[j]) - 1)) / prices_k[-1], 0)
                    '''
                    trades_stack[i].append((trade_a, len(store_a)))
                    trades_stack[j].append((trade_b, len(store_b)))
                    '''
                    currentPos[i] += trade_a

                
            # Build your function body here
        
    return currentPos
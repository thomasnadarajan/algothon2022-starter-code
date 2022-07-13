import numpy as np
import pandas as pd
import scipy.stats as stats

df1 = pd.read_csv('prices.txt', delim_whitespace=True, header = None)
interval = 300
code = 0
indexes = {}
normalised_indexes = {}
regressions = {}
closed_indexes = []
def calculate_indexes(additional = None):
    global code
    global indexes
    global df1
    
    pool = []
    used_starts = set()
    if additional is not None:
        pd.concat([df1, pd.DataFrame(additional)], axis=0, ignore_index=True)
    while len(used_starts) < len(df1.columns):
        if len(pool) < 28:
            pool = []
        for i in range(len(df1.columns)):
            if len(pool) == 28:
                indexes[code] = np.array(pool)
                pool = []
                code += 1
            store = df1.iloc[:, i].values
            if pool == [] and i not in used_starts:
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
                    pool.append(i)
    for index in indexes:
        temp_store = []
        for i in range(len(indexes[index])):
            store = df1.iloc[:, indexes[index][i]].values
            new_store = list()
            for j in range(1, len(store)):
                new_store.append(((store[j] - store[j-1])/store[j-1]))
            #print(new_store)
            temp_store.append(np.array(new_store))
        #print(temp_store)
        temp_store = np.array(temp_store)
        temp_store = temp_store.sum(axis=0) / len(indexes[index])
        normalised_indexes[index] = temp_store

def calculate_regressions():
    global regressions
    for ind in indexes:
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
        

calculate_indexes()
calculate_regressions()

def differenced(prcSoFar, currentPos, trades_stack):
    global regressions
    global interval
    if interval == prcSoFar.shape[1]:
        calculate_indexes(prcSoFar)
        calculate_regressions()
        interval += 100
    for i in range(prcSoFar.shape[0]):
        store_a = prcSoFar[i,:]
        if len(store_a) > 1:
            for index in indexes:
                if i in indexes[index]:
                    store_b = normalised_indexes[index]
                    scalar = regressions[i][index]
                    if store_b[len(store_b) - 1] - store_b[len(store_b) - 2] / store_a[len(store_a) - 1]- store_a[len(store_a) - 2] > 1.15 * scalar:
                        # volume test 2
                        
                        trade_b = -1000 * (store_b[len(store_b) - 1] - store_b[len(store_b) - 2] / store_a[len(store_a) - 1] - store_a[len(store_a) - 2]) / (1 * scalar)
                        trade_a = (trade_b * -1) / scalar
                        trade_a = round(trade_a/store_a[len(store_a) - 1], 0)
                        for k in indexes[index]:
                            if k != i:
                                prices_k = prcSoFar[k,:]
                                currentPos[k] = round((trade_b / (len(indexes[index]) - 1)) / prices_k[-1], 0)
                                trades_stack[k].append((trade_b, len(store_a), 'index'))

                        trades_stack[i].append((trade_a, len(store_a)))
                    elif store_b[len(store_b) - 1] - store_b[len(store_b) - 2] / store_a[len(store_a) - 1] - store_a[len(store_a) - 2] < 0.85 * scalar:
                        # volume test 2
                        trade_b = 1000 * (store_b[len(store_b) - 1] - store_b[len(store_b) - 2] / store_a[len(store_a) - 1] - store_a[len(store_a) - 2]) / (1 * scalar)
                        trade_a = (trade_b * -1) / scalar
                        trade_a = round(trade_a/store_a[len(store_a) - 1], 0)
                        for k in indexes[index]:
                            if k != i:
                                prices_k = prcSoFar[k,:]
                                currentPos[k] = round((trade_b / (len(indexes[index]) - 1)) / prices_k[-1], 0)
                                trades_stack[k].append((trade_b, len(store_a), 'index'))
                        trades_stack[i].append((trade_a, len(store_a), 'index'))
                    
                    currentPos[i] += trade_a

                
            # Build your function body here
    return (currentPos, trades_stack)

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

'''
for instrument in model_data:
    correlations.append(np.corrcoef(instrument, index_array))
    print(correlations[-1])
'''
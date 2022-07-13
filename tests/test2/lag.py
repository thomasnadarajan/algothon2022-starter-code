import numpy as np
import pandas as pd
import math
df2 = pd.read_csv('prices.txt', delim_whitespace=True, header = None)

def normal_dist(data):
    
    mean = np.average(data)
    sd = math.sqrt(np.average((data - mean) ** 2))
    
    return mean, sd


means1 = {}
sds1 = {}
means5 = {}
sds5 = {}
counter = 300
def update_stats(additional = None):
    global df2
    global means1
    global sds1
    global means5
    global sds5
    if additional:
        pd.concat([df2, pd.DataFrame(additional)], axis =0, ignore_index=True)
    for i in range(100):
        temp1 = []
        for j in range(1,len(df2.iloc[:, i])):
            temp1.append((df2.iloc[:, i][j] - df2.iloc[:, i][j-1]) / df2.iloc[:, i][j-1])
        temp2 = []
        for j in range(1,len(df2.iloc[:, i])):
            if j - 3 >= 0:
                temp2.append(df2.iloc[:, i][j] - df2.iloc[:, i][j-3])
        means1[i] = normal_dist(temp1)[0]
        sds1[i] = normal_dist(temp1)[1]
update_stats()

lagtimes = {}

def lag_time_finder(i, j):
    col_i = df2.iloc[:, i]
    col_j = df2.iloc[:, j]
    correlations = []
    for x in range(1, 21):
        sliced_i = col_i[x:]
        sliced_j = col_j[:-x]
        corr = np.corrcoef(sliced_i, sliced_j)[0][1]
        correlations.append((x, corr))
    return max(correlations, key=lambda x: x[1])

for i in range(100):
    for j in range(100):
        if i != j:
            lagtimes[(i, j)] = lag_time_finder(i, j)
def lag_trade(prcSoFar, currentPos, trades_stack):
    global means1
    global sds1
    global means5
    global sds5
    for i in range(prcSoFar.shape[0]):
        for j in range(prcSoFar.shape[0]):
            if i != j:
                lag = lagtimes[(i, j)]
                if lag[1] > 0.8 or lag[1] < -0.8:
                    store_i = prcSoFar[i,:]
                    store_j = prcSoFar[j,:]
                    if len(store_j) < lag[0] + 3:
                        continue
                    delta_j = store_j[-lag[0] - 2] - store_j[-lag[0] - 3] / store_j[-lag[0] - 3]
                    if lag[1] > 0:
                        if means1[j] < 0 and delta_j < means1[j] - 2 * sds1[j]:
                            currentPos[i] += round(-3500/store_i[-1], 0)
                            trades_stack[i].append((round(-3500/store_i[-1], 0), prcSoFar.shape[1], 'lag'))
                        if means1[j] > 0 and delta_j > means1[j] + 2 * sds1[j]:
                            currentPos[i] += round(+3500 / store_i[-1], 0)
                            trades_stack[i].append((round(+3500 / store_i[-1], 0), prcSoFar.shape[1], 'lag'))
                        
                    
                    else:
                        if means1[j] < 0 and delta_j < means1[j] - 2 * sds1[j]:
                            currentPos[i] += round(+3500/store_i[-1], 0)
                            trades_stack[i].append((round(+3500/store_i[-1], 0), prcSoFar.shape[1], 'lag'))
                        if means1[j] > 0 and delta_j > means1[j] + 2 * sds1[j]:
                            currentPos[i] += round(-3500 / store_i[-1], 0)
                            trades_stack[i].append((round(-3500 / store_i[-1], 0), prcSoFar.shape[1], 'lag'))

                    
                    '''
                    if len(store_j) < lag[0] + 5:
                        continue
                    delta_j = store_j[-lag[0] - 2] - store_j[-lag[0] - 5] / store_j[-lag[0] - 5]
                    if delta_j > means5[j] + 2 * sds5[j] and means5[j] + 2 * sds5[j] > 0 :
                        currentPos[i] += round(1000 / store_i[-1], 0)
                        #trades_stack[i].append((round(1000/store_i[-1], 0), prcSoFar.shape[1], 'lag'))
                    elif delta_j < means5[j] - 2 * sds5[j] and means5[j] - 2 * sds5[j] < 0:
                        currentPos[i] -= round(-1000/store_i[-1], 0)
                        #trades_stack[i].append((round(-1000/store_i[-1], 0), prcSoFar.shape[1], 'lag'))
                    '''

    return (currentPos, trades_stack)
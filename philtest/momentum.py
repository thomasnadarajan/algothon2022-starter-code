import pandas as pd
import numpy as np
import math
df2 = pd.read_csv('prices.txt', delim_whitespace=True, header = None)

def weighted_normal(data, weights):
    
    mean = np.average(data, weights= weights)
    sd = math.sqrt(np.average((data - mean) ** 2, weights = weights))
    
    return mean, sd


weight_i = []
means = {}
sds = {}
counter = 300
def update_weights(additional = None):
    global df2
    global weight_i
    global mean_i
    global sd_i
    if additional:
        pd.concat([df2, pd.DataFrame(additional)], axis =0, ignore_index=True)
    for i in range(100):
        weight_i = list(range(len(df2.iloc[:, i])))
        weight_i.pop(0)
        temp = []
        for j in range(1,len(df2.iloc[:, i])):
            temp.append(df2.iloc[:, i][j] - df2.iloc[:, i][j-1])
        means[i] = (weighted_normal(temp, weight_i)[0])
        sds[i] = (weighted_normal(temp, weight_i)[1])
update_weights()
def momentumChangeDetect(prcSoFar, currentPos, trades_stack):
    if prcSoFar.shape[1] == counter:
        update_weights(prcSoFar)
    for i in range(prcSoFar.shape[0]):
        store = prcSoFar[i,:]
        if len(store) > 1:
            if store[-1] - store[-2] > means[i] + 3 * sds[i]:
                currentPos[i] -= 1000
                trades_stack[i].append((-1000, prcSoFar.shape[1], 'momentum'))
            elif store[-1] - store[-2] < means[i] - 3 * sds[i]:
                currentPos[i] += 1000
                trades_stack[i].append((1000, prcSoFar.shape[1], 'momentum'))
    return (currentPos, trades_stack)
            



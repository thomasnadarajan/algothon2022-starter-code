import numpy as np
import pandas as pd
import scipy.stats as stats
from differencing import differenced
df = pd.read_csv('prices.txt', delim_whitespace=True, header = None)

model_data = []
regression_gradients = []
trades_stack = []
unregressed = []
for i in range(len(df.columns)):
    trades_stack.append([])
    store_a = df.iloc[:, i].values
    for j in range(len(df.columns)):
        store_b = df.iloc[:, j].values
        if i != j:
            b_1 = stats.linregress(store_a, store_b).slope
            if (b_1 >= 0.96 or b_1 <= -0.96):
                regression_gradients.append((b_1, i, j))
                if i in unregressed:
                    unregressed.remove(i)
                if j in unregressed:
                    unregressed.remove(j)
            else:
                unregressed.append(i)
                unregressed.append(j)

nInst=100
currentPos = np.zeros(nInst)


def getMyPosition (prcSoFar):
    global currentPos
    global regression_gradients
    currentPos = differenced(prcSoFar, currentPos)
    
    for regress in regression_gradients:
        scalar = regress[0]
        i = regress[1]
        j = regress[2]
        store_a = prcSoFar[i,:]
        store_b = prcSoFar[j,:]
        '''
        if len(trades_stack[i]) > 0:
            if store_b[len(store_b) - 1] - store_b[len(store_b) - 2] / store_a[len(store_a) - 1]- store_a[len(store_a) - 2] < 1.01 * scalar and store_b[len(store_b) - 1] - store_b[len(store_b) - 2] / store_a[len(store_a) - 1]- store_a[len(store_a) - 2] > 0.99 * scalar:
                for trade in trades_stack[i]:
                    currentPos[i] += trade[0] * -1
                trades_stack[i] = []
        if len(trades_stack[j]) > 0:
            if store_b[len(store_b) - 1] - store_b[len(store_b) - 2] / store_a[len(store_a) - 1]- store_a[len(store_a) - 2] < 1.01 * scalar and store_b[len(store_b) - 1] - store_b[len(store_b) - 2] / store_a[len(store_a) - 1]- store_a[len(store_a) - 2] > 0.99 * scalar:
                for trade in trades_stack[j]:
                    currentPos[j] += trade[0] * -1
                trades_stack[j] = []
        '''
        if len(store_a) > 1:
            if store_b[len(store_b) - 1] - store_b[len(store_b) - 2] / store_a[len(store_a) - 1]- store_a[len(store_a) - 2] > 1.15 * scalar:
                # volume test 1
                '''
                trade_a = round((1000/scalar)/store_b[len(store_b) - 1], 0)
                trade_b = round(-1000/store_b[len(store_b) - 1], 0)
                '''
                # volume test 2
                
                trade_b = -10 * (store_b[len(store_b) - 1] - store_b[len(store_b) - 2] / store_a[len(store_a) - 1] - store_a[len(store_a) - 2]) / (1.2 * scalar)
                trade_a = (trade_b * -1) / scalar
                trade_b = round(trade_b/store_b[len(store_b) - 1], 0)
                trade_a = round(trade_a/store_a[len(store_a) - 1], 0)
                trades_stack[i].append((trade_a, len(store_a)))
                trades_stack[j].append((trade_b, len(store_b)))
            elif store_b[len(store_b) - 1] - store_b[len(store_b) - 2] / store_a[len(store_a) - 1] - store_a[len(store_a) - 2] < 0.85 * scalar:
                # volume test 1
                '''
                trade_a = round((-1000/scalar)/store_b[len(store_b) - 1], 0)
                trade_b = round(1000/store_b[len(store_b) - 1], 0)
                '''
                # volume test 2
                trade_b = 10 * (store_b[len(store_b) - 1] - store_b[len(store_b) - 2] / store_a[len(store_a) - 1] - store_a[len(store_a) - 2]) / (1.2 * scalar)
                trade_a = (trade_b * -1) / scalar
                trade_b = round(trade_b/store_b[len(store_b) - 1], 0)
                trade_a = round(trade_a/store_a[len(store_a) - 1], 0)
                trades_stack[i].append((trade_a, len(store_a)))
                trades_stack[j].append((trade_b, len(store_b)))
        
            currentPos[i] += trade_a
            currentPos[j] += trade_b

        
    # Build your function body here

    return currentPos

    

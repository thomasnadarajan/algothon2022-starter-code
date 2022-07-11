import numpy as np
import pandas as pd
import math
df = pd.read_csv('prices.txt', delim_whitespace=True, header = None)

model_data = []
regression_gradients = []
trades_stack = []
for i in range(len(df.columns)):
    trades_stack.append([])
    store_a = df.iloc[:, i].values
    for j in range(len(df.columns)):
        store_b = df.iloc[:, j].values
        if i != j:
            mean_a = np.mean(store_a)
            mean_b = np.mean(store_b)
            SS_ab = np.sum(store_b * store_a) - len(store_a) * mean_a * mean_b
            SS_aa = np.sum(store_a * store_a) - len(store_a) * mean_a * mean_a

            b_1 = SS_ab / SS_aa
            if (b_1 >= 0.9 or b_1 <= -0.9):
                regression_gradients.append((b_1, i, j))

nInst=100
currentPos = np.zeros(nInst)


def getMyPosition (prcSoFar):
    global currentPos
    global regression_gradients
    for regress in regression_gradients:
        scalar = regress[0]
        i = regress[1]
        j = regress[2]
        store_a = prcSoFar[i,:]
        store_b = prcSoFar[j,:]
        if len(trades_stack[i]) > 0:
            marked = []
            for trade in trades_stack[i]:
                if len(store_a) == trade[1] + 8:
                    currentPos[i] += trade[0] * -1
                    marked.append(trade)
            for trade in marked:
                trades_stack[i].remove(trade)
        if len(trades_stack[j]) > 0:
            marked = []
            for trade in trades_stack[j]:
                if len(store_b) == trade[1] + 8:
                    currentPos[j] += trade[0] * -1
                    marked.append(trade)
            for trade in marked:
                trades_stack[j].remove(trade)
        if len(store_a) > 1:
            if store_b[len(store_b) - 1] - store_b[len(store_b) - 2] / store_a[len(store_a) - 1]- store_a[len(store_a) - 2] > 1.2 * scalar:
                # volume test 1
                '''
                trade_a = round((1000/scalar)/store_b[len(store_b) - 1], 0)
                trade_b = round(-1000/store_b[len(store_b) - 1], 0)
                '''
                # volume test 2
                
                trade_b = -1000 * (store_b[len(store_b) - 1] - store_b[len(store_b) - 2] / store_a[len(store_a) - 1] - store_a[len(store_a) - 2]) / (1.2 * scalar)
                trade_a = (trade_b * -1) / scalar
                trade_b = round(trade_b/store_b[len(store_b) - 1], 0)
                trade_a = round(trade_a/store_a[len(store_a) - 1], 0)
                trades_stack[i].append((trade_a, len(store_a) - 1))
                trades_stack[j].append((trade_b, len(store_b) - 1))
            elif store_b[len(store_b) - 1] - store_b[len(store_b) - 2] / store_a[len(store_a) - 1] - store_a[len(store_a) - 2] < 0.8 * scalar:
                # volume test 1
                '''
                trade_a = round((-1000/scalar)/store_b[len(store_b) - 1], 0)
                trade_b = round(1000/store_b[len(store_b) - 1], 0)
                '''
                # volume test 2
                trade_b = 1000 * (store_b[len(store_b) - 1] - store_b[len(store_b) - 2] / store_a[len(store_a) - 1] - store_a[len(store_a) - 2]) / (1.2 * scalar)
                trade_a = (trade_b * -1) / scalar
                trade_b = round(trade_b/store_b[len(store_b) - 1], 0)
                trade_a = round(trade_a/store_a[len(store_a) - 1], 0)
                trades_stack[i].append((trade_a, len(store_a) - 1))
                trades_stack[j].append((trade_b, len(store_b) - 1))
        
            currentPos[i] += trade_a
            currentPos[j] += trade_b

        
    # Build your function body here

    return currentPos

    

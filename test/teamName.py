import numpy as np
import pandas as pd
import math
df = pd.read_csv('prices.txt', delim_whitespace=True, header = None)

model_data = []
moving_averages = []
trades_stack = []
for i in range(len(df.columns)):
    trades_stack.append([])
    store = df.iloc[:, i].values
    cumul_sum = np.cumsum(store)
    moving_averages.append([store[0]])
    new_store = list()
    for j in range(1, len(store)):
        window_average = round(cumul_sum[j-1] / j, 2)
        moving_averages[len(moving_averages) - 1].append(window_average)
        new_store.append(((store[j] - store[j-1])/store[j-1]) * 100)
    avg = sum(new_store)/len(new_store)
    std_dv = 0
    for i in new_store:
        std_dv += (i - avg) ** 2
    std_dv /= len(new_store)
    std_dv = math.sqrt(std_dv)
    q3, q1 = np.percentile(new_store, [75 ,25])
    iqr = q3 - q1
    model_data.append({'avg': avg, 'dv': std_dv, 'iqr': iqr, 'q3': q3, 'q1': q1})

nInst=100
currentPos = np.zeros(nInst)


def getMyPosition (prcSoFar):
    global currentPos
    global model_data
    for i in range(prcSoFar.shape[0]):
        store = prcSoFar[i,:]
        if len(store) > 1:
            if len(trades_stack[i]) > 0:
                q = 0
                while q < len(trades_stack[i]):
                    if q >= len(trades_stack[i]):
                        break
                    if trades_stack[i][q][1] < 0:
                        if store[len(store) - 1] <= moving_averages[i][trades_stack[i][q][2]] and moving_averages[i][trades_stack[i][q][2]] < trades_stack[i][q][0]:
                            currentPos[i] += -1 * trades_stack[i][q][1]
                            trades_stack[i].pop(q)
                            continue
                    else:
                        if store[len(store) - 1] >= moving_averages[i][trades_stack[i][q][2]] and moving_averages[i][trades_stack[i][q][2]] > trades_stack[i][q][0]:
                            currentPos[i] += -1 * trades_stack[i][q][1]
                            trades_stack[i].pop(q)
                            continue
                    q+= 1
            change = ((store[len(store) - 1] - store[len(store) - 2]) / store[len(store) - 2]) * 100
            if (change >= 1.5 * model_data[i]['dv'] + model_data[i]['q3']) or (change <= 1.5 * model_data[i]['dv'] - model_data[i]['q1']):
                    trade_val = round((-1 * ((change - model_data[i]['avg']) * 10000))/store[len(store) - 1], 0)
                    currentPos[i] += trade_val
                    trades_stack[i].append((store[len(store) - 1], trade_val, len(store) - 1))
    # Build your function body here

    return currentPos

    

import numpy as np
import pandas as pd
from lag import lag_trade
import scipy.stats as stats
from index import indexing
interval = 300
df = pd.read_csv('prices.txt', delim_whitespace=True, header = None)
trades_stack = []
correl_stack = []
regression_gradients = []
for i in range(len(df.columns)):
    trades_stack.append([])
def calculate_regressions(additional = None):
    global regression_gradients
    if additional != None:
        pd.concat([df, pd.DataFrame(additional)], axis=0, ignore_index=True)
        regression_gradients = []
    for i in range(len(df.columns)):
        store_a = df.iloc[:, i].values
        for j in range(len(df.columns)):
            store_b = df.iloc[:, j].values
            if i != j:
                reg =  stats.linregress(store_a, store_b)
                if (reg.slope >= 0.96 or reg.slope <= -0.96):
                    regression_gradients.append((stats.linregress(store_a, store_b), i, j))
calculate_regressions()

nInst=100
currentPos = np.zeros(nInst)

def getMyPosition (prcSoFar):
    global currentPos
    global regression_gradients
    global trades_stack
    global correl_stack
    global interval
    marked = []
    for i in range(len(trades_stack)):
        if len(trades_stack[i]) > 0:
            for trade in trades_stack[i]:
                if prcSoFar.shape[1] == trade[1] + 1 and trade[2] == 'lag':
                    currentPos[i] += trade[0] * -1
                    marked.append(trade)
            for trade in marked:
                trades_stack[i].remove(trade)
            marked = []
    '''
    for trade in correl_stack:
        # check whether there are trades open for this instrument
        for regression in regression_gradients:
            if trade[2] == regression[1] and trade[3] == regression[2]:
                i = regression[1]
                j = regression[2]
                max_slope = regression[0].slope + 0.5 * regression[0].stderr
                min_slope = regression[0].slope - 0.5 * regression[0].stderr
                max_intercept = regression[0].intercept + 0.5 * regression[0].intercept_stderr
                min_intercept = regression[0].intercept - 0.5 * regression[0].intercept_stderr
                slopes = [max_slope, min_slope]
                intercepts = [max_intercept, min_intercept]
                if prcSoFar[j,:][-1] <= max(slopes) * prcSoFar[i,:][-1] + max(intercepts) and prcSoFar[j,:][-1] >=  min(slopes) * prcSoFar[i,:][-1] + min(intercepts):
                    currentPos[i] += trade[0] * -1
                    currentPos[j] += trade[1] * -1
                    marked.append(trade)
    '''
    for trade in marked:
        correl_stack.remove(trade)
    currentPos, trades_stack = indexing(prcSoFar, currentPos, trades_stack)
    currentPos, trades_stack = lag_trade(prcSoFar, currentPos, trades_stack)
    #return currentPos
    if interval == prcSoFar.shape[1]:
        calculate_regressions()
        interval += 100
    
    for regress in regression_gradients:
        scalar = regress[0].slope
        i = regress[1]
        j = regress[2]
        store_a = prcSoFar[i,:]
        store_b = prcSoFar[j,:]
        if len(store_a) > 1:
            if store_b[len(store_b) - 1] - store_b[len(store_b) - 2] / store_a[len(store_a) - 1]- store_a[len(store_a) - 2] > 1.15 * scalar:
                # volume test 2
                trade_b = -10 * (store_b[len(store_b) - 1] - store_b[len(store_b) - 2] / store_a[len(store_a) - 1] - store_a[len(store_a) - 2]) / (1.2 * scalar)
                trade_a = (trade_b * -1) / scalar
                trade_b = round(trade_b/store_b[len(store_b) - 1], 0)
                trade_a = round(trade_a/store_a[len(store_a) - 1], 0)
                correl_stack.append((trade_a, trade_b, i, j))
            elif store_b[len(store_b) - 1] - store_b[len(store_b) - 2] / store_a[len(store_a) - 1] - store_a[len(store_a) - 2] < 0.85 * scalar:
                # volume test 2
                trade_b = 10 * (store_b[len(store_b) - 1] - store_b[len(store_b) - 2] / store_a[len(store_a) - 1] - store_a[len(store_a) - 2]) / (1.2 * scalar)
                trade_a = (trade_b * -1) / scalar
                trade_b = round(trade_b/store_b[len(store_b) - 1], 0)
                trade_a = round(trade_a/store_a[len(store_a) - 1], 0)
                correl_stack.append((trade_a, trade_b, i, j))
        
            currentPos[i] += trade_a
            currentPos[j] += trade_b

        
    # Build your function body here

    return currentPos

    

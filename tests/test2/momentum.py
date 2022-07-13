import pandas as pd
import numpy as np
import math
df2 = pd.read_csv('prices.txt', delim_whitespace=True, header = None)


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
            



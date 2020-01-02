import pandas as pd
import ta
import sklearn.cluster as cluster
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("GOOG.csv", sep=',')

df = ta.add_all_ta_features(df, open="Open", high="High",
                            low="Low", close="Close", volume="Volume")

'''
    
    TO TRY:
        -CLUSTER ACROSS THE ENTIRE ROW (? NUMBER OF CLUSTERS)
        -CLUSTER BY EACH COLUMN (? NUMBER OF CLUSTERS)
        -https://openreview.net/forum?id=Bkl7bREtDr (Aggregated Memory)

'''

X = df[df.columns[7:]].values

n_clusters = 5
clustered = cluster.KMeans(n_clusters=n_clusters).fit(X)

predictions = []
rewards = []
for i in range(len(X)-1):
    predict = clustered.predict([X[i]])[0]
    predictions.append(predict)

max_count = max(set(predictions), key=predictions.count)
for i in range(n_clusters):
    print("Need {} more for {}".format(30 - predictions.count(i), i))

plt.title("Number of Observations per Cluster")
plt.hist(predictions)
plt.show()

'''

    START OF CLUSTERED Q LEARNING
        TODO
            - argmax of Q : values * confidence level (true positive on correct action)
                - Buy : value * level
                - Don't : value * (1 - level)

'''

n_actions = 2
n_states = n_clusters
n_episodes = 10
alpha = .01
epsilon = 1.0 / (n_actions * 1.0)

Q = np.zeros([n_states, n_actions])


def reward_structure(j, action):
    '''
        PERCENTAGE CHANGE BETWEEN NEXT DAY OPEN AND THE 5 FOLLOWING DAY HIGH'S
        ->DUE TO RL MODEL LOOKING AT CURRENT DAY AND THE ACTION HAPPENING THE FOLLOWING
        ->OPEN.

    :param j: INDEX OF STOCK
    :param action: TO BUY OR NOT TO BUY
    :return: PERCENTAGE BETWEEN MAX AND NEXT OPENING PRICE
    '''

    reward = max([x[1] for x in X[:6+j]]) / X[j+1][0]

    if action == 0:  # Buy and then sell on next 5 day high
        if reward > 1.01:
            return .01
        else:
            return -.05
    else:  # Don't Buy
        if reward > 1.01:
            return (-1.0) * reward
        else:
            return .05


for i in range(n_episodes):
    for j in range(len(X)-6):
        state = clustered.predict([X[j]])

        action = np.argmax(Q[state])
        if np.random.random() < epsilon:
            action = np.random.randint(n_actions)

        reward = reward_structure(j, action)

        Q[state, action] += alpha * reward

    print(Q)

test = [1000.00]
for i in range(len(X)-6):
    state = clustered.predict([X[i]])

    action = np.argmax(Q[state])

    if action == 0:
        sell_trigger = max([x[1] for x in X[:6+i]]) / X[i+1][0]

        if sell_trigger < 0:
            test.append(test[-1] * .95 + 50.00)
        else:
            test.append(test[-1] * 1.01 + 50.00)
    else:
        test.append(test[-1] + 50.0)

print("Percentage Increased: {}".format(test[-1] / test[0]))

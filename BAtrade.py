import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


ticker = ['AAPL', 'AVB', 'AMD', 'CCC', 'BA', 'BLL', 'C', 'INTC', 'GPS', 'MMM', 'T', 'TJX']
STK = len(ticker)

print('Reading data... \n')
price = pd.read_csv('price.csv').values
T = price.shape[0] - 1
R = np.ones((T, STK))  # data of returns
for idx in range(STK):
    R[:, idx] = (price[1:, idx] - price[:T, idx])/price[:T, idx]
R = R - 0.001  # excess return
prob = np.zeros(T)  # probability that model t is true
prob[:2] = 0.5
mu = np.ones((T, STK))/STK  # model t's mean
oldmu = np.ones(STK)/STK
cov = np.ones((T, STK, STK))  # model t's covariance
oldcov = np.zeros((T, STK, STK))
for i in range(T):
    cov[i, :, :] = np.eye(STK)
kappa = np.ones(T)
delta = np.ones(T)
L = np.zeros(T)
phi = np.zeros((T, STK))

print('Caculating the BA strategy... \n')
for t in range(3, T):
    # new model t is born
    mu[t, :] = np.sum(R[:t, :]) / (t - 1) / STK
    cov[t, :, :] = np.sum(np.power(R[:t, :] - np.sum(R[:t, :], axis=0), 2)) / (t - 2) / STK * np.eye(STK)
    # probability updated
    oldprob = np.copy(prob)
    for q in range(t):
        oldprob[q] = oldprob[q] / (t + 1 - q)
    prob[:t] = np.cumsum(oldprob[:t])
    prob[t] = prob[t - 1]

    # prob[:t+1] = 1 / (t + 1)

    # update mean and cov when new return on time t observed
    for idx in range(t + 1):
        oldmu = np.copy(mu[idx, :])
        mu[idx, :] = (kappa[idx] * oldmu + R[t, :]) / (kappa[idx] + 1)
        oldcov[idx, :, :] = np.copy(cov[idx, :, :])
        cov[idx, :, :] = (delta[idx] * (kappa[idx] + 1) * cov[idx, :, :] +
                          kappa[idx] * np.outer(R[t, :] - oldmu, R[t, :] - oldmu)) / (delta[idx] + 1) / (kappa[idx] + 1)
        delta[idx] = delta[idx] + 1
        kappa[idx] = kappa[idx] + 1

    # update probability when new return on t observed
    for idx in range(t + 1):
        detold = np.power(
             np.linalg.det((delta[idx] - 1) * oldcov[idx, :, :]) / np.linalg.det(delta[idx] * cov[idx, :, :]),
             (delta[idx] + STK) / 2)
        detnew = np.power(np.linalg.det(delta[idx] * cov[idx, :, :]), 1 / 2)
        # gammafrac = spy.gamma((delta[idx] + STK + 1) / 2) / spy.gamma((delta[idx] + STK) / 2 + (1-STK) / 2)
        kappafrac = np.power(1 - 1 / kappa[idx], STK / 2)
        L[idx] = detold / detnew * (delta[idx] + 10 + 1) / 2 * (delta[idx] + 8 + 1) / 2 * (delta[idx] + 6 + 1) / 2 * (
                 delta[idx] + 4 + 1) / 2 * (delta[idx] + 2 + 1) / 2 * (delta[idx] + 1) / 2 * kappafrac

    totalprob = np.dot(L, prob)
    prob = np.multiply(L, prob) / totalprob

    # calculate strategy
    MU = np.sum(np.multiply(mu[:t + 1, :], np.tile(prob[:t + 1], (STK, 1)).transpose()), axis=0)
    SIGMA = np.zeros((STK, STK))
    for idx in range(t + 1):
        SIGMA = SIGMA + (1 + kappa[idx]) / kappa[idx] * cov[idx, :, :] + np.outer(mu[idx, :], mu[idx, :]) * prob[idx]
    SIGMA = SIGMA - np.outer(MU, MU)
    if np.isnan(np.linalg.cond(SIGMA, 2)):
        print('Condition number for time {0} is NaN\n'.format(t))
    phi[t, :] = np.dot(np.linalg.inv(SIGMA), MU)

# backtest of the strategy, starting from Week 701 to Week 800

print('Backtesting... \n')
start = 700
ter = 800
profit = np.ones(ter-start)
weekrtn = np.zeros(ter-start-1)
for t in range(start, ter-1):
    profit[t-start+1] = profit[t-start] * (R[t+1, :].dot(1000*phi[t, :]) + 1.001)
    weekrtn[t-start] = R[t+1, :].dot(1000*phi[t, :])
print('Sharpe ratio of BA is %f\n' % (np.mean(weekrtn)/np.std(weekrtn)*np.sqrt(52)))
print('Annualized return is %f\n' % (np.power(profit[ter-start-1], 52/(ter-start))-1))
# rebalanced strategy
RBprofit = np.ones(ter-start)
RBweekrtn = np.zeros(ter-start+1)
for t in range(start, ter-1):
    RBprofit[t-start+1] = RBprofit[t-start] * (R[t+1, :].dot(np.ones(12))/13 + 1.001)
    RBweekrtn[t-start] = R[t+1, :].dot(np.ones(12)/13)
print('Sharpe ratio of Rebalanced strategy is %f\n' % (np.mean(RBweekrtn)/np.std(RBweekrtn)*np.sqrt(52)))
print('Annualized return is %f\n' % (np.power(RBprofit[ter-start-1], 52/(ter-start))-1))
# best stock
print('Final wealth of holding each stock is {0} \n'.format(price[ter, :]/price[start+1, :]))

# plot the comparison figure
plt.plot(profit, label='BA')
plt.plot(RBprofit, label='RB')
plt.plot(price[start+1:ter+1, 0]/price[start+1, 0], label='AAPL')
plt.plot(price[start+1:ter+1, 5]/price[start+1, 5], label='BLL')
plt.legend(loc='best')
plt.title('Comparison')
plt.savefig('compare.png', dpi=300)
#  plt.show()

# save figure showing the money invested in stock
plt.figure(2, figsize=(12, 9), dpi=80)
for idx in range(12):
    ax = plt.subplot(4, 3, idx+1)
    ax.set_title(ticker[idx])
    ax.plot(phi[start:ter, idx]*1000)
plt.tight_layout()
plt.savefig('strategy.png', dpi=300)

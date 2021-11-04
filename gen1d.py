import numpy as np
from scipy.stats import multivariate_normal,norm
import matplotlib.pyplot as plt

N = 300 # Number of data
K = 3

# mean parameter of synthetic data
mu_truth_kd_1 = np.array([[0], [20], [-20]])
mu_truth_kd_2 = np.array([[0], [20], [40]])

# variance parameter of synthetic data
sigma2_truth_kdd_1 = np.array([[5,0], [5,0], [5,0]])
sigma2_truth_kdd_2 = np.array([[6,0], [6,0], [6,0]])

start = mu_truth_kd_1[0][0] - 10 * 5
end = mu_truth_kd_1[0][0] + 10 * 7
X = np.arange(start, end, 0.1)

Y1_1 = norm.pdf(X, loc=mu_truth_kd_1[0][0], scale=10)
Y2_1 = norm.pdf(X, loc=mu_truth_kd_1[1][0], scale=10)
Y3_1 = norm.pdf(X, loc=mu_truth_kd_1[2][0], scale=10)
Y1_2 = norm.pdf(X, loc=mu_truth_kd_2[0][0], scale=10)
Y2_2 = norm.pdf(X, loc=mu_truth_kd_2[1][0], scale=10)
Y3_2 = norm.pdf(X, loc=mu_truth_kd_2[2][0], scale=10)

plt.figure()
plt.grid()
plt.plot(X, Y1_1, color='Blue',label="K=1")
plt.plot(X, Y2_1, color='Yellow',label="K=2")
plt.plot(X, Y3_1, color='Green',label="K=3")
plt.legend()
#plt.show()
plt.close()

label_0 = np.full(100, 0)
label_1 = np.full(100, 1)
label_2 = np.full(100, 2)
s_truth_n = np.concatenate([label_0,label_1,label_2])
label = s_truth_n

# make synthetic data
x_nd_1 = np.array([
    np.random.normal(
        loc=mu_truth_kd_1[k], scale=sigma2_truth_kdd_1[k][0], size=1
    ).flatten() for k in s_truth_n
])
x_nd_2 = np.array([
    np.random.normal(
        loc=mu_truth_kd_2[k], scale=sigma2_truth_kdd_2[k][0], size=1
    ).flatten() for k in s_truth_n
])

# plot observation
plt.figure()
plt.grid()
for k in reversed(range(K)):
    k_idx, = np.where(s_truth_n == k)
    if k_idx[0] == 0:
        color='Red'
        marker=","
    elif k_idx[0] == 100: 
        color='Blue'
        marker="o"
    elif k_idx[0] == 200:
        color='Green'
        marker="v"
    plt.scatter(x_nd_1[k_idx],np.zeros((100,1)),marker=marker,s=40,color=color)
plt.plot(X, Y1_1, color='Blue',label="K=1")
plt.plot(X, Y2_1, color='Orange',label="K=2")
plt.plot(X, Y3_1, color='Green',label="K=3")
plt.legend()
plt.savefig('./dataset/data1d_1.png')
plt.show()
plt.close()

plt.figure()
plt.grid()
for k in reversed(range(K)):
    k_idx, = np.where(s_truth_n == k)
    if k_idx[0] == 0:
        color='Red'
        marker=","
    elif k_idx[0] == 100: 
        color='Blue'
        marker="o"
    elif k_idx[0] == 200:
        color='Green'
        marker="v"
    plt.scatter(x_nd_2[k_idx],np.zeros((100,1)),marker=marker,s=40,color=color)
plt.plot(X, Y1_2, color='Blue',label="K=1")
plt.plot(X, Y2_2, color='Orange',label="K=2")
plt.plot(X, Y3_2, color='Green',label="K=3")
plt.legend()
plt.savefig('./dataset/data1d_2.png')
plt.show()
plt.close()

############################## Save synthetic data as .npy files ##############################
np.save('./dataset/data1d_1.npy', x_nd_1) # Corresponds to x_1 in the graphical model
np.save('./dataset/data1d_2.npy', x_nd_2) # Corresponds to x_2 in the graphical model
np.save('./dataset/true_label1d.npy', s_truth_n) # # True label (True z_n)

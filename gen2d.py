import numpy as np
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt

K = 3 # Number of clusters
N = 300 # Number of data
pi_synthetic_k = np.array([0.25, 0.25, 0.25]) # Mixing ratio for synthetic data


mu_synthetic_kd_1 = np.array( # This is observation1
    [[0.0, 0.0], 
     [13.0, 10.0], 
     [-10.0, 18.0]]
)
sigma2_synthetic_kdd_1 = np.array( # This is observation1
    [[[10.0, 0.0], 
    [0.0, 10.0]],

     [[10.0, 0.0], 
     [0.0, 10.0]],

     [[10.0, 0.0], 
     [0.0, 10.0]]]
)
mu_synthetic_kd_2 = np.array( # This is observation1
    [[0, 0.0], 
     [15.0, 0.0], 
     [0.0, 15.0]]
)
sigma2_synthetic_kdd_2 = np.array( # This is observation1
    [[[10.0, 0.0], 
    [0.0, 10.0]],

     [[10.0, 0.0], 
     [0.0, 10.0]],

     [[10.0, 0.0], 
     [0.0, 10.0]]]
)

############################## Make synthetic data ##############################
label_0 = np.full(100, 0)
label_1 = np.full(100, 1)
label_2 = np.full(100, 2)
z_synthetic_n = np.concatenate([label_0,label_1,label_2])
#z_synthetic_nk = np.random.multinomial(n=1, pvals=pi_synthetic_k, size=N)
#_, z_synthetic_n = np.where(z_synthetic_nk == 1)
# x_nd_1 will be the observation1 of Multimodal-GMM
x_nd_1 = np.array([
    np.random.multivariate_normal(
        mean=mu_synthetic_kd_1[k], cov=sigma2_synthetic_kdd_1[k], size=1
    ).flatten() for k in z_synthetic_n
])
# x_nd_2 will be the observation2 of Multimodal-GMM
x_nd_2 = np.array([
    np.random.multivariate_normal(
        mean=mu_synthetic_kd_2[k], cov=sigma2_synthetic_kdd_2[k], size=1
    ).flatten() for k in z_synthetic_n
])

############################## Setting for graph ##############################
x_1_line_1 = np.linspace(
    np.min(mu_synthetic_kd_1[:, 0] - 3 * np.sqrt(sigma2_synthetic_kdd_1[:, 0, 0])), 
    np.max(mu_synthetic_kd_1[:, 0] + 3 * np.sqrt(sigma2_synthetic_kdd_1[:, 0, 0])), 
    num=300
)
x_2_line_1 = np.linspace(
    np.min(mu_synthetic_kd_1[:, 1] - 3 * np.sqrt(sigma2_synthetic_kdd_1[:, 1, 1])), 
    np.max(mu_synthetic_kd_1[:, 1] + 3 * np.sqrt(sigma2_synthetic_kdd_1[:, 1, 1])), 
    num=300
)
x_1_grid_1, x_2_grid_1 = np.meshgrid(x_1_line_1, x_2_line_1)
x_point_1 = np.stack([x_1_grid_1.flatten(), x_2_grid_1.flatten()], axis=1)
x_dim_1 = x_1_grid_1.shape

x_1_line_2 = np.linspace(
    np.min(mu_synthetic_kd_2[:, 0] - 3 * np.sqrt(sigma2_synthetic_kdd_2[:, 0, 0])), 
    np.max(mu_synthetic_kd_2[:, 0] + 3 * np.sqrt(sigma2_synthetic_kdd_2[:, 0, 0])), 
    num=300
)
x_2_line_2 = np.linspace(
    np.min(mu_synthetic_kd_2[:, 1] - 3 * np.sqrt(sigma2_synthetic_kdd_2[:, 1, 1])), 
    np.max(mu_synthetic_kd_2[:, 1] + 3 * np.sqrt(sigma2_synthetic_kdd_2[:, 1, 1])), 
    num=300
)
x_1_grid_2, x_2_grid_2 = np.meshgrid(x_1_line_2, x_2_line_2)
x_point_2 = np.stack([x_1_grid_2.flatten(), x_2_grid_2.flatten()], axis=1)
x_dim_2 = x_1_grid_2.shape

# Ovservation model
true_model_1 = 0
for k in range(K):
    tmp_density_1 = multivariate_normal.pdf(x=x_point_1, mean=mu_synthetic_kd_1[k], cov=sigma2_synthetic_kdd_1[k])    
    true_model_1 += pi_synthetic_k[k] * tmp_density_1

true_model_2 = 0
for k in range(K):
    tmp_density_2 = multivariate_normal.pdf(x=x_point_2, mean=mu_synthetic_kd_2[k], cov=sigma2_synthetic_kdd_2[k])    
    true_model_2 += pi_synthetic_k[k] * tmp_density_2

# plot x_nd_1 graph
plt.figure(figsize=(12, 9))
for k in range(K):
    k_idx, = np.where(z_synthetic_n == k)
    plt.scatter(x=x_nd_1[k_idx, 0], y=x_nd_1[k_idx, 1], label='cluster:' + str(k + 1)) 
plt.contour(x_1_grid_1, x_2_grid_1, true_model_1.reshape(x_dim_1), linestyles='--')
plt.suptitle('Synthetic data for Multimodal-GMM:Observation1', fontsize=20)
plt.title('Number of data:' + str(N) + ', Number of clusters:' + str(K), loc='left')
plt.xlabel('$x_1$'); plt.ylabel('$x_2$')
plt.colorbar()
plt.savefig("./dataset/data2d_1.png")
plt.show()
plt.close()

# plot x_nd_2 graph
plt.figure(figsize=(12, 9))
for k in range(K):
    k_idx, = np.where(z_synthetic_n == k)
    plt.scatter(x=x_nd_2[k_idx, 0], y=x_nd_2[k_idx, 1], label='cluster:' + str(k + 1)) 
plt.contour(x_1_grid_2, x_2_grid_2, true_model_2.reshape(x_dim_2), linestyles='--')
plt.suptitle('Synthetic data for Multimodal-GMM:Observation2', fontsize=20)
plt.title('Number of data:' + str(N) + ', Number of clusters:' + str(K), loc='left')
plt.xlabel('$x_1$'); plt.ylabel('$x_2$')
plt.colorbar()
plt.savefig("./dataset/data2d_2.png")
plt.show()
plt.close()

############################## Save synthetic data as .txt files ##############################
np.save('./dataset/data2d_1.npy', x_nd_1) # Corresponds to x_1 in the graphical model
np.save('./dataset/data2d_2.npy', x_nd_2) # Corresponds to x_2 in the graphical model
np.save('./dataset/true_label2d.npy', z_synthetic_n) # # True label (True z_n)
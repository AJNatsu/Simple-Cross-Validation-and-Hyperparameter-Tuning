import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

# Data Preparation
ndata = 50
xa = np.linspace(0, 1, 100)
yt = np.sin(2 * np.pi * xa)

x = np.random.choice(xa, ndata, replace=False)  # Number of data points is ndata
tt = np.sin(2 * np.pi * x)
err = np.random.normal(0, 0.1, ndata)  # Number of data points is ndata
t = tt + err  # Adding noise to the sine function

plt.plot(xa, yt)  # Plot the original sine function
plt.plot(x, t, 'o')  # Plot the data points
plt.show()

# Data Splitting
x_train, x_test, t_train, t_test = train_test_split(x, t, test_size=0.15, random_state=42)

print(x_train.shape)
print(x_test.shape)
print(t_train.shape)
print(t_test.shape)



# Cross-Validation and Hyperparameter Tuning
def design_matrix(x, m):
    pf = PolynomialFeatures(degree=m)
    dmat = pf.fit_transform(x.reshape(-1, 1))
    return dmat

n_alphas = 100
alphas = np.logspace(-10, -6, n_alphas)
M = 9  # Polynomial degree
nsp = 10  # Number of splits for CV
rmse_train = []
rmse_test = []

# Loop for alpha
for a in alphas:
    kf = KFold(n_splits=nsp)
    krmse_train = []
    krmse_test = []

    # k-loop
    for ktrain_indx, ktest_indx in kf.split(x_train):
        x_ktrain, x_ktest = x_train[ktrain_indx], x_train[ktest_indx]
        t_ktrain, t_ktest = t_train[ktrain_indx], t_train[ktest_indx]
        phi = design_matrix(x_ktrain, M)
        phi = phi[:, 1:]  # Remove the constant term
        ss = StandardScaler()
        phi_std = ss.fit_transform(phi)
        phi_test = design_matrix(x_ktest, M)
        phi_test = phi_test[:, 1:]  # Remove the constant term
        phi_test_std = ss.transform(phi_test)
        ridge = Ridge(alpha=a, fit_intercept=True)
        ridge.fit(phi_std, t_ktrain)
        pre_train = ridge.predict(phi_std)
        pre_test = ridge.predict(phi_test_std)
        krmse_train.append(mean_squared_error(t_ktrain, pre_train, squared=False))
        krmse_test.append(mean_squared_error(t_ktest, pre_test, squared=False))

    rmse_train.append(np.mean(krmse_train))
    rmse_test.append(np.mean(krmse_test))

plt.plot(alphas, rmse_train, label='Train')
plt.plot(alphas, rmse_test, label='Test')
plt.xscale('log')
plt.legend()
plt.xlabel('alpha')
plt.ylabel('RMSE')
plt.title('CV error')
plt.show()

a_opt = alphas[np.argmin(rmse_test)]

# Training with Optimized Hyperparameter
phi = design_matrix(x_train, M)
phi = phi[:, 1:]  # Remove the constant term
phi_test = design_matrix(x_test, M)
phi_test = phi_test[:, 1:]  # Remove the constant term
ss = StandardScaler()
phi_std = ss.fit_transform(phi)
phi_test_std = ss.transform(phi_test)
ridge = Ridge(alpha=a_opt, fit_intercept=True)
ridge.fit(phi_std, t_train)
y_train = ridge.predict(phi_std)
y_test = ridge.predict(phi_test_std)

# Model Evaluation and Plotting
phi_xa = design_matrix(xa, M)
phi_xa = phi_xa[:, 1:]  # Remove the constant term
phi_xa_std = ss.transform(phi_xa)
ya = ridge.predict(phi_xa_std)

plt.plot(xa, yt)
plt.plot(x_train, t_train, 'o')
plt.plot(x_test, t_test, 'v')
plt.plot(xa, ya)
plt.ylim(-1.5, 1.5)
plt.show()

plt.plot(t_train, y_train, 'o', label="training")
plt.plot(t_test, y_test, 'v', label="test")
plt.plot([-1.5, 1.5], [-1.5, 1.5])
plt.ylim(-1.5, 1.5)
plt.xlabel('True')
plt.ylabel('Predicted')
plt.legend()
plt.title('Ridge')
plt.show()

print('training: RMSE =', mean_squared_error(t_train, y_train, squared=False))
print('test: RMSE =', mean_squared_error(t_test, y_test, squared=False))

# -*- coding: utf-8 -*-
"""
Created on Tue Feb  3 16:47:11 2026

@author: jafish
"""

import numpy as np
from scipy.integrate import solve_ivp
from reservoir_computer import reservoir_computer
from sklearn.preprocessing import StandardScaler
 

def lorenz63(t, state, sigma=10.0, beta=8/3, rho=28.0):
    x, y, z = state
    dxdt = sigma * (y - x)
    dydt = x * (rho - z) - y
    dzdt = x * y - beta * z
    return [dxdt, dydt, dzdt]

# Simulation parameters
t0 = 0.0
t_end = 50.0
dt = 0.01
t_eval = np.arange(t0, t_end, dt)

# Initial condition
x0 = [1.0, 1.0, 1.0]

# Integrate using RK45
sol = solve_ivp(
    lorenz63,
    t_span=(t0, t_end),
    y0=x0,
    t_eval=t_eval,
    method='RK45',
    rtol=1e-6,
    atol=1e-6
)

# Extract trajectory
X = sol.y.T

x0 = X[-1,:]
dt = 0.01
test_length = 50000
train_length = 20000
t_end = int((test_length+train_length)*dt)
t_eval = np.arange(0, t_end, dt)
sol = solve_ivp(
    lorenz63,
    t_span=(t0, t_end),
    y0=x0,
    t_eval=t_eval,
    method='RK45',
    rtol=1e-6,
    atol=1e-6
)

# Redo to remove transient
X = sol.y.T
t = sol.t

#scale the data for training
standard_scaler = StandardScaler()
X = standard_scaler.fit_transform(X)
#split into test and train
X_test = X[0:test_length-1,:]
Y_test = X[1:test_length,:]

prediction_length = 2000

res = reservoir_computer(X_test,Y_test,reservoir_type='watts_strogatz',ws_p=0.75)
res.train_reservoir()
pred = res.predict(X[test_length,:],prediction_length)
vpt = res.valid_prediction_time(X[test_length:test_length+prediction_length,:],pred)
print(vpt)
hd = res.henrici_departure()
print(hd)
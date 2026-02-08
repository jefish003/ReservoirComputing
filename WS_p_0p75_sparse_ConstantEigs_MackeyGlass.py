#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  8 17:19:35 2026

@author: ozge
"""

import numpy as np
from scipy.integrate import solve_ivp
from reservoir_computer import reservoir_computer
from sklearn.preprocessing import StandardScaler
from datetime import datetime
from matplotlib import pyplot as plt
 

def func_mackeyglass(x, x_tau, t):
    beta = 0.2
    gamma = 0.1
    power = 10
    return beta * x_tau / (1 + x_tau**power) - gamma * x


def rk4_delay(f, x0, t, tau):
    h = t[1] - t[0]
    n = len(t)
    x = np.zeros(n)

    # delay in integer steps (exact for dt = 0.01 and tau = 30)
    tau_steps = round(tau / h)

    # fill history
    x[:tau_steps] = x0

    for i in range(tau_steps, n - 1):

        # delayed indices for RK4 stages
        i_tau_1 = i - tau_steps
        i_tau_2 = i - tau_steps + round(0.5)
        i_tau_3 = i - tau_steps + round(0.5)
        i_tau_4 = i - tau_steps + 1

        k1 = f(x[i],               x[i_tau_1], t[i])
        k2 = f(x[i] + 0.5*h*k1,    x[i_tau_2], t[i] + 0.5*h)
        k3 = f(x[i] + 0.5*h*k2,    x[i_tau_3], t[i] + 0.5*h)
        k4 = f(x[i] + h*k3,        x[i_tau_4], t[i] + h)

        x[i+1] = x[i] + (h/6)*(k1 + 2*k2 + 2*k3 + k4)

    return x


#NOTE parameters below come from the optimized Zheng-Meng et. al. paper and github page
#found at https://github.com/Zheng-Meng/Reservoir-Computing-and-Hyperparameter-Optimization/tree/main
date_str = datetime.now().strftime("%Y-%m-%d")
reservoir_type = 'watts_strogatz'
non_normal_type = 'sparse'
dynamics_type='mackey_glass'
spectral_radius = 1.3200901
gamma = 0.1933000
alpha = 0.5577053
res_density = 0.6008646
in_noise = 0.0067821
maintain_res_eigs=True
npz = np.load('/Users/ozge/Downloads/ReservoirComputing-main/Mackey_Glass_Win.npz')
Win = npz['arr_0']
ws_p=0.75
npz = np.load("WS_p_0p75_Adjacency.npz")
A = npz['arr_0']

#Trials params
trials = 100
NUMaVals = 51
aVals = np.linspace(0.04,2.04,NUMaVals)
ValidTimeVals = np.zeros((trials,NUMaVals))
henrici_coeffs = np.zeros((trials,NUMaVals))
for j in range(NUMaVals):
    for i in range(trials):
        #get the henrici parameter a
        a = aVals[j]
        # Simulation parameters
        t0 = 0.0
        t_end = 200.0
        dt = 0.01
        t_eval = np.arange(t0, t_end, dt)
        
        # Initial condition
        x0 = 1.0
        
        # Integrate using RK45
        X = rk4_delay(func_mackeyglass,x0=x0,t = t_eval,tau=30)
        X = X.reshape(-1,1)
        
        
        x0 = X[-1,:]
        dt = 0.01
        test_length = 50000
        train_length = 20000
        burnin = 10000
        t_end = int((test_length+train_length+burnin)*dt)
        t_eval = np.arange(0, t_end, dt)
        #Integrate after transient
        X = rk4_delay(func_mackeyglass,x0=x0,t = t_eval,tau=30)
       
        # Redo to remove transient
        X = X.reshape(-1,1)
        X = X[burnin:,:]
        
        #scale the data for training
        standard_scaler = StandardScaler()
        X = standard_scaler.fit_transform(X)
        #split into test and train
        X_test = X[0:test_length-1,:]
        Y_test = X[1:test_length,:]
        
        prediction_length = 2000
        
        res = reservoir_computer(X_test,Y_test,
                                 A = A,
                                 reservoir_type=reservoir_type,
                                 a=a,
                                 non_normal_type=non_normal_type,
                                 gamma = gamma,
                                 spectral_radius=spectral_radius,
                                 alpha = alpha,
                                 res_density = res_density,
                                 in_noise = in_noise,
                                 Win=Win,
                                 ws_p=ws_p,
                                 maintain_res_eigs=maintain_res_eigs
                                 )
        res.train_reservoir()
        pred = res.predict(X[test_length,:],prediction_length)
        vpt = res.valid_prediction_time(X[test_length:test_length+prediction_length,:],pred)
        ValidTimeVals[i,j] = vpt
        print(vpt)
        hd = res.henrici_departure()
        henrici_coeffs[i,j] = hd
        print(hd)

D = {}
D['reservoir_type'] = reservoir_type
D['aVals'] = aVals
D['ValidPredictionTimes'] = ValidTimeVals
D['HenriciDepartures'] = henrici_coeffs
D['non_normal_type'] = non_normal_type
D['dynamics_type'] = dynamics_type

np.savez("Mackey_Glass_ConstantEigs_reservoirTest_"+date_str+"_"+reservoir_type+"_non_normal_type_"+non_normal_type+"ws_p_"+str(ws_p)+"_.npz",D)
plt.plot(np.mean(henrici_coeffs,axis=0),np.mean(ValidTimeVals,axis=0),'rx')
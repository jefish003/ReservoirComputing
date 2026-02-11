#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 11 18:35:22 2026

@author: ozge
"""

import numpy as np
from scipy.integrate import solve_ivp
from reservoir_computer import reservoir_computer
from sklearn.preprocessing import StandardScaler
from datetime import datetime
from matplotlib import pyplot as plt
 

def lorenz63(t, state, sigma=10.0, beta=8/3, rho=28.0):
    x, y, z = state
    dxdt = sigma * (y - x)
    dydt = x * (rho - z) - y
    dzdt = x * y - beta * z
    return [dxdt, dydt, dzdt]


date_str = datetime.now().strftime("%Y-%m-%d")
reservoir_type = 'erdos_renyi'
non_normal_type = 'dense'
dynamics_type='lorenz_63'
npz = np.load("ER_MackeyGlass_Adjacency.npz")
A = npz['arr_0']
npz = np.load('Constant_Win.npz')
Win = npz['arr_0']
spectral_radius = 2
maintain_res_eigs = True
ws_p = 0
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
        
        res = reservoir_computer(X_test,Y_test,
                                 reservoir_type=reservoir_type,
                                 A=A,
                                 a=a,
                                 non_normal_type=non_normal_type,
                                 Win=Win,
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
np.savez(dynamics_type+"_ConstantEigs_Changing_spectral_radius_sr_"+str(spectral_radius)+"_reservoirTest_"+date_str+"_"+reservoir_type+"_non_normal_type_"+non_normal_type+"ws_p_"+str(ws_p)+"_.npz",D)
plt.plot(np.mean(henrici_coeffs,axis=0),np.mean(ValidTimeVals,axis=0),'rx')
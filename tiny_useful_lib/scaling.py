#import tensorflow as tf
import tqdm
from numpy import pi,linspace,tensordot
import scipy.optimize
import numpy as np
import copy
import matplotlib.pyplot as plt
import numdifftools as nd
from scipy.optimize import root
from scipy.interpolate import interp1d
from scipy.sparse.linalg import eigsh
from scipy.integrate import solve_ivp
from scipy import interpolate
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator
from matplotlib.ticker import FuncFormatter
from scipy.optimize import minimize
from scipy.optimize import dual_annealing
from scipy.linalg import cosm, expm, sqrtm, det
from QuantumCircuits.tiny_useful_lib.main import *


# fun for searching of proper fluxonium 

def fluxonium_coup(f01, alpha, bounds=[(2, 100), (0.4, 1.5), (0.2, 6)]):

    def fun(x):
        eigval, _, q = Fluxonium(x[0], x[1], x[2], gridSize=60, numOfLvls=3, F=0)
        return (eigval[1] - f01)**2 + (eigval[2] - 2*eigval[1] - alpha)**2 - 0.001*abs(q[0, 1])**2
    
    opt1 = minimize(fun, x0=[4, 1.5, 4], bounds=bounds)
    opt2 = minimize(fun, x0=[3, 1, 4], bounds=bounds)
    opt3 = minimize(fun, x0=[4, 1, 6], bounds=bounds)

    loss = np.asarray([opt1.fun, opt2.fun, opt3.fun])
    x = np.asarray([opt1.x, opt2.x, opt3.x])
    
    return x[np.argmin(loss)]

def fluxonium_search(f01, f12, f03, bounds=[(2, 100), (0.5, 1.5), (0.2, 6)], weights=[1, 1, 1]):

    
    def fun(x):
        eigval, _, _ = Fluxonium(x[0], x[1], x[2], F=0.5)
        loss=weights[0]*(eigval[1] - f01)**2 +\
             weights[1]*(eigval[2] - eigval[1] - f12)**2 +\
             weights[2]*(eigval[3] - f03)**2
        return loss
    
    opt1 = minimize(fun, x0=[4, 1.5, 4], bounds=bounds)
    opt2 = minimize(fun, x0=[3, 1, 4], bounds=bounds)
    opt3 = minimize(fun, x0=[4, 1, 6], bounds=bounds)
    loss = np.asarray([opt1.fun, opt2.fun, opt3.fun])
    x = np.asarray([opt1.x, opt2.x, opt3.x])
    
    return x[np.argmin(loss)]


# fun for searching of proper transmon

def transmon_coup(f01, alpha, bounds=[(2, 100), (0.01, 3)]):

    def fun(x):
        eigval, _, q = Transmon(x[0], 0, x[1], numOfLvls=5)
        return (eigval[1] - f01)**2 + (eigval[2] - 2*eigval[1] - alpha)**2
    
    opt1 = minimize(fun, x0=[40, 1.5], bounds=bounds)
    loss = np.asarray([opt1.fun])
    x = np.asarray([opt1.x])
    
    return x[np.argmin(loss)]


# fun for couplers zz optimization

def g_coups_opt(coup_1, coup_2, qubit, g1, g2, regime=0):

    def loss(g):

        (spect_F, phi_F, q_F) = map(np.copy, qubit)
        (spect_C1, phi_C1, q_C1) = map(np.copy, coup_1)
        (spect_C2, phi_C2, q_C2) = map(np.copy, coup_2)
        
        # емкостно смешиваем 3 подсистемы системы
        (mixEnrg, mixStates, mixH) = MixOfThreeSys(spect_C1, spect_F, spect_C2,
                                                   q12=q_C1, q13=q_C1,
                                                   q21=q_F, q23=q_F,
                                                   q32=q_C2, q31=q_C2,
                                                   g12=g1, g23=g2, g31=g, numOfLvls=150, project=True)
        
        
        key, purity = StatesPurity(mixStates, (spect_C1.shape[0], spect_F.shape[0], spect_C2.shape[0]))

        if(regime):
            return (mixEnrg[key[1, 1, 1]] - mixEnrg[key[0, 1, 1]] - mixEnrg[key[1, 1, 0]] + mixEnrg[key[0, 1, 0]])*1e6
        else:
            return (mixEnrg[key[1, 0, 1]] - mixEnrg[key[0, 0, 1]] - mixEnrg[key[1, 0, 0]] + mixEnrg[key[0, 0, 0]])*1e6
    

        
    if(regime):
        opt = minimize(loss, x0=[0], bounds=[(-0.2, 0)])
        center = opt.x[0]
        if(opt.fun > 0):
            print("can't kill zz")
            return 0, 0
        opt_r = root(loss, x0=center+1e-3)
        opt_l = root(loss, x0=center-1e-3)

    else:
        opt = minimize(loss, x0=[0], bounds=[(0, 0.2)])
        center = opt.x[0]
        if(opt.fun > 0):
            print("can't kill zz")
            return 0, 0
        opt_r = root(loss, x0=center+1e-3)
        opt_l = root(loss, x0=center-1e-3)


    g_l = opt_l.x[0]
    g_r = opt_r.x[0]

    (spect_F, phi_F, q_F) = qubit
    (spect_C1, phi_C1, q_C1) = coup_1
    (spect_C2, phi_C2, q_C2) = coup_2
    
    # емкостно смешиваем 3 подсистемы системы
    (mixEnrg_l, mixStates, mixH) = MixOfThreeSys(spect_C1, spect_F, spect_C2,
                                                    q12=q_C1, q13=q_C1,
                                                    q21=q_F, q23=q_F,
                                                    q32=q_C2, q31=q_C2,
                                                    g12=g1, g23=g2, g31=g_l, numOfLvls=150, project=True)
    
    
    key_l, purity_l = StatesPurity(mixStates, (spect_C1.shape[0], spect_F.shape[0], spect_C2.shape[0]))

    # емкостно смешиваем 3 подсистемы системы
    (mixEnrg_r, mixStates, mixH) = MixOfThreeSys(spect_C1, spect_F, spect_C2,
                                                    q12=q_C1, q13=q_C1,
                                                    q21=q_F, q23=q_F,
                                                    q32=q_C2, q31=q_C2,
                                                    g12=g1, g23=g2, g31=g_r, numOfLvls=150, project=True)
    
    
    key_r, purity_r = StatesPurity(mixStates, (spect_C1.shape[0], spect_F.shape[0], spect_C2.shape[0]))

    if(regime):

        zz_l = (mixEnrg_l[key_l[1, 1, 1]] - mixEnrg_l[key_l[0, 1, 1]] - mixEnrg_l[key_l[1, 1, 0]] + mixEnrg_l[key_l[0, 1, 0]])*1e6
        zz_r = (mixEnrg_r[key_r[1, 1, 1]] - mixEnrg_r[key_r[0, 1, 1]] - mixEnrg_r[key_r[1, 1, 0]] + mixEnrg_r[key_r[0, 1, 0]])*1e6
        
        out_l = (g_l, purity_l[key_l[1, 1, 1]], purity_l[key_l[0, 1, 1]], purity_l[key_l[1, 1, 0]], zz_l)
        out_r = (g_r, purity_r[key_r[1, 1, 1]], purity_r[key_r[0, 1, 1]], purity_r[key_r[1, 1, 0]], zz_r)

    else:

        zz_l = (mixEnrg_l[key_l[1, 0, 1]] - mixEnrg_l[key_l[0, 0, 1]] - mixEnrg_l[key_l[1, 0, 0]] + mixEnrg_l[key_l[0, 0, 0]])*1e6
        zz_r = (mixEnrg_r[key_r[1, 0, 1]] - mixEnrg_r[key_r[0, 0, 1]] - mixEnrg_r[key_r[1, 0, 0]] + mixEnrg_r[key_r[0, 0, 0]])*1e6
        
        out_l = (g_l, purity_l[key_l[1, 0, 1]], purity_l[key_l[0, 0, 1]], purity_l[key_l[1, 0, 0]], zz_l)
        out_r = (g_r, purity_r[key_r[1, 0, 1]], purity_r[key_r[0, 0, 1]], purity_r[key_r[1, 0, 0]], zz_r)

    return out_l, out_r



def zz_far_QC(coup_1, qubit_1, coup_2, qubit_2, g_q1_c1, g_q1_c2, g_q2_c2, g_q1_q2, g_c1_c2, g_long=0, regime=0):

    (spect_C1, phi_C1, q_C1) = map(np.copy, coup_1)
    (spect_C2, phi_C2, q_C2) = map(np.copy, coup_2)
       
    (spect_Q1, phi_Q1, q_Q1) = map(np.copy, qubit_1)
    (spect_Q2, phi_Q2, q_Q2) = map(np.copy, qubit_2)
    
    (mixEnrg_in, mixStates, mixH,
     opersC1, opersQ1, opersC2) = MixOfThreeSys(spect_C1, spect_Q1, spect_C2,
                                                q12=q_C1, q13=q_C1,
                                                q21=q_Q1, q23=q_Q1,
                                                q32=q_C2, q31=q_C2,
                                                opers1=np.asarray([phi_C1, q_C1]),
                                                opers2=np.asarray([phi_Q1, q_Q1]),
                                                opers3=np.asarray([phi_C2, q_C2]),
                                                g12=g_q1_c1, 
                                                g23=g_q1_c2, g31=g_c1_c2,
                                                numOfLvls=spect_C1.shape[0]*spect_Q1.shape[0]*spect_C2.shape[0], 
                                                project=True)
    
    
    key_in, purity_in, stlist_in = StatesPurity(mixStates, 
                                                (spect_C1.shape[0], spect_Q1.shape[0], spect_C2.shape[0]), 
                                                stList=True, dirtyBorder=0.0001)
    
    q_C1_new = opersC1[1]
    q_C2_new = opersC2[1]
    q_Q1_new = opersQ1[1]
    
    # mix of CQC and Q
    (mixEnrg, mixStates, mixH) = MixOfTwoSys(mixEnrg_in, spect_Q2, 
                                             g_q1_q2*q_Q1_new + g_q2_c2*q_C2_new + g_long*q_C1_new, 
                                             q_Q2, g=1, numOfLvls=mixEnrg_in.shape[0]*spect_Q2.shape[0], project=True)
    
    key, purity, stlist = StatesPurity(mixStates, (mixEnrg_in.shape[0], spect_Q2.shape[0]), stList=True, dirtyBorder=0.0001)

    
    if(regime==0):
        
        zz = (mixEnrg[key[key_in[1, 0, 0], 1]] - mixEnrg[key[key_in[0, 0, 0], 1]] - mixEnrg[key[key_in[1, 0, 0], 0]])
        pur_1 = purity[key[key_in[1, 0, 0], 1]]
        pur_2 = (purity[key[key_in[1, 0, 0], 0]] + purity[key[key_in[0, 0, 0], 1]])/2

    elif(regime==1):

        zz = (mixEnrg[key[key_in[1, 1, 0], 1]] - mixEnrg[key[key_in[0, 1, 0], 1]] \
              - mixEnrg[key[key_in[1, 1, 0], 0]] + mixEnrg[key[key_in[0, 1, 0], 0]])
        pur_1 = purity[key[key_in[1, 1, 0], 1]]
        pur_2 = (purity[key[key_in[1, 1, 0], 0]] + purity[key[key_in[0, 1, 0], 1]])/2


    return zz, (pur_1, pur_2)



# fun for qubits zz and gap optimization

def gap_one_side(Q, coup, g):

    (spect_Q, phi_Q, q_Q) = map(np.copy, Q)
    (spect_C, phi_C, q_C) = map(np.copy, coup)

    # mix of Q – C
    (spect, mixStates, _) = MixOfTwoSys(spect_Q, spect_C, 
                                                               q_Q, q_C, g=g, 
                                                               numOfLvls=spect_Q.shape[0]*spect_C.shape[0], 
                                                               project=True)

    key, _ = StatesPurity(mixStates, (spect_Q.shape[0], spect_C.shape[0]), dirtyBorder=0.0001)
        
    return abs(spect[key[1, 1]] - spect[key[1, 0]] - spect[key[0, 1]])


def light_gap_opt(Q, C, gap_t):

    def loss(g):
        gap = gap_one_side(Q, C, g)
        return 1e8*(gap - gap_t)**2

    opt = scp.optimize.dual_annealing(loss, bounds=[(0.1, 0.3)], maxiter=40)
    
    return opt.x[0]


def g_qubits_opt_assim(qubit_1, qubit_2, coup, gap_target, regime=0, regular=0.01, 
                       bounds=([0, 0.8], [0, 0.8]), border=0.2, mod='k^2/d^2',maxiter=200):

    (spect_Q1, phi_Q1, q_Q1) = map(np.copy, qubit_1)
    (spect_Q2, phi_Q2, q_Q2) = map(np.copy, qubit_2)
    
    (spect_C, phi_C, q_C) = map(np.copy, coup)

    def gap_loss(x):

        g_c_1 = x[0]
        g_c_2 = x[1]
        
        # емкостно смешиваем 3 подсистемы системы
        (mixEnrg, mixStates, mixH, opersC) =MixOfThreeSys(spect_Q1, spect_C, spect_Q2,
                                                          q12=q_Q1, q13=q_Q1,
                                                          q21=q_C, q23=q_C,
                                                          q32=q_Q2, q31=q_Q2,
                                                          opers2=np.asarray([phi_C, q_C]),
                                                          g12=g_c_1, g23=g_c_2, g31=0, 
                                                          numOfLvls=min(200,spect_Q1.shape[0]*spect_Q2.shape[0]*spect_C.shape[0]),
                                                          project=True)
        
        
        phi_C_mix = opersC[0]
        
        key, purity, stlist = StatesPurity(mixStates, (spect_Q1.shape[0], spect_C.shape[0], spect_Q2.shape[0]), stList=True)

        if(regime):
            _, leakage_param, _ = trans_isolation(init_st=key[1, 0, 1], target_st=key[1, 1, 1], pert_oper=phi_C_mix,
                                                  spectrum=mixEnrg, border=border, 
                                                  other_st_list=[key[1, 0, 0], key[0, 0, 1], key[0, 0, 0]], mod=mod)
        else:
            _, leakage_param, _ = trans_isolation(init_st=0, target_st=key[0, 1, 0], pert_oper=phi_C_mix,
                                                  spectrum=mixEnrg, border=border, 
                                                  other_st_list=[key[1, 0, 0], key[0, 0, 1], key[1, 0, 1]], mod=mod)            

        return (abs(leakage_param[0, 1]) - gap_target)**2 + (regular*g_c_1 + regular*g_c_2)**2

    sol = dual_annealing(gap_loss, bounds=bounds, maxiter=maxiter)
    g_c_1 = sol.x[0]
    g_c_2 = sol.x[1]

    def zz_loss(g_qq):
        # емкостно смешиваем 3 подсистемы системы
        (mixEnrg, mixStates, mixH) = MixOfThreeSys(spect_Q1, spect_C, spect_Q2,
                                                   q12=q_Q1, q13=q_Q1,
                                                   q21=q_C, q23=q_C,
                                                   q32=q_Q2, q31=q_Q2,
                                                   g12=g_c_1, g23=g_c_2, g31=g_qq, 
                                                   numOfLvls=min(200, spect_Q1.shape[0]*spect_Q2.shape[0]*spect_C.shape[0]),
                                                   project=True)
        
        key, purity, stlist = StatesPurity(mixStates, (spect_Q1.shape[0], spect_C.shape[0], spect_Q2.shape[0]), stList=True)
        zz = (mixEnrg[key[1, 0, 1]] - mixEnrg[key[0, 0, 1]] - mixEnrg[key[1, 0, 0]])*1e6

        
        return zz**2

    sol = minimize(zz_loss, x0=0.01)
    g_qq = sol.x[0]

    # емкостно смешиваем 3 подсистемы системы
    (mixEnrg, mixStates, mixH, opersC) = MixOfThreeSys(spect_Q1, spect_C, spect_Q2,
                                                       q12=q_Q1, q13=q_Q1,
                                                       q21=q_C, q23=q_C,
                                                       q32=q_Q2, q31=q_Q2,
                                                       opers2=np.asarray([phi_C, q_C]),
                                                       g12=g_c_1, g23=g_c_2, g31=g_qq, 
                                                       numOfLvls=min(200, spect_Q1.shape[0]*spect_Q2.shape[0]*spect_C.shape[0]),
                                                       project=True)
    phi_C_mix = opersC[0]
    
    key, purity, stlist = StatesPurity(mixStates, (spect_Q1.shape[0], spect_C.shape[0], spect_Q2.shape[0]), stList=True)
    zz = (mixEnrg[key[1, 0, 1]] - mixEnrg[key[0, 0, 1]] - mixEnrg[key[1, 0, 0]])*1e6

    if(regime):
        _, leakage_param, _ = trans_isolation(init_st=key[1, 0, 1], target_st=key[1, 1, 1], pert_oper=phi_C_mix,
                                              spectrum=mixEnrg, border=border, 
                                              other_st_list=[key[1, 0, 0], key[0, 0, 1], key[0, 0, 0]], mod=mod)
    else:
        _, leakage_param, _ = trans_isolation(init_st=0, target_st=key[0, 1, 0], pert_oper=phi_C_mix,
                                              spectrum=mixEnrg, border=border, 
                                              other_st_list=[key[1, 0, 0], key[0, 0, 1], key[1, 0, 1]], mod=mod)            

    print('gap:', abs(leakage_param[0, 1]))
    
    return g_c_1, g_c_2, g_qq, zz
    


def g_qubits_opt_sim(qubit_1, qubit_2, coup, gap_target, regime=0, border=0.2, mod='k^2/d^2'):

    (spect_Q1, phi_Q1, q_Q1) = map(np.copy, qubit_1)
    (spect_Q2, phi_Q2, q_Q2) = map(np.copy, qubit_2)
    
    (spect_C, phi_C, q_C) = map(np.copy, coup)

    def gap_loss(g_c):
        # емкостно смешиваем 3 подсистемы системы
        (mixEnrg, mixStates, mixH, opersT)=MixOfThreeSys(spect_Q1, spect_C, spect_Q2,
                                                             q12=q_Q1, q13=q_Q1,
                                                             q21=q_C, q23=q_C,
                                                             q32=q_Q2, q31=q_Q2,
                                                             opers2=np.asarray([phi_C, q_C]),
                                                             g12=g_c, g23=g_c, g31=0, 
                                                          numOfLvls=min(200,spect_Q1.shape[0]*spect_Q2.shape[0]*spect_C.shape[0]), 
                                                             project=True)
        
        
        phi_C_mix = opersT[0]
        
        key, purity, stlist = StatesPurity(mixStates, (spect_Q1.shape[0], spect_C.shape[0], spect_Q2.shape[0]), stList=True)

        if(regime):
            _, leakage_param, _ = trans_isolation(init_st=key[1, 0, 1], target_st=key[1, 1, 1], pert_oper=phi_C_mix,
                                                      spectrum=mixEnrg, border=border, 
                                                      other_st_list=[key[1, 0, 0], key[0, 0, 1], key[0, 0, 0]], mod=mod)
        else:
            _, leakage_param, _ = trans_isolation(init_st=0, target_st=key[0, 1, 0], pert_oper=phi_C_mix,
                                                      spectrum=mixEnrg, border=border, 
                                                      other_st_list=[key[1, 0, 0], key[0, 0, 1], key[1, 0, 1]], mod=mod)            

        return (abs(leakage_param[0, 1]) - gap_target)**2

    sol = scp.optimize.minimize(gap_loss, x0=0.1)
    g_c = sol.x[0]

    def zz_loss(g_qq):
        # емкостно смешиваем 3 подсистемы системы
        (mixEnrg, mixStates, mixH) = MixOfThreeSys(spect_Q1, spect_C, spect_Q2,
                                                       q12=q_Q1, q13=q_Q1,
                                                       q21=q_C, q23=q_C,
                                                       q32=q_Q2, q31=q_Q2,
                                                       g12=g_c, g23=g_c, g31=g_qq, 
                                                       numOfLvls=min(200, spect_Q1.shape[0]*spect_Q2.shape[0]*spect_C.shape[0]),
                                                       project=True)
        
        key, purity, stlist = StatesPurity(mixStates, (spect_Q1.shape[0], spect_C.shape[0], spect_Q2.shape[0]), stList=True)
        zz = (mixEnrg[key[1, 0, 1]] - mixEnrg[key[0, 0, 1]] - mixEnrg[key[1, 0, 0]])*1e6

        
        return zz**2

    sol = scp.optimize.minimize(zz_loss, x0=0.01)
    g_qq = sol.x[0]

    # емкостно смешиваем 3 подсистемы системы
    (mixEnrg, mixStates, mixH) = MixOfThreeSys(spect_Q1, spect_C, spect_Q2,
                                                   q12=q_Q1, q13=q_Q1,
                                                   q21=q_C, q23=q_C,
                                                   q32=q_Q2, q31=q_Q2,
                                                   g12=g_c, g23=g_c, 
                                                   g31=g_qq, 
                                                   numOfLvls=min(200, spect_Q1.shape[0]*spect_Q2.shape[0]*spect_C.shape[0]),
                                                   project=True)
    
    key, purity, stlist = StatesPurity(mixStates, (spect_Q1.shape[0], spect_C.shape[0], spect_Q2.shape[0]), stList=True)
    zz = (mixEnrg[key[1, 0, 1]] - mixEnrg[key[0, 0, 1]] - mixEnrg[key[1, 0, 0]])*1e6
    
    return g_c, g_qq, zz
    
    
def g_qubits_test(qubit_1, qubit_2, coup, g_c1, g_c2, g_qq, regime=0, border=0.2, mod='k^2/d^2'):

    (spect_Q1, phi_Q1, q_Q1) = map(np.copy, qubit_1)
    (spect_Q2, phi_Q2, q_Q2) = map(np.copy, qubit_2)
    
    (spect_C, phi_C, q_C) = map(np.copy, coup)


    (mixEnrg, mixStates, mixH, opersT)=MixOfThreeSys(spect_Q1, spect_C, spect_Q2,
                                                         q12=q_Q1, q13=q_Q1,
                                                         q21=q_C, q23=q_C,
                                                         q32=q_Q2, q31=q_Q2,
                                                         opers2=np.asarray([phi_C, q_C]),
                                                         g12=g_c1, g23=g_c2, g31=g_qq,
                                                         numOfLvls=min(200,spect_Q1.shape[0]*spect_Q2.shape[0]*spect_C.shape[0]), 
                                                         project=True)
        
        
    phi_C_mix = opersT[0]
    
    key, purity, stlist = StatesPurity(mixStates, (spect_Q1.shape[0], spect_C.shape[0], spect_Q2.shape[0]), stList=True)
    zz = (mixEnrg[key[1, 0, 1]] - mixEnrg[key[0, 0, 1]] - mixEnrg[key[1, 0, 0]])*1e6
    
    if(regime):
        _, leakage_param, _ = trans_isolation(init_st=key[1, 0, 1], target_st=key[1, 1, 1], 
                                              pert_oper=phi_C_mix,
                                              spectrum=mixEnrg, border=border, 
                                              other_st_list=[key[1, 0, 0], key[0, 0, 1], key[0, 0, 0]], mod=mod)
    else:
        _, leakage_param, _ = trans_isolation(init_st=0, 
                                              target_st=key[0, 1, 0], pert_oper=phi_C_mix,
                                              spectrum=mixEnrg, border=border, 
                                              other_st_list=[key[1, 0, 0], key[0, 0, 1], key[1, 0, 1]], mod=mod)   
    
    gap = abs(leakage_param[0, 1])
        
    return gap, zz
    
    
# fun for couplers connection optimization

def zz_far_QC_optimize(Q1, C1, Q2, C2, Q3, g_q1_c1, g_q2_c1, g_q2_c2, g_q3_c2, g_q1_q2, g_q2_q3, regime=0, optimizer='grad'):

    def loss(x):

        zz_1, _ = zz_far_QC(C1, Q2, C2, Q3, 
                                g_q1_c1=g_q2_c1, 
                                g_q1_c2=g_q2_c2, 
                                g_q2_c2=g_q3_c2,
                                g_q1_q2=g_q2_q3,
                                g_c1_c2=x, 
                                regime=regime)

        zz_2, _ = zz_far_QC(C2, Q2, C1, Q1, 
                                g_q1_c1=g_q2_c2, 
                                g_q1_c2=g_q2_c1, 
                                g_q2_c2=g_q1_c1,
                                g_q1_q2=g_q1_q2,
                                g_c1_c2=x,
                                regime=regime)

        if(optimizer == 'grad'):
            return (abs(zz_1) + abs(zz_2))**2 * 1e12
        elif(optimizer == 'root'):
            return (abs(zz_1) - abs(zz_2)) * 1e5

    
    if(optimizer == 'grad'):

        if(regime):
            opt = scp.optimize.minimize(loss, x0=[0], bounds=[(-0.2, 0)])
            g = opt.x[0]
        else:
            opt = scp.optimize.minimize(loss, x0=[0], bounds=[(0, 0.2)])
            g = opt.x[0]
            
    elif(optimizer == 'root'):
        
        if(regime):
            opt = scp.optimize.root(loss, x0=-0.01)
            g = opt.x[0]
        else:
            opt = scp.optimize.root(loss, x0=0.01)
            g = opt.x[0]
        

    zz_1, pur_1 = zz_far_QC(C1, Q2, C2, Q3, 
                                g_q1_c1=g_q2_c1, 
                                g_q1_c2=g_q2_c2, 
                                g_q2_c2=g_q3_c2,
                                g_q1_q2=g_q2_q3,
                                g_c1_c2=g, 
                                regime=regime)

    if(pur_1[0] < 0.993):
        print('ALARM, purity problem with 1!')
    
    zz_2, pur_2 = zz_far_QC(C2, Q2, C1, Q1, 
                                g_q1_c1=g_q2_c2, 
                                g_q1_c2=g_q2_c1, 
                                g_q2_c2=g_q1_c1,
                                g_q1_q2=g_q1_q2,
                                g_c1_c2=g,
                                regime=regime)
    if(pur_2[0] < 0.993):
        print('ALARM, purity problem with 1!')
    
    return g, (zz_1, zz_2, pur_1, pur_2)


def zz_far_QC_test(Q1, C1, Q2, C2, Q3, g_q1_c1, g_q2_c1, g_q2_c2, g_q3_c2, g_q1_q2, g_q2_q3, g, g_long=0, regime=0):



    zz_1, pur_1 = zz_far_QC(C1, Q2, C2, Q3, 
                                g_q1_c1=g_q2_c1, 
                                g_q1_c2=g_q2_c2, 
                                g_q2_c2=g_q3_c2,
                                g_q1_q2=g_q2_q3,
                                g_c1_c2=g,g_long, 
                                regime)

    if(pur_1[0] < 0.993):
        print('ALARM, purity problem with 1!')
            
    zz_2, pur_2 = zz_far_QC(C2, Q2, C1, Q1, 
                                g_q1_c1=g_q2_c2, 
                                g_q1_c2=g_q2_c1, 
                                g_q2_c2=g_q1_c1,
                                g_q1_q2=g_q1_q2,
                                g_c1_c2=g,g_long,
                                regime)

    if(pur_2[0] < 0.993):
        print('ALARM, purity problem with 1!')
            
    return g, (zz_1, zz_2, pur_1, pur_2)


def zz_far_QC_osc(coup_1, qubit_1, coup_2, qubit_2, g_q1_c1, g_q1_c2, g_q2_c2, g_q1_q2, g_c1_c2, 
                      g_long=0, gO_long=0, gO_Q=0,  El=0.5, Ec=4, g_o=0):

    # fun for zz calculation of 0-type cooplers with connection via an oscillator
    # system: C-Q-O-C
    
    (spect_C1, phi_C1, q_C1) = map(np.copy, coup_1)
    (spect_C2, phi_C2, q_C2) = map(np.copy, coup_2)
    
    (spect_Q1, phi_Q1, q_Q1) = map(np.copy, qubit_1)
    (spect_Q2, phi_Q2, q_Q2) = map(np.copy, qubit_2)

    # creation of binding oscillator
    (spect_O, phi_O, q_O) = Oscillator_circuit(El=El, Ec=Ec, numOfLvls=3)

    # mix of Q and extra O
    (mixEnrg, mixStates, _, opersQ1, opersO) = MixOfTwoSys(spect_Q1, spect_O,
                                                                            q1=q_Q1, q2=q_O,
                                                                            opers1=np.asarray([phi_Q1, q_Q1]),
                                                                            opers2=np.asarray([phi_O, q_O]),
                                                                            g=gO_Q,
                                                                            numOfLvls=14)

    phi_Q1 = opersQ1[0]
    q_Q1 = opersQ1[1]

    phi_O = opersO[0]
    q_O = opersO[1]


    key_F, purity_F, stlist_F = StatesPurity(mixStates, 
                                                 (spect_Q1.shape[0], spect_O.shape[0]),
                                                 stList=True, dirtyBorder=0.0001)

    
    (mixEnrg_in, mixStates, mixH,
     opersC1, opersQ1, opersC2) = MixOfThreeSys(spect_C1, mixEnrg, spect_C2,
                                                    q12=q_C1, q13=q_C1,
                                                    q21=g_q1_c1*q_Q1 + g_o*q_O,
                                                    q23=-g_q1_c2*q_Q1 - g_o*q_O,
                                                    q32=q_C2, q31=q_C2,
                                                    opers1=np.asarray([phi_C1, q_C1]),
                                                    opers2=np.asarray([phi_Q1, q_Q1, q_O]),
                                                    opers3=np.asarray([phi_C2, q_C2]),
                                                    g12=1, g23=1, g31=g_c1_c2,
                                                    numOfLvls=spect_C1.shape[0]*mixEnrg.shape[0]*spect_C2.shape[0]-5)


    key_in, purity_in, stlist_in = StatesPurity(mixStates, 
                                                    (spect_C1.shape[0], mixEnrg.shape[0], spect_C2.shape[0]), 
                                                    stList=True, dirtyBorder=0.0001)    
    q_C1_new = opersC1[1]
    q_C2_new = opersC2[1]
    q_Q1_new = opersQ1[1]    
    q_O_new = opersQ1[2]   
    # mix of CQC and Q
    (mixEnrg, mixStates, mixH) = MixOfTwoSys(mixEnrg_in, spect_Q2, 
                                                 g_long*q_C1_new + g_q1_q2*q_Q1_new + g_q2_c2*q_C2_new + gO_long*q_O_new, 
                                                 q_Q2, g=1, numOfLvls=38)
        
    key, purity, stlist = StatesPurity(mixStates, (mixEnrg_in.shape[0], spect_Q2.shape[0]), stList=True, dirtyBorder=0.0001)

        
    zz = (mixEnrg[key[key_in[1, 0, 0], 1]] - mixEnrg[key[key_in[0, 0, 0], 1]] - mixEnrg[key[key_in[1, 0, 0], 0]])
    pur_1001 = purity[key[key_in[1, 0, 0], 1]]*purity_in[key_in[1, 0, 0]]*purity_F[0]
    pur_1000 = purity[key[key_in[1, 0, 0], 0]]*purity_in[key_in[1, 0, 0]]*purity_F[0]
    pur_0001 = purity[key[key_in[0, 0, 0], 1]]*purity_in[key_in[0, 0, 0]]*purity_F[0]
    pur_0100 = purity[key[key_in[0, 1, 0], 0]]*purity_in[key_in[0, 1, 0]]*purity_F[1]

    return zz, (pur_1001, pur_1000, pur_0001, pur_0100)


# fun for light gap adjustment

def gap_one_side(Q, coup, g):

    # calculation of zz between qubit and coopler = gap
    
    (spect_Q, phi_Q, q_Q) = map(np.copy, Q)
    (spect_C, phi_C, q_C) = map(np.copy, coup)

    # mix of Q – C
    (spect, mixStates, _) = MixOfTwoSys(spect_Q, spect_C, 
                                                               q_Q, q_C, g=g, 
                                                               numOfLvls=spect_Q.shape[0]*spect_C.shape[0], 
                                                               project=True)

    key, _ = StatesPurity(mixStates, (spect_Q.shape[0], spect_C.shape[0]), dirtyBorder=0.0001)
        
    return abs(spect[key[1, 1]] - spect[key[1, 0]] - spect[key[0, 1]])


def light_gap_opt(Q, C, gap_t):

    # calculation of optimal g for target gap
    
    def loss(g):
        gap = gap_one_side(Q, C, g)
        return 1e8*(gap - gap_t)**2
    
    opt = scp.optimize.dual_annealing(loss, bounds=[(0.1, 0.3)], maxiter=40)
    
    return opt.x[0]


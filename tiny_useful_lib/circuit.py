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


# theoretical equations
def josephson_energy(R, delta=204e-6*e):
    return delta/8/e**2/R/1e9

def josephson_resistance(Ej, delta=204e-6*e):
    return delta/8/e**2/Ej/1e9

def oscillator_Ec(freq, Z=50):
    # calcs Ec(freq, Z) in GHz
    # Z - impedance
    return 2*np.pi*e/Fq * Z * freq

def oscillator_freq(Ec, Z=50):
    # calcs freq(Ec, Z) in GHz
    # Z - impedance
    return Ec/(2*np.pi*e/Fq * Z)

def oscillator_El(freq, Z=50):
    # calcs El(freq, Z) in GHz
    # Z - impedance
    return freq/Z*Fq/8/np.pi/e

def oscillator_g_a(g_n=None, g_phi=None, Z=50):
    # g_n - g for coopers pairs representation V = g*n_o*n
    # g_phi - g for coopers pairs representation V = g*phi_o*phi
    # Z - impedance
    
    if(g_n!=None):
        return 2*np.sqrt(2)*g_n * e * np.sqrt(Z/hpl)
    elif(g_phi!=None):
        return g_phi/(np.sqrt(2)* e * np.sqrt(Z/hpl))
    else:
        assert True

def oscillator_g_n(g_a, Z=50):
    # g_a - g for eigenenergies representation V = g*(a + a^+)*n
    # Z - impedance
    return g_a/(2*np.sqrt(2) * e * np.sqrt(Z/hpl))

def oscillator_g_phi(g_a, Z=50):
    # g_a - g for eigenenergies representation V = i*g*(a - a^+)*phi
    # Z - impedance
    return g_a*(np.sqrt(2) * e * np.sqrt(Z/hpl))

def oscillator_Z(freq=None, C=None, Ec=None, El=None):
    # freq in GHz
    # C in fF
    if(freq!=None and C!=None):
        freq *= 1e9
        C *= 1e-15
        return 1/(2*np.pi*freq*C)
    elif(Ec!=None and El!=None):
        return Fq/(4*np.pi*e)*np.sqrt(Ec/El)
        
        
# some staff to manage capacitance-energy matrixes


# adds a table enumeration
def print_table(M, integer=False):
    
    # allows to print a huge matrix
    np.set_printoptions(linewidth=120)

    A=M.shape[0] + 1
    B=M.shape[1] + 1
    
    if(integer):
        matrixForPrint = np.zeros((A, B), int)
    else:
        matrixForPrint = np.zeros((A, B), float)
    
    for i in range(A):
        matrixForPrint[i, 0] = i - 1
        
    for i in range(B):
        matrixForPrint[0, i] = i - 1
        
    matrixForPrint[0, 0] = 0
        
    matrixForPrint[1:(A+1), 1:(B+1)] = M[:, :]
    
    print(matrixForPrint)
    
    
def backward_quant(Ec_in, El_in=np.asarray([None]), S=np.asarray([None]), g_v_in=np.asarray([None])):
    # El – строка длины n (кол-во степеней свободы) с индуктивными энергиями каждой подсистемы
    # Ec – матрица nxn с емкостными энергиями каждой подсистемы и связями между ними
    # S – матрица перехода от реальных потоков к модельным
    # g_c – вектор связей с антенками в GHz/V
        
    Ec = np.copy(Ec_in)
    n = Ec.shape[0]
    
    if(El_in.any() == None):
        flag_L = False
        El = np.ones(n)
    else: 
        flag_L = True
        El = np.copy(El_in)
        
    flag = False
    
    if(S.any() == None):
        S = np.diag(np.ones(n))
    
    if(g_v_in.any() == None):
        g_v = np.zeros(n)
    else:
        flag = True
        g_v = np.copy(g_v_in)
    
    
    # заполняем обратную емкость InvC и индуктивность InvL
    InvC = np.zeros((n, n))
    InvL = np.zeros((n, n))
    
    # учтем, что энергия в GHz, а емкость в fF
    for i in range(n):
        for j in range(n):
            if(j == i):
                InvC[i, i] = Fq/e * Ec[i, i] / 10**6 * 10**3
                InvL[i, i] = 16 * np.pi **2 *e/Fq * El[i]
            elif(j > i):
                InvC[i, j] = InvC[j, i] = 1/2*Fq/e * Ec[i, j] / 10**6 * 10**3
    
    # находим матрицу емкости
    C = np.linalg.inv(InvC) * 10**3
    
    # переходим к реальной цепи заменами
    if(S.shape[0] != 0):
        St = np.copy(S)
        C = np.transpose(St) @ C @ St
        InvL = np.transpose(St) @ InvL @ St
        
    L = np.linspace(0, 0, n)
    
    C_v = np.zeros(n)
    
    # высчитываем C_v
    if(flag):
        
        S_inv = np.linalg.inv(S)
        C_v = Fq*C@S_inv@g_v*1e9
    
    # переведем C в удобный для чтения вид (вычтем C связей из диагонали)
    
    for i in range(n):
        
        # также вернем L
        if(InvL[i, i] != 0):
            L[i] = 1/InvL[i, i]
        else:
            L[i] = 99999999
        
        for j in range(i + 1, n):    
            
            # вычитаем антенку
            C[j, j] -= C_v[j]
            
            if(C[i, j]!=0):
                C[j, i] = 0
                C[i, j] = -C[i, j]
                
                # чистим диагональ
                C[i, i] = C[i, i] - abs(C[i, j])
                C[j, j] = C[j, j] - abs(C[i, j])
    
    # на выходе емкости в fF и индуктивности в nH
    
    if(flag_L):
        if(flag):
            return (L, C, C_v)
        else:
            return (L, C)
    else:
        if(flag):
            return (C, C_v)
        else:
            return C


def forward_quant(C_in, L_in=np.asarray([None]), S=np.asarray([None]), C_v_in=np.asarray([None])):
    # C – матрица nxn с емкостями
    # S – матрица перехода от реальных потоков к модельным
    # C_v – вектор разм. n из антенковых ёмкостей на iый узел (каждую отдельную антенку нужно считать независимо)
        
    C = np.copy(C_in)
    n = C.shape[0]
    
    if(L_in.any() == None):
        flag_L=False
        L = np.ones(n)
    else:
        flag_L=True
        L = np.copy(L_in)
        
    if(C_v_in.any() == None):
        C_v = np.zeros(n)
    else:
        C_v = np.copy(C_v_in)
        
    if(S.any() == None):
        S = np.diag(np.ones(n))
    
    InvL = np.zeros((n, n))
    # заполним нормльно матрицы
    for i in range(n):
        
        C[i, i] += C_v[i]
        
        InvL[i, i] = 1/L[i]
        for j in range(n):
            if(i < j and C[i, j]!=0):
                
                
                C[j, i] = C[i, j] = - C[i, j]
                
                C[i, i] += abs(C[i, j])
                C[j, j] += abs(C[i, j])
    
    
    # переходим к модельной цепи заменами
    St = np.copy(S)
    St = np.linalg.inv(St)

    C_original = np.copy(C)
    C = np.transpose(St) @ C @ St
    InvL = np.transpose(St) @ InvL @ St
    
    # находим матрицу энергий
    CInv = np.linalg.inv(C)
    if(C_v_in.any() != 0):
        CInv_or = np.linalg.inv(C_original)
    
    Ec = np.zeros((n, n))
    El = np.zeros(n)
    
    # учтем, что энергия в GHz, а емкость в fF
    for i in range(n):
        El[i] = InvL[i, i] * Fq /16/np.pi**2/e
        
        for j in range(i, n):
            if(j == i):
                Ec[i, i] = e/Fq * CInv[i, i] * 10**6
            elif(j > i):
                Ec[i, j] = Ec[j, i] = 2*e/Fq * CInv[i, j] * 10**6
                Ec[j, i] = 0
                
    if(flag_L):
        if(C_v_in.any() == None):
            return (El, Ec)
        else:
            # находим множители связи с антенкой g_v*V_in = g, g_v – GHz/V
            g_v = S@CInv_or@C_v_in/Fq/1e9
            return (El, Ec, g_v)
    else:
        if(C_v_in.any() == None):
            return Ec
        else:
            # находим множители связи с антенкой g_v*V_in = g, g_v – GHz/V
            g_v = S@CInv_or@C_v_in/Fq/1e9
            return (Ec, g_v)        
    

def backward_quant_natural(Ec0, deltaEcMax, weightС, zeal=10, targetC=np.asarray(None), S=np.asarray([None])):
    # энергии в ГГц!!!, a С в фФ
    # weightС - матрица с весами зануления емкостей
    
    if(S.any() == None):
        S = np.diag(np.ones(Ec0.shape[0]))
        
    size = Ec0.shape[0]
    indexSpace = []
    valueSpace = []

    Ec0*=1e3
    deltaEcMax*=1e3
    
    if(targetC.any()==None):
        targetC=np.zeros((size, size))
    
    # оперделим область параметров с помощью deltaEc
    bounds = []
    
    for n in range(size):
        for m in range(size):
            if(deltaEcMax[n, m] != 0):
                indexSpace.append([n, m])
                bounds.append((-deltaEcMax[n, m], +deltaEcMax[n, m]))
                valueSpace.append(deltaEcMax[n, m])
    
    dim = len(indexSpace)

    
    # работаем в области deltaEc
    
    def loss(deltaEc):
        Ec = np.copy(Ec0)
        for i in range(dim):
            n = indexSpace[i][0]
            m = indexSpace[i][1]
            
            Ec[n, m] += deltaEc[i]
            
        C = backward_quant(Ec/1000, S=S)
        
        answ = 0
        
        for n in range(size):
            for m in range(size - n):
                answ += weightC[n, n + m] * (C[n, n + m] - targetC[n, n + m])**2
        return answ
        
    # теперь устроим оптимизацию с рандомными начальными точками и выберем лучшее
    ans = np.zeros(dim)
    lossVal = loss(ans)
    
    for sample in range(zeal):
        
        # генерируем x0
        x0 = np.random.rand(dim) * 2*np.asarray(valueSpace) - np.asarray(valueSpace)
        # оптимизируем
        sol = minimize(loss, x0=x0, bounds=bounds)
        
        if(sol.success != True):
            continue
        
        if(sol.fun < lossVal):
            lossVal = sol.fun
            ans = sol.x
    
    finalAns = np.copy(Ec0)
    
    for i in range(dim):
        n = indexSpace[i][0]
        m = indexSpace[i][1]
            
        finalAns[n, m] += ans[i]
        
    C = forward_quant(El/1000, finalAns/1000, S=S)
    
    return(finalAns, C)


def forward_quant_opt(C0, S, deltaCMax, weightEc, zeal=10, targetEc=np.asarray(None), S=np.asarray([None])):
    # энергии в ГГц!!!, a С в фФ
    # weightС - матрица с весами зануления емкостей
    
    if(S.any() == None):
        S = np.diag(np.ones(C0.shape[0]))
        
    size = C0.shape[0]
    indexSpace = []
    valueSpace = []
    if(targetEc.any()==None):
        targetEc=np.zeros((size, size))

    targetEc *= 1e3
    
    # оперделим область параметров с помощью deltaEc
    bounds = []
    
    for n in range(size):
        for m in range(size):
            if(deltaCMax[n, m] != 0):
                indexSpace.append([n, m])
                bounds.append((-deltaCMax[n, m], +deltaCMax[n, m]))
                valueSpace.append(deltaCMax[n, m])
    
    dim = len(indexSpace)

    
    # работаем в области deltaEc
    
    def loss(deltaC):
        C = np.copy(C0)
        for i in range(dim):
            n = indexSpace[i][0]
            m = indexSpace[i][1]
            
            C[n, m] += deltaC[i]
            
        Ec = forward_quant(C, S=S)
        
        Ec = Ec*1000
        answ = 0
        
        for n in range(size):
            for m in range(size - n):
                answ += weightEc[n, n + m] * (Ec[n, n + m] - targetEc[n, n + m])**2
        
        return answ
        
    # теперь устроим оптимизацию с рандомными начальными точками и выберем лучшее
    ans = np.zeros(dim)
    lossVal = loss(ans)
    print(lossVal, '\n')
    
    for sample in range(zeal):

        # генерируем x0
        x0 = np.random.rand(dim) * 2*np.asarray(valueSpace) - np.asarray(valueSpace)
        # оптимизируем
        sol = minimize(loss, x0=x0, bounds=bounds)
        
        
        if(sol.success != True):
            continue
        
        if(sol.fun < lossVal):
            print(lossVal, '\n')
            lossVal = sol.fun
            ans = sol.x
    
    finalAns = np.copy(C0)
    
    for i in range(dim):
        n = indexSpace[i][0]
        m = indexSpace[i][1]
            
        finalAns[n, m] += ans[i]
        
    Ec = forward_quant(finalAns, S=S)
    
    return(finalAns, Ec)

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
from scipy.linalg import cosm, expm, sqrtm, det

e=1.6*10**(-19)
hpl=1.05*10**(-34)
Rq=hpl/(e)**2
Fq=(2*np.pi*hpl)/(2*e)
kb=1.38*10**(-23)
epsilon0=8.85*10**(-12)
mu0=1.256*1e-6
epsilon=10
d=2*10**(-9)
Z0=50
j=0.5*10**6
S=(1000*500)*10**(-18)


# rounding of number with n values after the first non-zero one
def around(x, n):
    
    deg = np.log(x)/np.log(10)
    
    if(deg > 0): deg = int(deg)
    else: deg = int(np.ceil(deg))
        
    deg -= n
    
    return np.around(x, -deg)


# make all non-zero matrix element equal to 1
def to_ones(x):
    if(x!=0): return 1
    else: return 0
to_ones = np.vectorize(to_ones)


def subspace(M, indices):
    
    tmp = []
    tmp.append(indices)
    tmp = np.asarray(tmp, dtype=int)

    y = np.ones((tmp.shape[0], 1), dtype=int)@tmp
    x = y.transpose()
    
    return M[x, y]
    


def kron(*opers):

    prod = np.kron(opers[0], opers[1])

    for n in range(len(opers) - 2):

        prod =np.kron(prod, opers[n + 2])

    return prod
    


def trans_isolation(init_st, target_st, pert_oper, spectrum, border, other_st_list=[], mod='k^2/d', 
                    rounding=3, multiphoton_trigger=1e-6):

    # mod 0: search based on k**2/delta, where k = m_tr/m_aim (inspired by three-lvl Rabi), here border=(k**2/delta)_min
    # mod 1: search based on k**2/delta**2, where k = m_tr/m_aim (inspired by three-lvl Rabi), here border=(k**2/delta**2)_min
    # mod 2: search with border[0] – minimal value of k and border[1] – maximum transition's frequencies delta
    # mod 3: TWO-PHOTON LEAKAGE with border[0] – minimum of k=sum_v(|k_iv*k_vf/(f-2f_virt)|) and border[1] – maximum |f_signal-f/2|
    
    if(mod=='k^2/d' or mod==0):
        mod = 0
    elif(mod=='k^2/d^2' or mod==1):
        mod = 1
    elif(mod=='k_min, d_max' or mod==2):
        mod = 2
    elif(mod=='two-photon' or mod==3):
        mod = 3
        
    # output: leakage_st[0] – init leakage states, leakage_st[1] – target leakage states; leakage_param[0] – k, leakage_param[1] – delta
    
    other_st_list = np.asarray(other_st_list)
    full_st_list = np.zeros((2 + other_st_list.shape[0]), int)
    full_st_list[0] = init_st
    full_st_list[1] = target_st
    full_st_list[2:] = other_st_list

    m_0 = abs(pert_oper[init_st, target_st])
    f_0 = abs(spectrum[init_st] - spectrum[target_st])

    # arrays for output
    leakage_trans = []
    leakage_k = []
    leakage_delta = []

    
    # transitions init -> fin
    for init in full_st_list:
        for fin in range(spectrum.shape[0]):

            flag = False
            for st in full_st_list:
                if(fin == st):
                    flag = True
                
            if(flag):
                continue

            m = abs(pert_oper[init, fin])
            k = m/m_0
            delta = abs(abs(spectrum[init] - spectrum[fin]) - f_0)

            if(delta == 0):
                continue

            if(mod==0 and k**2/delta > border):
                
                flag = True
                for trans in leakage_trans: 
                    if(trans[0] == init and trans[1] == fin): flag = False
                if(flag):
                    leakage_trans.append([init, fin])
                    leakage_k.append(k)
                    leakage_delta.append(delta)
                
            elif(mod==1 and k**2/delta**2 > border):
                
                flag = True
                for trans in leakage_trans: 
                    if(trans[0] == init and trans[1] == fin): flag = False
                if(flag):
                    leakage_trans.append([init, fin])
                    leakage_k.append(k)
                    leakage_delta.append(delta)

            elif(mod==2 and k > border[0] and delta < border[1]):
                
                flag = True
                for trans in leakage_trans: 
                    if(trans[0] == init and trans[1] == fin): flag = False
                if(flag):
                    leakage_trans.append([init, fin])
                    leakage_k.append(k)
                    leakage_delta.append(delta)
               
            elif(mod==3):
                
                k_multi = 0
                
                for virt in range(spectrum.shape[0]):
                    
                    flag = False
                    for st in full_st_list:
                        if(virt == st):
                            flag = True

                    if(flag):
                        continue
                        
                    k_multi += abs(pert_oper[init, virt]*pert_oper[virt, fin]/m_0**2\
                    /(spectrum[fin] - 2*spectrum[virt] + spectrum[init]))
                    
                delta = abs(abs(spectrum[init] - spectrum[fin])/2 - f_0)
                
                flag = True
                for trans in leakage_trans: 
                    if(trans[0] == init and trans[1] == fin): flag = False
                if(flag):
                    if(k_multi > border[0] and delta < border[1]):
                        leakage_trans.append([init, fin])
                        leakage_k.append(k_multi)
                        leakage_delta.append(delta)

    if(mod==0):
        tmp = np.asarray(leakage_k)**2/np.asarray(leakage_delta)
        sort = np.argsort(np.asarray(tmp))
        sort = np.flip(sort)
        
    if(mod==1):
        tmp = np.asarray(leakage_k)**2/np.asarray(leakage_delta)**2
        sort = np.argsort(np.asarray(tmp))
        sort = np.flip(sort)

    if(mod==2 or mod==3):
        sort = np.argsort(np.asarray(leakage_delta))
        

    leakage_st = np.zeros((sort.shape[0], 2), int)
    leakage_param = np.zeros((sort.shape[0], 2))
    string_list = []
    
    for i in range(sort.shape[0]):
        
        leakage_st[i, 0] = leakage_trans[sort[i]][0]
        leakage_st[i, 1] = leakage_trans[sort[i]][1]
        
        leakage_param[i, 0] = leakage_k[sort[i]]
        leakage_param[i, 1] = leakage_delta[sort[i]]
        
        tmp_1 = leakage_param[i, 0]**2/leakage_param[i, 1]
        tmp_2 = leakage_param[i, 0]**2/leakage_param[i, 1]**2

        if(mod!=3):
            string_list.append("{0} -> {1} : k={2}, ∆={3}, k**2/∆={4}, k**2/∆**2={5}".format(leakage_st[i, 0], 
                                                                                        leakage_st[i, 1], 
                                                                                        around(leakage_param[i, 0], rounding), 
                                                                                        around(leakage_param[i, 1], rounding), 
                                                                                        around(tmp_1, rounding), 
                                                                                        around(tmp_2, rounding)))
        else:
            string_list.append("{0} -> {1} : ∑|k_iv*k_vf/(fr_f-2fr_v)|={2}, ∆={3}".format(leakage_st[i, 0], leakage_st[i, 1], 
                                                                  around(leakage_param[i, 0], rounding), 
                                                                  around(leakage_param[i, 1], rounding)))
    
    return leakage_st, leakage_param, string_list


def SQUID_flux_finder(Ej1, Ej2, Ej):
    # finde proper flux point for SQUID to obtain
    # appropriate value of effective Ej
    # (based on equation from quantum engenering guide)
    
    if(Ej1 + Ej2 < Ej or abs(Ej1 - Ej2) > Ej):
        
        raise ValueError('Unreachable target Ej value')
    
    gamma = Ej2/Ej1
    d = (gamma - 1)/(gamma + 1)
    Phi = np.arcsin(np.sqrt((Ej**2 - (Ej1 + Ej2)**2)/(Ej1 + Ej2)**2/(d**2 - 1)))

    return Phi/np.pi

# funcs for index management
def index_linear(coord_in, dim_in):

    # return index in lianerized mixed basis (reverse of the next function)
    coord = np.asarray(coord_in)
    dim = np.asarray(dim_in)

    index = 0
    
    for n in range(coord.shape[0] - 1):

        index = (index + coord[n])*dim[n + 1]

    return index


def index_coord(index, dim_in):
    # rebuild index in tensor product space A@B@.. into N coordinat 
    # representation i -> (a_i, b_i, ...) 
    # dim_in = (dim(A), dim(B), ...)
    
    dim = np.copy(np.asarray(dim_in))
    coord = np.zeros(dim.shape[0], int)

    dim[0] = 1
    
    for it in range(dim.shape[0]):

        d_prod = np.prod(dim)
        coord[it] = int(index//d_prod)
        index -= coord[it]*d_prod
        
        if(it < dim.shape[0] - 1):
            dim[it + 1] = 1
    
    return coord


# Funcs for fits
def parabolic_fit(X_in, Y_in, bounds, maxiter=300, no_local_search=False, x0=None):
    # annealing based parabolic fitter
    
    X = np.asarray(X_in)
    Y = np.asarray(Y_in)
    # model A*(x - x_0)**2 + B**2
    def loss(x):
        
        a = x[0]
        b = x[1]
        c = x[2]
    
        return np.sum((Y - a*X**2 - b*X - c)**2)
    
    # оптимизируем
    sol = sc.optimize.dual_annealing(loss, bounds=bounds, 
                                     maxiter=maxiter,
                                     no_local_search=no_local_search,
                                     x0=x0)
    ans = sol.x
    
    hess = nd.Hessian(loss, step=1e-4, method='central', order=2)(ans)
    # аналогично scipy.optimize.curve_fit
    cov = np.linalg.inv(hess)*sol.fun/(X.shape[0] - ans.shape[0]) * 2

    return ans, cov
    

def linear_regression_fit(X_in, Y_in):
    # analytical solver for linear regression problem X_in.shape[0] -> dots, X_in.shape[1] -> dimensions (constant is included)
    
    Y = np.copy(np.asarray(Y_in))
    X_0 = np.copy(np.asarray(X_in))
    
    if(len(X_0.shape) == 1):
        
        X = np.ones((X_0.shape[0], 2))
        X[:,1] = X_0
        
    else:
    
        X = np.ones((X_0.shape[0], X_0.shape[1] + 1))
        X[:,1:] = X_0
    
    tmp = np.einsum('mi,mk->ik', X, X)
    tmp = np.linalg.inv(tmp)
    tmp = np.einsum('ik,nk->in', tmp, X)
    
    beta = np.einsum('in,n->i', tmp, Y)
        
    mse = np.sqrt(np.sum((Y - np.einsum('k,ik->i', beta, X))**2))/Y.shape[0]

    return (beta, mse)



def dagger(a):
    return np.conjugate(a.transpose())


def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw=None, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (M, N).
    row_labels
        A list or array of length M with the labels for the rows.
    col_labels
        A list or array of length N with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if ax is None:
        ax = plt.gca()

    if cbar_kw is None:
        cbar_kw = {}

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # Show all ticks and label them with the respective list entries.
    ax.set_xticks(np.arange(data.shape[1]), labels=col_labels)
    ax.set_yticks(np.arange(data.shape[0]), labels=row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    ax.spines[:].set_visible(False)

    ax.set_xticks(np.arange(data.shape[1] + 1) - .5, minor=True)
    ax.set_yticks(np.arange(data.shape[0] + 1) - .5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=("black", "white"),
                     threshold=None, **textkw):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A pair of colors.  The first is used for values below a threshold,
        the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max()) / 2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts


def funcHeatmap(x, pos):
    return "{:.2f}".format(x).replace("0.", ".").replace("1.00", "1")


def StatesRepr(state, bSize, size):
    # генерим подписи осей, где oX – первый базис, а oY – второй
    oX = np.zeros(size)
    oY = np.zeros(size)

    for n in range(size):
        oX[n] = str(n)
        oY[n] = str(n)

    # задаем матричку для диаграммы
    matrix = np.zeros((size, size))

    for n in range(size):
        for m in range(size):
            matrix[n, m] = abs(state[n + m * bSize]) ** 2

    fig, ax = plt.subplots()
    plt.rcParams["figure.figsize"] = (5, 5)
    im, _ = heatmap(matrix, oX, oY, ax=ax,
                    cmap="YlGn", vmin=0, vmax=1., cbarlabel="Probability")

    annotate_heatmap(im, valfmt=FuncFormatter(funcHeatmap), size=40 / np.sqrt(size))

    fig.tight_layout()
    plt.show()


def Oscillator(freq, Z=None, numOfLvls=20):
    # собственные значения энергии
    eigEnergies = np.linspace(0, freq * (numOfLvls - 1), numOfLvls)

    # оператор уничтожения
    a = np.zeros((numOfLvls, numOfLvls), dtype=complex)
    for n in range(numOfLvls - 1):
        a[n, n + 1] = np.sqrt(n + 1)

    # оператор рождения
    at = np.zeros((numOfLvls, numOfLvls), dtype=complex)
    for n in range(numOfLvls - 1):
        at[n + 1, n] = np.sqrt(n + 1)

    if(Z==None):
        return (eigEnergies, at, a)
    else:
        phi = -1j*np.sqrt(2*np.pi*e*Z/Fq)*(at - a)
        q = np.sqrt(Fq/(8*np.pi*e*Z))*(at + a)
        return (eigEnergies, phi, q)


def Oscillator_circuit(El, Ec, numOfLvls=20):
    
    (eigEnergies, at, a) = Oscillator(2*np.sqrt(Ec*El), numOfLvls=numOfLvls)
    
    phi = -1j*(Ec/4/El)**0.25*(at - a)
    q = (El/4/Ec)**0.25*(at + a)
    
    return (eigEnergies, phi, q)


def OperInEigStates(eigVectors, gridSize=0, h=0, leftBorder=0):
    # построим проекции канонических q и p на собственные векторы в q представлении

    # построим проектор pr (действ. на строки) и матрицы q и p в координатном базисе сетки (q)
    pr = eigVectors
    q = np.zeros((gridSize, gridSize), dtype=complex)
    p = np.zeros((gridSize, gridSize), dtype=complex)

    for n in range(gridSize):
        # поток
        q[n, n] = h * n + leftBorder

        # заряд
        if (n == 0):
            p[n, n + 1] = -1j / (2 * h)
        elif (n == gridSize - 1):
            p[n, n - 1] = 1j / (2 * h)
        else:
            p[n, n + 1] = -1j / (2 * h)
            p[n, n - 1] = 1j / (2 * h)

    # проецируем
    pNew = np.conjugate(pr.transpose()) @ p @ pr
    qNew = np.conjugate(pr.transpose()) @ q @ pr

    return (qNew, pNew)


def Fluxonium_old(Ej, El, Ec, gridSize=100, numOfLvls=100, leftBorder=-20, rightBorder=20, F=0):
    # Ej, El и Ec - эффективные энергии на джоз. эл., индуктивности и емкости

    # h - шаг сетки
    h = (rightBorder - leftBorder) / gridSize

    # H - матрица гамильтониана
    H = np.zeros((gridSize, gridSize), dtype=complex)

    # заполнение H по разностной схеме 2 порядка с нулевыми гран.усл.
    for n in range(gridSize):

        phi = h * n + leftBorder

        if (n == 0):
            H[n, n] = 2 * Ec / h ** 2 + El * (phi + 2 * np.pi * F) ** 2 - Ej * np.cos(phi - np.pi)
            H[n, n + 1] = -Ec / h ** 2
        elif (n == gridSize - 1):
            H[n, n] = 2 * Ec / h ** 2 + El * (phi + 2 * np.pi * F) ** 2 - Ej * np.cos(phi - np.pi)
            H[n, n - 1] = -Ec / h ** 2
        else:
            H[n, n] = 2 * Ec / h ** 2 + El * (phi + 2 * np.pi * F) ** 2 - Ej * np.cos(phi - np.pi)
            H[n, n - 1] = -Ec / h ** 2
            H[n, n + 1] = -Ec / h ** 2

    # диагонализация
    (eigEnergies, eigVectors) = eigsh(H, k=numOfLvls, which='SA', maxiter=4000)

    order = np.argsort(np.real(eigEnergies))
    eigEnergies = eigEnergies[order]
    eigVectors = eigVectors[:, order]

    (phi, q) = OperInEigStates(eigVectors, gridSize=gridSize, h=h, leftBorder=leftBorder)
    
    eigEnergies = eigEnergies - eigEnergies[0]
    
    return (eigEnergies, phi, q)

def Fluxonium(Ej, El, Ec, gridSize=None, numOfLvls=5, F=0, Q=0):
    # Ej, El и Ec - эффективные энергии на джоз. эл., индуктивности и емкости

    if(gridSize == None):
        gridSize = int(1.5*numOfLvls) + 20

    nu=2*np.sqrt(El*Ec)
    _, at, a = Oscillator(nu, numOfLvls=gridSize)
    one = np.diag(np.ones(gridSize))
    phi = -1j*(Ec/4/El)**0.25*(at - a)
    q = (El/4/Ec)**0.25*(at + a)
    
    H = nu*at@a + 2*Ec*Q*q - Ej*cosm(phi + 2*np.pi*one*F)
    (e, v) = eigsh(H, k=numOfLvls, which='SA', maxiter=5000)
    
    sorted_indices = np.argsort(e)
    eigEnergies = e[sorted_indices]
    eigVectors = v[:, sorted_indices]
    
    phi = dagger(eigVectors)@phi@eigVectors
    q = dagger(eigVectors)@q@eigVectors
    eigEnergies = eigEnergies - eigEnergies[0]
    
    return (eigEnergies, phi, q)


def Transmon(Ej1, Ej2, Ec, gridSize=None, numOfLvls=100, F=0, Q=0):

    if(gridSize == None):
        gridSize=int(1.5*numOfLvls) + 10
    
    # Ej и Ec - эффективные энергии на джоз. эл. и емкости

    # h - шаг сетки (из-за дескретности заряда шаг = 1)
    h = 1

    # H - матрица гамильтониана
    H = np.zeros((2 * gridSize + 1, 2 * gridSize + 1), dtype=complex)

    # заполнение H по разностной схеме 2 порядка с нулевыми гран.усл.
    for n in range(2 * gridSize + 1):

        q = h * n - gridSize

        if (n == 0):
            H[n, n] = Ec * (q + Q) ** 2
            H[n, n + 1] = -(Ej1 + Ej2) / 2 * np.cos(2*np.pi*F / 2) + (Ej2 - Ej1) / 2j * np.sin(2*np.pi*F / 2)
        elif (n == 2 * gridSize):
            H[n, n] = Ec * (q + Q) ** 2
            H[n, n - 1] = -(Ej1 + Ej2) / 2 * np.cos(2*np.pi*F / 2) - (Ej2 - Ej1) / 2j * np.sin(2*np.pi*F / 2)
        else:
            H[n, n] = Ec * (q + Q) ** 2
            H[n, n - 1] = -(Ej1 + Ej2) / 2 * np.cos(2*np.pi*F / 2) - (Ej2 - Ej1) / 2j * np.sin(2*np.pi*F / 2)
            H[n, n + 1] = -(Ej1 + Ej2) / 2 * np.cos(2*np.pi*F / 2) + (Ej2 - Ej1) / 2j * np.sin(2*np.pi*F / 2)

    # диагонализация
    (eigEnergies, eigVectors) = eigsh(H, k=numOfLvls, which='SA', maxiter=4000)

    order = np.argsort(np.real(eigEnergies))
    eigEnergies = eigEnergies[order]
    eigVectors = eigVectors[:, order]

    (q, phi) = OperInEigStates(eigVectors, gridSize=2 * gridSize + 1, h=h, leftBorder=-gridSize)

    eigEnergies = eigEnergies - eigEnergies[0]
    
    return (eigEnergies, phi, q)

def StatesPurity(states, nS, stList=False, dirtyBorder=0.01):
    # расшифровывает собственные состояния тензорных произведений 2 и 3 систем
    tmp = np.asarray(nS)
    outList = []
    
    if(tmp.shape[0] == 2):
        N1 = tmp[0]
        N2 = tmp[1]
        
        key = np.zeros((N1, N2), dtype=object)
        purity = np.zeros(N1*N2)
        
        for n in range(states.shape[1]):
            s = abs(states[:, n])
            s = s.reshape(N1, N2)
            
            oldNum = key[np.unravel_index(s.argmax(), s.shape)]
            purity[n] = abs(s[np.unravel_index(s.argmax(), s.shape)])**2 
            
            if(oldNum != 0):
                if(purity[oldNum] <= purity[n]):
                    key[np.unravel_index(s.argmax(), s.shape)] = int(n)
                    
            else:
                key[np.unravel_index(s.argmax(), s.shape)] = int(n)
                    
            if(stList):
                string = str(n) + ': '
                
                while(True):
                    localPur = abs(s[np.unravel_index(s.argmax(), s.shape)])**2
                    if(localPur > dirtyBorder):
                        string += str(localPur * 100)+'% of '+str(np.unravel_index(s.argmax(), s.shape))+"\n    "
                    else:
                        break
                    
                    s[np.unravel_index(s.argmax(), s.shape)] = 0
                    
                outList.append(string)
            
        # выделяем подавленные состояния с помощью None
        for n in range(key.shape[0]):
            for m in range(key.shape[1]):
                    if(n + m != 0 and key[n, m] == 0):
                        key[n, m] = None
        
        if(stList):
            return (key, purity, outList)
        else:
            return (key, purity)
        
        
    elif(tmp.shape[0] == 3):
        N1 = tmp[0]
        N2 = tmp[1]
        N3 = tmp[2]
        
        key = np.zeros((N1, N2, N3), dtype=object)
        purity = np.zeros(N1*N2*N3)
        
        for n in range(states.shape[1]):
            s = abs(states[:, n])
            s = s.reshape(N1, N2, N3)
            
            oldNum = key[np.unravel_index(s.argmax(), s.shape)]
            purity[n] = abs(s[np.unravel_index(s.argmax(), s.shape)])**2 
            
            if(oldNum != 0):
                if(purity[oldNum] <= purity[n]):
                    key[np.unravel_index(s.argmax(), s.shape)] = int(n)
                    
            else:
                key[np.unravel_index(s.argmax(), s.shape)] = int(n)
                    
            if(stList):
                string = str(n) + ': '
                
                while(True):
                    localPur = abs(s[np.unravel_index(s.argmax(), s.shape)])**2
                    if(localPur > dirtyBorder):
                        string += str(localPur * 100)+'% of '+str(np.unravel_index(s.argmax(), s.shape))+"\n    "
                    else:
                        break
                    
                    s[np.unravel_index(s.argmax(), s.shape)] = 0
                    
                outList.append(string)
            
        # выделяем подавленные состояния с помощью None
        for n in range(key.shape[0]):
            for m in range(key.shape[1]):
                for k in range(key.shape[2]):
                    if(n + m + k != 0 and key[n, m, k] == 0):
                        key[n, m, k] = None
                        
        if(stList):
            return (key, purity, outList)
        else:
            return (key, purity)
        
        
    else:
        print("To much")
        return 1
    
# new
def mix_two_sys(spect1, spect2, q1, q2, opers1=[], opers2=[], 
                g=0, numOfLvls=5, purity_calc=True, stList=True, dirtyBorder=0.01,
                eigVectors_output=False, project=True):
    # связываем две системы через операторы q1 и q2, попутно расширяя их операторы на общее пространство
    # opers – список из матриц операторов соотв. системы
    
    dim_1 = spect1.shape[0]
    dim_2 = spect2.shape[0]
    
    # единичная матрица 
    E1 = np.diag(np.ones(dim_1))
    E2 = np.diag(np.ones(dim_2))
    
    # диагонализованные гамильтонианы
    H1 = np.diag(spect1)
    H2 = np.diag(spect2)
    
    # объединяем линейные пространства
    H1 = np.kron(H1, E2)
    H2 = np.kron(E1, H2)    
    
    # q в общем базисе
    M = np.kron(q1, q2)
    
    # полный гамильтониан
    H = H1 + H2 + g * M
                                
    # диагонализация
    (eigEnergies, eigVectors) = eigsh(H, k=numOfLvls, which='SA', maxiter=4000)
    
    order=np.argsort(np.real(eigEnergies))
    eigEnergies=eigEnergies[order]
    eigVectors=eigVectors[:, order]
        
    # сдвигаем 0
    eigEnergies = eigEnergies - eigEnergies[0]
    output = [eigEnergies]
    
    if(eigVectors_output): output.append(eigVectors)
        
    if(purity_calc):
        purity_info = StatesPurity(eigVectors, (dim_1, dim_2), stList=stList, dirtyBorder=dirtyBorder)
        output.append(purity_info)
        
    # перетягиваем операторы
    if(len(opers1) != 0):
        newOpers1 = []
        if(project):
            for i in range(len(opers1)):
                M = np.kron(opers1[i], E2)
                newOpers1.append(dagger(eigVectors) @ M @ eigVectors)
        
        else:
            for i in range(len(opers1)):
                newOpers1.append(np.kron(opers1[i], E2))
        output.append(opers1)
        
    if(len(opers2) != 0):
        newOpers2 = []
        if(project):
            for i in range(len(opers2)):
                M = np.kron(E1, opers2[i])
                newOpers2.append(dagger(eigVectors) @ M @ eigVectors)
        
        else:
            for i in range(len(opers2)):
                newOpers2.append(E1, opers2[i])
        output.append(opers2)
        
    return output
    
    
# old
def MixOfTwoSys(spect1, spect2, q1, q2, opers1=np.asarray([]), opers2=np.asarray([]), 
                g=0, numOfLvls=5, project=True):
    # связываем две системы через операторы q1 и q2, попутно расширяя их операторы на общее пространство
    # opers – список из матриц операторов соотв. системы
    
    size1 = spect1.size
    size2 = spect2.size
    
    # единичная матрица 
    E1 = np.diag(np.linspace(1, 1, size1))
    E2 = np.diag(np.linspace(1, 1, size2))
    
    # диагонализованные гамильтонианы
    H1 = np.diag(spect1)
    H2 = np.diag(spect2)
    
    # объединяем линейные пространства
    H1 = np.kron(H1, E2)
    H2 = np.kron(E1, H2)    
    
    # q в общем базисе
    M = np.kron(q1, q2)
    
    # полный гамильтониан
    H = H1 + H2 + g * M
                                
    # диагонализация
    (eigEnergies, eigVectors) = eigsh(H, k=numOfLvls, which='SA', maxiter=4000)
    
    order=np.argsort(np.real(eigEnergies))
    eigEnergies=eigEnergies[order]
    eigVectors=eigVectors[:, order]
    
    if(project):
        pr = eigVectors
        H = dagger(pr) @ H @ pr
    
    # перетягиваем операторы
    if(opers1.shape[0] != 0):
        if(project):
            newOpers1 = np.zeros((opers1.shape[0], numOfLvls, numOfLvls), dtype=complex)
            for i in range(opers1.shape[0]):
                M = np.kron(opers1[i, :, :], E2)
                newOpers1[i, :, :] = dagger(pr) @ M @ pr
        
        else:
            newOpers1 = np.zeros((opers1.shape[0], size1*size2, size1*size2), dtype=complex)
            for i in range(opers1.shape[0]):
                newOpers1[i, :, :] = np.kron(opers1[i, :, :], E2)
    
    if(opers2.shape[0] != 0):
        if(project):
            newOpers2 = np.zeros((opers2.shape[0], numOfLvls, numOfLvls), dtype=complex)
            for i in range(opers2.shape[0]):
                M = np.kron(E1, opers2[i, :, :])
                newOpers2[i, :, :] = dagger(pr) @ M @ pr
        
        else:
            newOpers2 = np.zeros((opers2.shape[0], size1*size2, size1*size2), dtype=complex)
            for i in range(opers2.shape[0]):
                newOpers2[i, :, :] = np.kron(E1, opers2[i, :, :])

    # сдвигаем 0
    eigEnergies = eigEnergies - eigEnergies[0]
    
    if(opers1.shape[0] != 0 and opers2.shape[0] != 0):
        return (eigEnergies, eigVectors, H, newOpers1, newOpers2)
    elif(opers1.shape[0] != 0):
        return (eigEnergies, eigVectors, H, newOpers1)
    elif(opers2.shape[0] != 0):
        return (eigEnergies, eigVectors, H, newOpers2)
    else:
        return (eigEnergies, eigVectors, H)
        

# old
def MixOfThreeSys(spect1, spect2, spect3, q12=None, q21=None, q23=None, q32=None, q31=None, q13=None, 
                  opers1=np.asarray([]), 
                  opers2=np.asarray([]),
                  opers3=np.asarray([]),
                  g12=None, 
                  g23=None,
                  g31=None,
                  numOfLvls=10, project=True):

    
    size1 = spect1.size
    size2 = spect2.size
    size3 = spect3.size
    
    # единичная матрица 
    E1 = np.diag(np.ones(size1))
    E2 = np.diag(np.ones(size2))
    E3 = np.diag(np.ones(size3))
    
    # диагонализованные гамильтонианы
    H1 = np.diag(spect1)
    H2 = np.diag(spect2)
    H3 = np.diag(spect3)
    
    # объединяем линейные пространства
    H1 = np.kron(np.kron(H1, E2), E3)
    H2 = np.kron(np.kron(E1, H2), E3)
    H3 = np.kron(np.kron(E1, E2), H3)
    
    # полный гамильтониан
    H = H1 + H2 + H3
          
    if(g12 != None):
        M = np.kron(np.kron(q12, q21), E3)
        H = H + g12 * M
        
    if(g23 != None):
        M = np.kron(np.kron(E1, q23), q32)
        H = H + g23 * M
        
    if(g31 != None):
        M = np.kron(np.kron(q13, E2), q31)
        H = H + g31 * M
        
        
    # диагонализация
    (eigEnergies, eigVectors) = eigsh(H, k=numOfLvls, which='SA', maxiter=4000)
        
    order=np.argsort(np.real(eigEnergies))
    eigEnergies=eigEnergies[order]
    eigVectors=eigVectors[:, order]
    
    if(project):
        pr = eigVectors
        H = dagger(pr) @ H @ pr
    
    eigEnergies = eigEnergies - eigEnergies[0]
    
    corn = [eigEnergies, eigVectors, H]
    
    # перетягиваем операторы
    if(opers1.shape[0] != 0):
        if(project):
            newOpers1 = np.zeros((opers1.shape[0], numOfLvls, numOfLvls), dtype=complex)
            for i in range(opers1.shape[0]):
                M = np.kron(np.kron(opers1[i, :, :], E2), E3)
                newOpers1[i, :, :] = dagger(pr) @ M @ pr
            
        else:
            newOpers1 = np.zeros((opers1.shape[0], size1*size2*size3, size1*size2*size3), dtype=complex)
            for i in range(opers1.shape[0]):
                newOpers1[i, :, :] = np.kron(np.kron(opers1[i, :, :], E2), E3)
            
        corn.append(newOpers1)
    
    if(opers2.shape[0] != 0):
        if(project):
            newOpers2 = np.zeros((opers2.shape[0], numOfLvls, numOfLvls), dtype=complex)
            for i in range(opers2.shape[0]):
                M = np.kron(np.kron(E1, opers2[i, :, :]), E3)
                newOpers2[i, :, :] = dagger(pr) @ M @ pr
                
        else: 
            newOpers2 = np.zeros((opers2.shape[0], size1*size2*size3, size1*size2*size3), dtype=complex)
            for i in range(opers2.shape[0]):
                newOpers2[i, :, :] = np.kron(np.kron(E1, opers2[i, :, :]), E3)
            
        corn.append(newOpers2)
        
    if(opers3.shape[0] != 0):
        if(project):
            newOpers3 = np.zeros((opers3.shape[0], numOfLvls, numOfLvls), dtype=complex)
            for i in range(opers3.shape[0]):
                M = np.kron(np.kron(E1, E2), opers3[i, :, :])
                newOpers3[i, :, :] = dagger(pr) @ M @ pr
        else:        
            newOpers3 = np.zeros((opers3.shape[0], size1*size2*size3, size1*size2*size3), dtype=complex)
            for i in range(opers3.shape[0]):
                newOpers3[i, :, :] = np.kron(np.kron(E1, E2), opers3[i, :, :])
            
        corn.append(newOpers3)
        
    
    return tuple(corn)

# new
def mix_three_sys(spect1, spect2, spect3, q12=None, q21=None, q23=None, q32=None, q31=None, q13=None, 
                  opers1=[], 
                  opers2=[],
                  opers3=[],
                  g12=None, 
                  g23=None,
                  g31=None,
                  numOfLvls=10, purity_calc=True, stList=True, dirtyBorder=0.01,
                  eigVectors_output=False, project=True):

    
    dim_1 = spect1.shape[0]
    dim_2 = spect2.shape[0]
    dim_3 = spect3.shape[0]
    
    # единичная матрица 
    E1 = np.diag(np.ones(dim_1))
    E2 = np.diag(np.ones(dim_2))
    E3 = np.diag(np.ones(dim_3))
    
    # диагонализованные гамильтонианы
    H1 = np.diag(spect1)
    H2 = np.diag(spect2)
    H3 = np.diag(spect3)
    
    # объединяем линейные пространства
    H1 = np.kron(np.kron(H1, E2), E3)
    H2 = np.kron(np.kron(E1, H2), E3)
    H3 = np.kron(np.kron(E1, E2), H3)
    
    # полный гамильтониан
    H = H1 + H2 + H3
          
    if(g12 != None):
        M = np.kron(np.kron(q12, q21), E3)
        H = H + g12 * M
        
    if(g23 != None):
        M = np.kron(np.kron(E1, q23), q32)
        H = H + g23 * M
        
    if(g31 != None):
        M = np.kron(np.kron(q13, E2), q31)
        H = H + g31 * M
        
        
    # диагонализация
    (eigEnergies, eigVectors) = eigsh(H, k=numOfLvls, which='SA', maxiter=4000)
        
    order=np.argsort(np.real(eigEnergies))
    eigEnergies=eigEnergies[order]
    eigVectors=eigVectors[:, order]
    
    eigEnergies = eigEnergies - eigEnergies[0]
    
    output = [eigEnergies]
    
    if(eigVectors_output): output.append(eigVectors)
    
    if(purity_calc):
        purity_info = StatesPurity(eigVectors, (dim_1, dim_2, dim_3), stList=stList, dirtyBorder=dirtyBorder)
        output.append(purity_info)
        
    # перетягиваем операторы
    if(len(opers1) != 0):
        newOpers1 = []
        if(project):
            for i in range(len(opers1)):
                M = kron(opers1[i], E2, E3)
                newOpers1.append(dagger(eigVectors) @ M @ eigVectors)
            
        else:
            for i in range(len(opers1)):
                newOpers1.append(kron(opers1[i], E2, E3))
            
        output.append(newOpers1)
    
    if(len(opers2) != 0):
        newOpers2 = []
        if(project):
            for i in range(len(opers2)):
                M = kron(E1, opers2[i], E3)
                newOpers2.append(dagger(eigVectors) @ M @ eigVectors)
                
        else: 
            for i in range(len(opers2)):
                newOpers2.append(kron(E1, opers2[i], E3))
            
        output.append(newOpers2)
        
    if(len(opers3) != 0):
        newOpers3 = []
        if(project):
            for i in range(len(opers3)):
                M = kron(E1, E2, opers3[i])
                newOpers3.append(dagger(eigVectors) @ M @ eigVectors)
        else:        
            for i in range(len(opers3)):
                newOpers3.append(kron(E1, E2, opers3[i]))
            
        output.append(newOpers3)
        
    
    return output


def Graphs(t, X, x='x', y='y', full=False, save=False, filename='', xborders=None, yborders=None, lsize=10, grid=True, subGrid=True):
    
    if(xborders!=None):
        plt.xlim(xborders)
    
    if(yborders!=None):
        plt.ylim(yborders)
        
    plt.rcParams["figure.figsize"] = (10, 10)

    if(len(X.shape) == 2):
        for n in range(np.shape(X)[0]):
            lbl = str(n)
            plot = plt.plot(t, X[n, :], lw=1.5, label=lbl)
        plt.legend(loc='center left', bbox_to_anchor=(1.01, 0.5))
    else:
        plot = plt.plot(t, X[:], lw=1.5)
        

    if (full):
        Xf = np.zeros(np.shape(X)[1])
        for n in range(np.shape(X)[0]):
            for m in range(np.shape(X)[1]):
                Xf[m] += X[n, m]

        plot = plt.plot(t, Xf, color="gray", lw=1.5, label=lbl)

    if(grid):
        # врубаем сетку
        plt.minorticks_on()

        # Определяем внешний вид линий основной сетки:
        plt.grid(which='major',
                 color='k',
                 linewidth=0.5)
            
        if(subGrid):
            # Определяем внешний вид линий вспомогательной
            # сетки:
            plt.grid(which='minor',
                     color='k',
                     linestyle=':')

    plt.rcParams['font.size'] = str(lsize)
    
    plt.xlabel(x)
    plt.ylabel(y)

    if (save):
        plt.savefig(filename, facecolor='white')

    plt.show()
    plt.close()


def PlotPcolormesh(fidelity, x, y, xlabel = 'x', ylabel = 'y', opt_lines=True, 
                    title=None, save=False, filename=''):
    fig, axs = plt.subplots(nrows = 1, ncols = 1,figsize = (10, 10))
    
    xGrid, yGrid = np.meshgrid(x, y)
    cmap_set = 'PiYG'
    cb = axs.pcolormesh(xGrid, yGrid, np.transpose(fidelity[:, :]), cmap = cmap_set)
    axs.set_xlabel(xlabel)
    axs.set_ylabel(ylabel)
    fig.colorbar(cb, ax=axs)

    opt_x_ind = np.argmax(np.real(fidelity))//fidelity.shape[1]
    opt_y_ind = np.argmax(np.real(fidelity))%fidelity.shape[1]
    
    
    axs.text(x[0], y[len(y)-1] + (y[len(y)-1] - y[0])*0.05, 
             'opt ' + ylabel + ' = ' + str(y[opt_y_ind]) + ' with index ' + str(opt_y_ind))
    axs.text(x[0], y[len(y)-1] + (y[len(y)-1] - y[0])*0.08, 
             'opt ' + xlabel + ' = ' + str(x[opt_x_ind]) + ' with index ' + str(opt_x_ind))
    axs.text(x[0], y[len(y)-1] + (y[len(y)-1] - y[0])*0.11, 
             'max fidelity = ' + str(np.abs(fidelity[opt_x_ind, opt_y_ind])))
    
    if opt_lines:
        axs.hlines(y[opt_y_ind], x[0], x[-1])
        axs.vlines(x[opt_x_ind], y[0], y[-1])
    if title != None:
        plt.title(title)

    if(save):
        plt.savefig(filename, facecolor = 'white')    
        
    plt.show()


def OperLin(A, B):
    # при линеаризации матрицы Ro по строкам в вектор ro преобразует A@Ro@B -> M@ro
    size = A.shape[0]
    M = np.zeros((size**2, size**2), dtype=complex) 
    for a in range(size):
        for b in range(size):
            for c in range(size):
                for d in range(size):
                    M[a*size + b, d + c*size] += A[a, c]*B[d, b]
                    
    return M


def PsiToRo(psi):
    # принимает psi - столбец!
    ro = psi@np.conjugate(psi.transpose())
    
    return ro


def LindbladLin(H, L):
    # H – гамильтониан
    # возарвщвет эффективную матрицу M ур. Л. для лианеризованной по строкам ro
    # L – массив операторов Линдблада
    size = H.shape[0]
    M = np.zeros((size**2, size**2), dtype=complex)
    
    # единичная матрица
    E = np.diag(np.linspace(1, 1, size))
    
    # бездиссипативная часть
    M += -1j*(OperLin(H, E) - OperLin(E, H))
    
    # диссипативная часть
    for n in range(L.shape[0]):
        M += OperLin(L[n], dagger(L[n])) -\
        1/2 * (OperLin(dagger(L[n])@L[n], E) +\
               OperLin(E, dagger(L[n])@L[n]))
        
    return M


def UnitMDecomposer(U):
    # на вход принимаем np.array с входной унитарной матрицей
    dim = U.shape[0]
    
    # создадим кортеж матриц разложения
    decomp = []
    
    
    # пройдемся по подматрицам nxn от n=dim до n = 2
    for i in range(dim):
        # проход по столбцу
        for j in range(dim - 1 - i):
            # шаблон
            Ustep = np.linspace(1, 1, dim)
            Ustep = np.diag(Ustep)
            Ustep = np.asarray(Ustep, dtype=complex)
            
            a = U[i, i]
            b = U[i + j + 1, i]
            
            if((b == 0) and (j != dim - 2 - i)):
                continue
                
            elif((b == 0) and (a != 1)):
                # последний элемент столбца обрабатываем иначе
                Ustep[i, i] = np.conj(a)
                U = Ustep@U
                decomp.append(np.conj(Ustep.transpose()))
                
            else:
                
                c = np.sqrt(np.conj(a)*a + np.conj(b)*b)
                
                Ustep[i, i] = np.conj(a)/c
                Ustep[i + j + 1, i + j + 1] = -a/c
                Ustep[i + j + 1, i] = b/c
                Ustep[i, i + j + 1] = np.conj(b)/c
                
                U = Ustep@U
                decomp.append(np.conj(Ustep.transpose()))
                
    # последний элемент в правом нижнем углу
    if(U[dim - 1, dim - 1] != 1):
        Ustep = np.linspace(1, 1, dim)
        Ustep = np.diag(Ustep)
        Ustep = np.asarray(Ustep, dtype=complex)
        Ustep[dim - 1, dim - 1] = np.conj(U[dim - 1, dim - 1])
        U = Ustep@U
        decomp.append(np.conj(Ustep.transpose()))
        
    
    # тест на корректность (проверяемую точночть можно корректировать)
    for n in range(dim):
        if(abs(U[n, n] - 1) > 10**-8):
            print("ERROR")
            return
    
    return decomp
    
    
def FluxoniumFitter(specDots, borders, 
                    weights=np.asarray(None), zeal=40, cutError=None, 
                    FlGrid=400, permission = 10, onAir=False, fixEc=None):
    
# specDots = ([потоки относительно полкванта], 
#             [номер потоковой точки, номер уровня начиная с 1, энергия в ГГц])
    
    Lvls = specDots[1].shape[1]
    if(weights.any()==None):
        weights=np.ones(specDots[1].shape)
    
    if(fixEc==None):
        def loss(optParams):

            optEj = optParams[0]
            optEl = optParams[1]
            optEc = optParams[2]

            error = 0

            for i in range(specDots[0].shape[0]):
                flux = specDots[0][i]

                spectrum, _, _ = Fluxonium(Ej=optEj, El=optEl, Ec=optEc, 
                                               gridSize=FlGrid, numOfLvls=Lvls+1, F=flux)

                for j in range(Lvls):
                    for k in range(specDots[1].shape[2]):

                        if(specDots[1][i, j, k] != 0):
                            error += weights[i, j, k]*(10000*(specDots[1][i, j, k] - spectrum[j + 1]))**2

            return error

        lossVal=1000000
        ans = borders[:, 0]

        if(cutError==None):
            for i in tqdm.tqdm(range(zeal)):

                # генерируем x0
                x0 = np.random.rand(3) * (borders[:, 1] - borders[:, 0]) + borders[:, 0]
                # оптимизируем
                sol = minimize(loss, x0=x0, bounds=borders)

                if(sol.success != True):
                    continue

                if(sol.fun < lossVal):
                    limbo = (ans[0] - sol.x[0])**2 + (ans[1] - sol.x[1])**2 + (ans[2] - sol.x[2])**2
                    if(limbo > permission*(sol.fun - lossVal)**2):
                        print("ALARM, lack of dots or weights!")

                    lossVal = sol.fun
                    ans = sol.x

                    if(onAir):
                        print(ans, "\n", lossVal, "\n")

        else:
            while(lossVal > cutError):

                # генерируем x0
                x0 = np.random.rand(3) * (borders[:, 1] - borders[:, 0]) + borders[:, 0]
                # оптимизируем
                sol = minimize(loss, x0=x0, bounds=borders)

                if(sol.success != True):
                    continue

                if(sol.fun < lossVal):
                    limbo = (ans[0] - sol.x[0])**2 + (ans[1] - sol.x[1])**2 + (ans[2] - sol.x[2])**2
                    if(limbo > permission*(sol.fun - lossVal)**2):
                        print("ALARM, lack of dots or weights!")

                    lossVal = sol.fun
                    ans = sol.x

                    if(onAir):
                        print(ans, "\n", lossVal, "\n")
    
    else:
        def loss(optParams):

            optEj = optParams[0]
            optEl = optParams[1]
            optEc = fixEc

            error = 0

            for i in range(specDots[0].shape[0]):
                flux = specDots[0][i]

                spectrum, _, _ = Fluxonium(Ej=optEj, El=optEl, Ec=optEc, 
                                               gridSize=FlGrid, numOfLvls=Lvls+1, F=flux)

                for j in range(Lvls):
                    for k in range(specDots[1].shape[2]):

                        if(specDots[1][i, j, k] != 0):
                            error += weights[i, j, k]*(10000*(specDots[1][i, j, k] - spectrum[j + 1]))**2

            return error

        lossVal=1000000
        ans = borders[:, 0]

        if(cutError==None):
            for i in tqdm.tqdm(range(zeal)):

                # генерируем x0
                x0 = np.random.rand(2) * (borders[:, 1] - borders[:, 0]) + borders[:, 0]
                # оптимизируем
                sol = minimize(loss, x0=x0, bounds=borders)

                if(sol.success != True):
                    continue

                if(sol.fun < lossVal):
                    limbo = (ans[0] - sol.x[0])**2 + (ans[1] - sol.x[1])**2
                    if(limbo > permission*(sol.fun - lossVal)**2):
                        print("ALARM, lack of dots or weights!")

                    lossVal = sol.fun
                    ans = sol.x

                    if(onAir):
                        print(ans, "\n", lossVal, "\n")

        else:
            while(lossVal > cutError):

                # генерируем x0
                x0 = np.random.rand(2) * (borders[:, 1] - borders[:, 0]) + borders[:, 0]
                # оптимизируем
                sol = minimize(loss, x0=x0, bounds=borders)

                if(sol.success != True):
                    continue

                if(sol.fun < lossVal):
                    limbo = (ans[0] - sol.x[0])**2 + (ans[1] - sol.x[1])**2
                    if(limbo > permission*(sol.fun - lossVal)**2):
                        print("ALARM, lack of dots or weights!")

                    lossVal = sol.fun
                    ans = sol.x

                    if(onAir):
                        print(ans, "\n", lossVal, "\n")
                
    
    return ans


def pseudoosc_amplitude_decay_operator(T, E):
    # create a jump operator for N lvl system in similar way with annihilation oscillator operator
    # E – spectrum, T – decay time in ns
    size = E.shape[0]
    a = np.zeros((size, size), dtype=complex)
    
    for n in range(size - 1):
        a[n, n + 1] = np.sqrt(E[n + 1]/E[1])
        
    a = 1/np.sqrt(2*np.pi*T) * a
    
    return a


#coolwarm, magma, YlGnBu, winter

from matplotlib.ticker import MaxNLocator
def PlotContourf(fidelity, x, y, xlabel = 'x', ylabel = 'y', opt_lines=True, 
                    title=None, save=False, filename='', lsize=10):
    fig, axs = plt.subplots(nrows = 1, ncols = 1,figsize = (10, 10))
    
    plt.rcParams['font.size'] = str(lsize)
    xGrid, yGrid = np.meshgrid(x, y)
    cmap_set = 'coolwarm'
    levels = MaxNLocator(nbins=100).tick_values(0, 1)
    cb = axs.contourf(xGrid, yGrid, np.transpose(fidelity[:, :]), levels=levels, cmap = cmap_set)
    axs.set_xlabel(xlabel)
    axs.set_ylabel(ylabel)
    fig.colorbar(cb, ax=axs)

    opt_x_ind = np.argmax(np.real(fidelity))//fidelity.shape[1]
    opt_y_ind = np.argmax(np.real(fidelity))%fidelity.shape[1]
    
    
    axs.text(x[0], y[len(y)-1] + (y[len(y)-1] - y[0])*0.10, 
             'opt ' + ylabel + ' = ' + str(np.around(y[opt_y_ind], 2)))
    axs.text(x[0], y[len(y)-1] + (y[len(y)-1] - y[0])*0.16, 
             'opt ' + xlabel + ' = ' + str(np.around(x[opt_x_ind], 2)))
    axs.text(x[0], y[len(y)-1] + (y[len(y)-1] - y[0])*0.22, 
             'max fidelity = ' + str(np.around(np.abs(fidelity[opt_x_ind, opt_y_ind]), 5)))
    
    if opt_lines:
        axs.hlines(y[opt_y_ind], x[0], x[-1], color='k')
        axs.vlines(x[opt_x_ind], y[0], y[-1], color='k')
    if title != None:
        plt.title(title)

    if(save):
        plt.savefig(filename, facecolor = 'white')    
        
    plt.show()

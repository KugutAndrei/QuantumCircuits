import tensorflow as tf
import tqdm
from numpy import pi,linspace,tensordot
import scipy.optimize
import numpy as np
import copy
import matplotlib.pyplot as plt
from scipy.optimize import root
from scipy.interpolate import interp1d
from scipy.sparse.linalg import eigsh
from scipy.integrate import solve_ivp
from scipy import interpolate
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator
from matplotlib.ticker import FuncFormatter
from scipy.optimize import minimize

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


def Oscillator(omega, numOfLevels=100):
    # собственные значения энергии
    eigEnergies = np.linspace(0, omega * (numOfLevels - 1), numOfLevels)

    # оператор уничтожения
    a = np.zeros((numOfLevels, numOfLevels), dtype=complex)
    for n in range(numOfLevels - 1):
        a[n, n + 1] = np.sqrt(n + 1)

    # оператор рождения
    at = np.zeros((numOfLevels, numOfLevels), dtype=complex)
    for n in range(numOfLevels - 1):
        at[n + 1, n] = np.sqrt(n + 1)

    return (eigEnergies, at, a)


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


def Fluxonium(Ej, El, Ec, gridSize=100, numOfLvls=100, leftBorder=-20, rightBorder=20, F=0):
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

    return (eigEnergies, phi, q)


def Transmon(Ej1, Ej2, Ec, gridSize=100, numOfLvls=100, F=0, Q=0):
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

    return (eigEnergies, phi, q)


def MixOfTwoSys(spect1, spect2, q1, q2, opers1=np.asarray([]), opers2=np.asarray([]), 
                g=0, numOfLvls=5, project=False):
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
    q1 = np.kron(q1, E2)
    q2 = np.kron(E1, q2)
    
    # полный гамильтониан
    H = H1 + H2 + g * q1@q2
                                
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
    
    if(opers1.shape[0] != 0 and opers2.shape[0] != 0):
        return (eigEnergies, eigVectors, H, newOpers1, newOpers2)
    elif(opers1.shape[0] != 0):
        return (eigEnergies, eigVectors, H, newOpers1)
    elif(opers2.shape[0] != 0):
        return (eigEnergies, eigVectors, H, newOpers2)
    else:
        return (eigEnergies, eigVectors, H)



def Graphs(t, X, x='x', y='y', full=False, save=False, filename=''):
    plt.rcParams["figure.figsize"] = (10, 10)

    for n in range(np.shape(X)[0]):
        lbl = str(n)
        plot = plt.plot(t, X[n, :], lw=1.5, label=lbl)

    plt.legend(loc='center left', bbox_to_anchor=(1.01, 0.5))

    if (full):
        Xf = np.zeros(np.shape(X)[1])
        for n in range(np.shape(X)[0]):
            for m in range(np.shape(X)[1]):
                Xf[m] += X[n, m]

        plot = plt.plot(t, Xf, color="gray", lw=1.5, label=lbl)

    # врубаем сетку
    plt.minorticks_on()

    # Определяем внешний вид линий основной сетки:
    plt.grid(which='major',
             color='k',
             linewidth=0.5)

    # Определяем внешний вид линий вспомогательной
    # сетки:
    plt.grid(which='minor',
             color='k',
             linestyle=':')

    plt.xlabel(x)
    plt.ylabel(y)

    if (save):
        plt.savefig(filename, facecolor='white')

    plt.show()


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


"""
from VirtualQubitSystem import *

def SchrodingerParamEx(H, V, psiIn, psiTarg, timelist, force, psi_flag=True):
    #решаем Шредингера с гамильтонианом H + V*F(t, a), где a – вектор параметров
    # psiIn, psiTarg – волновые функции-столбцы

    # прописывем матрички в модель
    energies = []
    Phi = []

    energies.append(tf.constant(H, dtype=tf.complex128))
    Phi.append(tf.constant(V, dtype=tf.complex128))
    size = H.shape[0]
 
    sys_args = {
        'nQbt'     : 1,
        'nLvl'     : [H.shape[0]],
        'energies' : energies,
        'N'        : [0],
        'g'        : [],
        't_dep'    : [], 
    }

    model = VirtualQubitSystem(sys_args)


    # preparing for calculating time-dependent H

    def calc_fluxoniumdriveH(phi):
        Drive = np.asarray(phi)
        H_td = Drive[:,tf.newaxis,tf.newaxis]*Phi[0][tf.newaxis]
        return model.H[0][tf.newaxis, :, :] + H_td


    model.set_calc_timedepH(calc_fluxoniumdriveH)


    # вписываем волновые функции-столбцы

    psiIn = tf.convert_to_tensor(psiIn, dtype=tf.complex128)
    psiTarg = tf.convert_to_tensor(psiTarg, dtype=tf.complex128)

    initstate = tf.stack([tf.reshape(psiIn, (size, 1))],
                        axis = 2)

    targetstate = tf.stack([tf.reshape(psiTarg, (size, 1))],
                        axis = 2)


    initstate = tf.reshape(initstate, (initstate.shape[0], initstate.shape[2]))
    targetstate = tf.reshape(targetstate, (targetstate.shape[0], targetstate.shape[2]))

    model.set_initstate(initstate)
    model.set_targetstate(targetstate)

    # сетка t
    model.set_timelist(timelist)

    
    запускаем дифурорешатель, который выдает:

    1) psilist[A, B, C, D]
        A - индекс вектора параметров сигнала
        B – индекс узла сетки t
        C – индекс проекции в волновой вектор-функции
        D – индекс заданных initstate

    2) fid[A, B, C]
        A - индекс вектора параметров сигнала
        B – индекс узла сетки t
        C – индекс заданных initstate

    

    if(psi_flag):
        # решение с выводом полной волновой функции на сетке t (psilist)
        _, psilist = model.scan_fidelitySE(force, psi_flag=psi_flag, progress_bar=True)
        psilist = psilist.numpy()
        return psilist

    else:
        # выдает только проекцию на заданное targetstate на сетке t (fid)
        fid = model.scan_fidelitySE(calc_Phi3D, psi_flag=False, progress_bar=True)
        fid = fid.numpy()
        return fid


def LindbladParamEx(H, V, L, psiIn, timelist, force):

    # L - списорк операторов линдблада
    energies = []
    Phi = []

    # строим эффективнуя матрицу дифура H + V*F(t) из уравнения Линдблада
    size = H.shape[0]

    # лианирезуем ro по строкам
    if(L.ndim == 2):
        Hl= LindbladLin(H, np.asarray([L]))
    else: 
        Hl= LindbladLin(H, L)

    Vl = LindbladLin(V, np.asarray([]))


    # костыль перехода от Шредингера к обычной СЛДУ
    Hl = 1j*Hl
    Vl = 1j*Vl
    psiTarg = np.zeros((size, 1), dtype=complex)

    energies.append(tf.constant(Hl, dtype=tf.complex128))
    Phi.append(tf.constant(Vl, dtype=tf.complex128))


    sys_args = {
        'nQbt'     : 1,
        'nLvl'     : [Hl.shape[0]],
        'energies' : energies,
        'N'        : [0],
        'g'        : [],
        't_dep'    : [],
    }

    model = VirtualQubitSystem(sys_args)


    # preparing for calculating time-dependent H

    def calc_fluxoniumdriveH(phi):
        Drive = np.asarray(phi)
        H_td = Drive[:,tf.newaxis,tf.newaxis]*Phi[0][tf.newaxis]
        return model.H[0][tf.newaxis, :, :] + H_td


    model.set_calc_timedepH(calc_fluxoniumdriveH)


    # преобразуем в матрицы плотности и сразу растягиваем их в строки
    RoIn = PsiToRo(psiIn)
    roIn = RoIn.reshape(RoIn.shape[0]**2)
    roIn = tf.convert_to_tensor(roIn, dtype=tf.complex128)

    RoTarg = PsiToRo(psiIn)
    roTarg = RoTarg.reshape(RoTarg.shape[0]**2)
    roTarg = tf.convert_to_tensor(roTarg, dtype=tf.complex128)

    initstate = tf.stack([tf.reshape(roIn, (size**2, 1))],
                        axis = 2)

    targetstate = tf.stack([tf.reshape(roTarg, (size**2, 1))],
                        axis = 2)


    initstate = tf.reshape(initstate, (initstate.shape[0], initstate.shape[2]))
    targetstate = tf.reshape(targetstate, (targetstate.shape[0], targetstate.shape[2]))

    model.set_initstate(initstate)
    model.set_targetstate(targetstate)

    # сетка t
    model.set_timelist(timelist)

    
    запускаем дифурорешатель, который выдает:

    1) psilist[A, B, C, D]
        A - индекс вектора параметров сигнала
        B – индекс узла сетки t
        C – индекс вектор-функции
        D – индекс заданных initstate
    

    # решение с выводом полной волновой функции на сетке t (psilist)
    _, rolist = model.scan_fidelitySE(force, psi_flag=True, progress_bar=True)
    rolist = rolist.numpy()
    rolistM = rolist.reshape(rolist.shape[0], 
                             rolist.shape[1],
                             size, size,
                             rolist.shape[3])
    
    return rolistM

"""
        

def MixOfThreeSys(spect1, spect2, spect3, q12=None, q21=None, q23=None, q32=None, q31=None, q13=None, 
                  opers1=np.asarray([]), 
                  opers2=np.asarray([]),
                  opers3=np.asarray([]),
                  g12=None, 
                  g23=None,
                  g31=None,
                  numOfLvls=10, project=False):

    
    size1 = spect1.size
    size2 = spect2.size
    size3 = spect3.size
    
    # единичная матрица 
    E1 = np.diag(np.linspace(1, 1, size1))
    E2 = np.diag(np.linspace(1, 1, size2))
    E3 = np.diag(np.linspace(1, 1, size3))
    
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
        q12 = np.kron(np.kron(q12, E2), E3)
        q21 = np.kron(np.kron(E1, q21), E3)
        H = H + g12 * q12@q21
        
    if(g23 != None):
        q23 = np.kron(np.kron(E1, q23), E3)
        q32 = np.kron(np.kron(E1, E2), q32)
        H = H + g23 * q23@q32
        
    if(g31 != None):
        q13 = np.kron(np.kron(q13, E2), E3)
        q31 = np.kron(np.kron(E1, E2), q31)
        H = H + g31 * q13@q31
        
        
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

def ReverseQuantization(Elin, Ecin, S=np.asarray([])):
    # El – строка длины n (кол-во степеней свободы) с индуктивными энергиями каждой подсистемы
    # Ec – матрица nxn с емкостными энергиями каждой подсистемы и связями между ними
    # S – матрица перехода от реальных потоков к модельным
    
    El = np.copy(Elin)
    Ec = np.copy(Ecin)
    
    n = Ec.shape[0]
    
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
    
    # переведем C в удобный для чтения вид (вычтем C связей из диагонали)
    
    
    for i in range(n):
        
        # также вернем L
        if(InvL[i, i] != 0):
            L[i] = 1/InvL[i, i] 
        else:
            L[i] = 99999999
        
        for j in range(n):
            if(i > j):
                C[i, j] = 0
                
            elif(i < j and C[i, j]!=0):
                C[i, j] = -C[i, j]
                
                # чистим диагональ
                C[i, i] = C[i, i] - abs(C[i, j])
                C[j, j] = C[j, j] - abs(C[i, j])
    
    # на выходе емкости в fF и индуктивности в nH
    
    return(L, C)


def ForwardQuantization(Lin, Cin, S=np.asarray([])):
    # C – матрица nxn с емкостями
    # S – матрица перехода от реальных потоков к модельным
    
    L = np.copy(Lin)
    C = np.copy(Cin)
    
    n = L.shape[0]
    InvL = np.zeros((n, n))
    # заполним нормльно матрицы
    for i in range(n):
        InvL[i, i] = 1/L[i]
        for j in range(n):
            if(i < j and C[i, j]!=0):
                
                
                C[j, i] = C[i, j] = - C[i, j]
                
                C[i, i] = C[i, i] + abs(C[i, j])
                C[j, j] = C[j, j] + abs(C[i, j])
    
    
    # переходим к модельной цепи заменами
    if(S.shape[0] != 0):
        St = np.copy(S)
        St = np.linalg.inv(St)
        
        C = np.transpose(St) @ C @ St
        InvL = np.transpose(St) @ InvL @ St
        
    # находим матрицу энергий
    CInv = np.linalg.inv(C)
    
    Ec = np.zeros((n, n))
    El = np.zeros(n)
    
    # учтем, что энергия в GHz, а емкость в fF
    for i in range(n):
        El[i] = InvL[i, i] * Fq /16/np.pi**2/e
        
        for j in range(n):
            if(j == i):
                Ec[i, i] = e/Fq * CInv[i, i] * 10**6
            elif(j > i):
                Ec[i, j] = Ec[j, i] = 2*e/Fq * CInv[i, j] * 10**6
                Ec[j, i] = 0
    
    
    return(El, Ec)


def PhysOptReverseQuantization(El, Ec0, S, deltaEcMax, weightС, zeal=10):
    # энергии в MГц!!!, a С в фФ
    # weightС - матрица с весами зануления емкостей
    size = Ec0.shape[0]
    indexSpace = []
    valueSpace = []
    
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
            
        _, C = tul.ReverseQuantization(El/1000, Ec/1000, S=S)
        
        answ = 0
        
        for n in range(size):
            for m in range(size - n):
                answ += weightC[n, n + m] * C[n, n + m]**2
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
        
    _, C = tul.ReverseQuantization(El/1000, finalAns/1000, S=S)
    
    return(finalAns, C)


def PhysOptForwardQuantization(L, C0, S, deltaCMax, weightEc, zeal=10, method=0):
    # энергии в MГц!!!, a С в фФ
    # weightС - матрица с весами зануления емкостей
    size = C0.shape[0]
    indexSpace = []
    valueSpace = []
    
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
            
        _, Ec = tul.ForwardQuantization(L, C, S=S)
        
        Ec = Ec*1000
        answ = 0
        
        for n in range(size):
            for m in range(size - n):
                answ += weightEc[n, n + m] * Ec[n, n + m]**2
        
        return answ
        
    # теперь устроим оптимизацию с рандомными начальными точками и выберем лучшее
    ans = np.zeros(dim)
    lossVal = loss(ans)
    print(lossVal, '\n')
    
    for sample in range(zeal):

        # генерируем x0
        x0 = np.random.rand(dim) * 2*np.asarray(valueSpace) - np.asarray(valueSpace)
        # оптимизируем
        if(method == 0):
            sol = minimize(loss, x0=x0, bounds=bounds, method='Nelder-Mead')
        elif(method == 1):
            sol = minimize(loss, x0=x0, bounds=bounds, method='Powell')
        elif(method == 2):
            sol = minimize(loss, x0=x0, bounds=bounds, method='CG')
        elif(method == 3):
            sol = minimize(loss, x0=x0, bounds=bounds, method='BFGS')
        elif(method == 4):
            sol = minimize(loss, x0=x0, bounds=bounds, method='Newton-CG')
        elif(method == 5):
            sol = minimize(loss, x0=x0, bounds=bounds, method='L-BFGS-B')
        elif(method == 6):
            sol = minimize(loss, x0=x0, bounds=bounds, method='TNC')
        elif(method == 7):
            sol = minimize(loss, x0=x0, bounds=bounds, method='COBYLA')
        elif(method == 8):
            sol = minimize(loss, x0=x0, bounds=bounds, method='SLSQP')
        elif(method == 9):
            sol = minimize(loss, x0=x0, bounds=bounds, method='trust-constr')
        elif(method == 10):
            sol = minimize(loss, x0=x0, bounds=bounds, method='dogleg')
        elif(method == 11):
            sol = minimize(loss, x0=x0, bounds=bounds, method='trust-ncg')
        elif(method == 12):
            sol = minimize(loss, x0=x0, bounds=bounds, method='trust-krylov')
        elif(method == 13):
            sol = minimize(loss, x0=x0, bounds=bounds, method='trust-exact')
        
        
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
        
    _, Ec = tul.ForwardQuantization(L, finalAns, S=S)
    
    return(finalAns, 1000*Ec)


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
            s = abs(mixStates[:, n])
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
                for k in range(key.shape[2]):
                    if(n + m + k != 0 and key[n, m, k] == 0):
                        key[n, m, k] = None
        
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
            s = abs(mixStates[:, n])
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
    
    

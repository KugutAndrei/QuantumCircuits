import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from tqdm import tqdm
from copy import copy




mytensordot = lambda A, B: tf.reshape(tf.transpose(tf.tensordot(A, B, axes=0), (0, 2, 1, 3)),\
                                      (A.shape[0]*B.shape[0], A.shape[1]*B.shape[1]))




def plot_pcolormesh(f, x, y, xlabel = '', ylabel = '', opt_lines=True, title=None):
    fig, axs = plt.subplots(nrows = 1, ncols = 1, figsize = (5, 4))
    xGrid, yGrid = np.meshgrid(x, y)
    cmap_set = 'RdBu'
    cb = axs.pcolormesh(xGrid, yGrid, f[:,:], cmap = cmap_set)
    axs.set_xlabel(xlabel)
    axs.set_ylabel(ylabel)
    fig.colorbar(cb, ax=axs)

    opt_y_ind = np.argmax(np.real(f))//f.shape[1]
    opt_x_ind = np.argmax(np.real(f))%f.shape[1]
    print('opt max '+xlabel, float(x[opt_x_ind]), 'index', opt_x_ind)
    print('opt max '+ylabel, y[opt_y_ind], 'index', opt_y_ind)
    print('function max ',np.abs(f[opt_y_ind, opt_x_ind]))

    if opt_lines:
        axs.hlines(y[opt_y_ind], x[0], x[-1])
        axs.vlines(x[opt_x_ind], y[0], y[-1])

    opt_y_ind = np.argmin(np.real(f)) // f.shape[1]
    opt_x_ind = np.argmin(np.real(f)) % f.shape[1]
    print('opt min ' + xlabel, float(x[opt_x_ind]), 'index', opt_x_ind)
    print('opt min ' + ylabel, y[opt_y_ind], 'index', opt_y_ind)
    print('function min ', np.abs(f[opt_y_ind, opt_x_ind]))

    if opt_lines:
        axs.hlines(y[opt_y_ind], x[0], x[-1])
        axs.vlines(x[opt_x_ind], y[0], y[-1])
    if title != None:
        plt.title(title)
    plt.show() 




def mytensordotlist(l):
    res = l[0]
    if len(l) > 1:
        res = l[0]
        for i in range(1, len(l)):
            res = mytensordot(res, l[i])
    return res




class VirtQ:
    '''
    Class VirtQ for simulation of quantum system with fluxonium qubits.
    It is a new version of VirtualQubitSystem

    Params
    ------
    calc_timedepH (function)   :    function that returns the Hamiltonian as a function of some parameter(s),
                                    function of the pararmeter(s) calls in functions scan_fidelitySE and
                                    scan_fidelityME as calc_H_as_time_function(t):
                                    calc_timedepH(calc_H_as_time_function(t))
    '''
    def __init__(self, calc_timedepH):
        self.calc_timedepH = calc_timedepH
        self.Lindblad_operators = []
            
            

                    
    def set_timelist(self, timelist):
        self.timelist = timelist
        
    
    
    
    def set_initstate(self, initstate):
        '''
        Set initstate

        Params
        ------
            initstate (tensor): initial state (psi fuction)
        '''
        self.initstate = initstate
        if initstate.shape[1] == 1:
            self.initrho = initstate @ tf.linalg.adjoint(initstate)
    
    
    
    
    def set_targetstate(self, targetstate):
        '''
        Set targetstate

        Params
        ------
            targetstate (tensor): target state (psi fuction)
        '''
        self.targetstate = targetstate
        self.targetstateadjoint = tf.linalg.adjoint(targetstate)
    
    
    
    
    def calc_fidelity_psi(self, psi):
        return tf.reshape(tf.linalg.diag_part(self.targetstateadjoint @ psi),\
                          (psi.shape[0], psi.shape[2])) 




    def calc_fidelity_rho(self, rho):
        # print(tf.sqrt(tf.math.abs(self.targetstateadjoint@rho@self.targetstate)))
        return tf.reshape(tf.sqrt(tf.math.abs(self.targetstateadjoint@rho@self.targetstate)),\
                          (rho.shape[0], 1))




    def __solveSE_expm(self, psi, H, dt):
        return tf.linalg.expm(-1j*dt*H)@psi

    def __Schrodinger_step(self, psi, t):
        H = self.calc_timedepH(self.calc_H_as_time_function(t), t)
        return -1j * H @ psi

    def __solveSE_RK4(self, psi, t, dt):
        k1 = self.__Schrodinger_step(psi, t)
        k2 = self.__Schrodinger_step(psi + dt * k1 / 2, t + dt / 2)
        k3 = self.__Schrodinger_step(psi + dt * k2 / 2, t + dt / 2)
        k4 = self.__Schrodinger_step(psi + dt * k3, t + dt)
        return psi + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    
    
    
    def set_Lindblad_operators(self, Lindblad_operators):
        self.Lindblad_operators = Lindblad_operators


    def __Lindblad(self, rho, t):
        H = self.calc_timedepH(self.calc_H_as_time_function(t), t)
        res = -1j*(H@rho - rho@H)
        for c in self.Lindblad_operators:
            cadj = tf.linalg.adjoint(c)
            res += c[tf.newaxis, :, :]@rho@cadj[tf.newaxis, :, :] - 0.5*\
                   (cadj[tf.newaxis, :, :]@c[tf.newaxis, :, :]@rho +\
                    rho@cadj[tf.newaxis, :, :]@c[tf.newaxis, :, :])
        return res



    def __solveME(self, rho, t, dt):
        k1 = self.__Lindblad(rho, t)
        k2 = self.__Lindblad(rho+dt*k1/2, t+dt/2)
        k3 = self.__Lindblad(rho+dt*k2/2, t+dt/2)
        k4 = self.__Lindblad(rho+dt*k3, t+dt)
        return rho + dt/6*(k1+2*k2+2*k3+k4)



    
    
    def scan_fidelitySE(self, calc_H_as_time_function, psi_flag = False, fid_flag = False, progress_bar = True, solver='expm'):
        """
        Calculate evolution of Schedinger equation under multiple Hamiltonians (multiple drives)

        :param calc_H_as_time_function (function):  parameters of Hamiltonian
        :param psi_flag (bool): save evolution of wavefunctions if False return only final psi
        :param progress_bar(bool): show progress_bar
        :param solver: exmp or RK4
        :return: fidelity(initstate, targetstate), psilist (if psi_flag==True)
        """
        if solver == 'RK4':
            self.calc_H_as_time_function = calc_H_as_time_function
        psi = tf.tile(self.initstate[tf.newaxis],\
                      (self.calc_timedepH(calc_H_as_time_function(self.timelist[0]), self.timelist[0]).shape[0], 1, 1))
        if fid_flag:
            resultFid = []
            resultFid.append(self.calc_fidelity_psi(psi))
        if psi_flag:
            psilist = []
            psilist.append(psi)
        if progress_bar:
            i_range = tqdm(range(1, self.timelist.shape[0]))
        else:
            i_range = range(1, self.timelist.shape[0])
        for i in i_range:
            if solver == 'expm':
                psi = self.__solveSE_expm(psi, self.calc_timedepH(calc_H_as_time_function(self.timelist[i-1]), self.timelist[i-1]),\
                                     self.timelist[i]-self.timelist[i-1])
            elif solver == 'RK4':
                psi = self.__solveSE_RK4(psi, self.timelist[i], self.timelist[i] - self.timelist[i - 1])
            if fid_flag:
                resultFid.append(self.calc_fidelity_psi(psi))
            if psi_flag:
                psilist.append(psi)
        if(psi_flag and fid_flag):
            return tf.transpose(tf.math.abs(tf.convert_to_tensor(resultFid)), (1,0,2)),\
                   tf.transpose(tf.convert_to_tensor(psilist, psi.dtype), (1,0,2,3))
        elif(fid_flag):
            return tf.transpose(tf.math.abs(tf.convert_to_tensor(resultFid)), (1,0,2))
        elif(psi_flag):
            return tf.transpose(tf.convert_to_tensor(psilist, psi.dtype), (1,0,2,3))
        else:
            return psi




    def scan_fidelityME(self, calc_H_as_time_function, rho_flag = False, fid_flag=False,
                        progress_bar = False):
        self.calc_H_as_time_function = calc_H_as_time_function
#         if self.initstate.shape[1] != 1:
#             print('No initial rho if it is supposed to be set via set_initstate')
        rho = tf.tile(self.initrho[tf.newaxis],\
                   (self.calc_timedepH(calc_H_as_time_function(self.timelist[0]),
                                       self.timelist[0]).shape[0], 1, 1))
        if(fid_flag):
            resultFid = []
            resultFid.append(self.calc_fidelity_rho(rho))
        if rho_flag:
            rholist = []
            rholist.append(rho)
        if progress_bar:
            i_range = tqdm(range(1, self.timelist.shape[0]))
        else:
            i_range = range(1, self.timelist.shape[0])
        for i in i_range:
            rho = self.__solveME(rho, self.timelist[i-1], self.timelist[i]-self.timelist[i-1])
#             if np.abs(tf.linalg.trace(rho)[0]-1)>0.1:
#                 print('Farewell! Time:', self.timelist[i], tf.linalg.trace(rho))
            if(fid_flag): resultFid.append(self.calc_fidelity_rho(rho))
            if(rho_flag): rholist.append(rho)
        if(rho_flag and fid_flag):
            return tf.transpose(tf.math.abs(tf.convert_to_tensor(resultFid)), (1,0,2)),\
                   tf.transpose(tf.convert_to_tensor(rholist, rho.dtype), (1,0,2,3))
        elif(fid_flag):
            return tf.transpose(tf.math.abs(tf.convert_to_tensor(resultFid)), (1,0,2))
        elif(rho_flag):
            return tf.transpose(tf.convert_to_tensor(rholist, rho.dtype), (1,0,2,3))
        else:
            return rho
            
    def get_superoperator(self, hilbert_dim, basis, calc_Phi, progress_bar=False):
        
        superoperator = []
        
        for n in tqdm(range(len(basis)**2)):
            rho = np.zeros((hilbert_dim, hilbert_dim))
            rho[basis[n%len(basis)], basis[n//len(basis)]] = 1
            
            self.initrho = tf.convert_to_tensor(rho, dtype=tf.complex128)

            rholist = self.scan_fidelityME(calc_Phi, 
                                           progress_bar=progress_bar)
            
            rholist = rholist.numpy()[:, basis, :][:, :, basis]

#             # Some spanish shame
#             superoperator_ = []
#             for rho in rholist:
#                 superoperator_.append(np.ravel(rho.T))
#             superoperator.append(np.asarray(superoperator_))
            rholist = rholist.reshape((rholist.shape[0], 
                                       len(basis) ** 2), 
                                      order='F')
            superoperator.append(rholist)
        
        return np.stack(superoperator, axis=1)
#         return np.transpose(np.asarray(superoperator), (1, 0, 2))
    
    
    def unitary2superoperator(self, U, basis=None):
        
        if(basis==None): basis=list(range(0, U.shape[0]))
        superoperator = []
        
        for n in tqdm(range(len(basis)**2)):
            rho = np.zeros((hilbert_dim, hilbert_dim))
            rho[basis[n%len(basis)], basis[n//len(basis)]] = 1
            
            rho = U@rho@np.conjugate(U.T)
            
            rho = rho[basis, :][:, basis]
#             # Some spanish shame
#             superoperator_ = []
#             for rho in rholist:
#                 superoperator_.append(np.ravel(rho.T))
#             superoperator.append(np.asarray(superoperator_))
            rho = rho.reshape((len(basis) ** 2), order='F')
            superoperator.append(rholist)
        
        return np.stack(superoperator, axis=0)
#         return np.transpose(np.asarray(superoperator), (1, 0, 2))
    
    
    
    def plot_fidelity(self, fidelity, x, y, xlabel = '', ylabel = '', opt_lines = True):
        '''
        Plot fidelity colormap.

        Params
        ------
            resultFid      : fidelity[time, yGrid]
            yGrid          : ylist (e.g. frequency list)
            opt_lines      : plot optimal frequency line (True/False)
        '''
        fig, axs = plt.subplots(nrows = 1, ncols = fidelity.shape[2], figsize = (5*fidelity.shape[2], 4))
        xGrid, yGrid = np.meshgrid(x, y)
        cmap_set = 'RdBu'
        if fidelity.shape[2] == 1:
            cb = axs.pcolormesh(xGrid, yGrid, fidelity[:,:,0], cmap = cmap_set)
            axs.set_title('fidelity')
            axs.set_xlabel(xlabel)
            axs.set_ylabel(ylabel)
            fig.colorbar(cb, ax=axs)

            opt_y_ind = np.argmax(np.real(fidelity))//fidelity.shape[1]
            opt_x_ind = np.argmax(np.real(fidelity))%fidelity.shape[1]
            print('opt '+xlabel, float(x[opt_x_ind]))
            print('opt '+ylabel, y[opt_y_ind])
            print('fidelity',np.abs(fidelity[opt_y_ind, opt_x_ind, 0]))
            if opt_lines:
                axs.hlines(y[opt_y_ind], x[0], x[-1])
                axs.vlines(x[opt_x_ind], y[0], y[-1])
        else:
            for i in range(fidelity.shape[2]):
                cb = axs[i].pcolormesh(xGrid, yGrid, fidelity[:,:,i], cmap = cmap_set)
                axs[i].set_title('fidelity plot '+str(i))
                axs[i].set_xlabel(xlabel)
                axs[i].set_ylabel(ylabel)
                fig.colorbar(cb, ax=axs[i])

                opt_y_ind = np.argmax(np.real(fidelity[:,:,i]))//fidelity.shape[1]
                opt_x_ind = np.argmax(np.real(fidelity[:,:,i]))%fidelity.shape[1]
                print('plot', i, ':')
                print('opt '+xlabel, float(x[opt_x_ind]))
                print('opt '+ylabel, y[opt_y_ind])
                print('fidelity',np.abs(fidelity[opt_y_ind, opt_x_ind, i]))
                if opt_lines:
                    axs[i].hlines(y[opt_y_ind], x[0], x[-1])
                    axs[i].vlines(x[opt_x_ind], y[0], y[-1])
        plt.show()
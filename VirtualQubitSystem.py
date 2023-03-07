# copied from https://github.com/SimakovIlya/VirtualQubits

import tensorflow as tf
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from tqdm import tqdm
from copy import copy




mytensordot = lambda A, B: tf.reshape(tf.transpose(tf.tensordot(A, B, axes=0), (0, 2, 1, 3)),\
                                      (A.shape[0]*B.shape[0], A.shape[1]*B.shape[1]))




def mytensordotlist(l):
    res = l[0]
    if len(l) > 1:
        res = l[0]
        for i in range(1, len(l)):
            res = mytensordot(res, l[i])
    return res




def basis(n, i):
    psi = np.zeros((n, 1))
    psi[i] = 1
    return tf.constant(psi, dtype=tf.complex128)




class VirtualQubitSystem:
    '''
    Class VirtualQubitSystem for simulatining quantum system with fluxonium qubits.

    Params
    ------
    H_args = {
        'nQbt':     number of qubits
        'nLvl':     list of number of levels taken into account for each qubit (2 or 3) 
        'energies': list of |l><l| operators (Qobj)
        'N':        list of charge operators (Qobj)
        'g':        list of couplings between qubits (in sequence 01, 02, 03,..., 12, 13,...)
        't_dep':    list of time-dependent indecies in Hamiltonian
        'T1':       list of amplitude damping times (unnecessary) only for 2 and 3 lvls
        'Tf':       list of phase damping times (unnecessary) only for 2 and 3 lvls
    }
    '''
    def __init__(self, sys_args):
        self.nQbt = sys_args['nQbt']
        self.nLvl = sys_args['nLvl']
        self.allLvl = np.prod(self.nLvl)
        self.energies = copy(sys_args['energies'])
        self.N = copy(sys_args['N'])
        
        assert self.nQbt == len(self.nLvl)
        assert self.nQbt == len(self.energies)
        assert self.nQbt == len(self.N)
        assert self.nQbt*(self.nQbt-1)//2 == len(sys_args['g'])
        
        self.g = np.zeros((self.nQbt, self.nQbt))
        tmp = 0
        for i in range(self.nQbt):
            for j in range(i+1, self.nQbt):
                self.g[i,j] = sys_args['g'][tmp]
                tmp += 1
        # convert energies and N to multi-qubit space 
        for i in range(self.nQbt):
            tmp_energies = []
            tmp_N = []
            for j in range(self.nQbt):
                if i == j:
                    tmp_energies += [self.energies[i]]
                    tmp_N += [self.N[i]]
                else:
                    tmp_energies += [tf.eye(self.nLvl[j], dtype=tf.complex128)]
                    tmp_N += [tf.eye(self.nLvl[j], dtype=tf.complex128)]
            self.energies[i] = mytensordotlist(tmp_energies)
            self.N[i] = mytensordotlist(tmp_N)
        # constucting Hamiltonian
        H_energies = []
        H_int = []
        for i in range(self.nQbt):
            H_energies += [self.energies[i]]
            for j in range(i+1, self.nQbt):
                H_int += [self.g[i,j]*self.N[i]@self.N[j]]
        # finding [H0, Ht1, Ht2, ...]
        self.H = [0*H_energies[0]]
        H_list = H_energies + H_int
        cnt = 0
        for i in range(len(H_list)):
            flag = False
            for j in range(len(sys_args['t_dep'])):
                if cnt == sys_args['t_dep'][j]:
                    flag = True
            if flag:
                self.H += [H_list[i]]
            else:
                self.H[0] = self.H[0] + H_list[i]
            cnt += 1
            
        self.c_T1 = []
        if ('T1' in sys_args):
            for i in range(self.nQbt):
                if sys_args['T1'][i] > 0:
                    coef1 = np.sqrt(1/sys_args['T1'][i])
                    if self.nLvl[i] == 2:
                        dampingChannels = [coef1*np.array([[0, 1],
                                                           [0, 0]])]
                    elif self.nLvl[i] == 3:
                        dampingChannels = [coef1*np.array([[0, 1, 0],
                                                           [0, 0, 0],
                                                           [0, 0, 0]])]
                    for item in dampingChannels:
                        tmp = []
                        for j in range(self.nQbt):
                            if i == j:
                                tmp += [item]
                            else:
                                tmp += [np.identity(self.nLvl[j])]
                        c_T1 = tmp[0]
                        for j in range(1, len(tmp)):
                            c_T1 = np.kron(c_T1, tmp[j]) 
                        self.c_T1 += [tf.constant(c_T1, dtype=tf.complex128)]
                elif sys_args['T1'][i] != 0:
                    print('T1 <= 0 and != 0')
        self.c_Tf = []
        if ('Tf' in sys_args):
            for i in range(self.nQbt):
                if sys_args['Tf'][i] > 0:
                    coef2 = np.sqrt(1/(2*sys_args['Tf'][i]))
                    if self.nLvl[i] == 2:
                        dampingChannels = [coef2*np.array([[1, 0],
                                                           [0, -1]])]
                    elif self.nLvl[i] == 3:
                        dampingChannels = [coef2*np.array([[1, 0, 0],
                                                           [0, -1, 0],
                                                           [0, 0, 0]])]
                    for item in dampingChannels:
                        tmp = []
                        for j in range(self.nQbt):
                            if i == j:
                                tmp += [item]
                            else:
                                tmp += [np.identity(self.nLvl[j])]
                        c_Tf = tmp[0]
                        for j in range(1, len(tmp)):
                            c_Tf = np.kron(c_Tf, tmp[j]) 
                        self.c_Tf += [tf.constant(c_Tf, dtype=tf.complex128)]
                elif sys_args['Tf'][i] != 0:
                    print('Tf <= 0 and != 0')
                
            
            
            
            
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




    def __solveSE(self, psi, H, dt):
        return tf.linalg.expm(-1j*dt*H)@psi
    
    
    
    
    def __Lindblad(self, rho, t):
        H = self.calc_timedepH(self.calc_Phi(t))
        res = -1j*(H@rho - rho@H)
        for c in self.c_T1:
            cadj = tf.linalg.adjoint(c)
            res += c[tf.newaxis, :, :]@rho@cadj[tf.newaxis, :, :] - 0.5*\
                   (cadj[tf.newaxis, :, :]@c[tf.newaxis, :, :]@rho +\
                    rho@cadj[tf.newaxis, :, :]@c[tf.newaxis, :, :])
        for c in self.c_Tf:
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




    def set_calc_timedepH(self, func):
        self.calc_timedepH = func

    
    
    
    def scan_fidelitySE(self, calc_Phi, psi_flag = False, progress_bar = True):
        psi = tf.tile(self.initstate[tf.newaxis],\
                      (self.calc_timedepH(calc_Phi(self.timelist[0])).shape[0], 1, 1))
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
            psi = self.__solveSE(psi, self.calc_timedepH(calc_Phi(self.timelist[i-1])),\
                                 self.timelist[i]-self.timelist[i-1])
            resultFid.append(self.calc_fidelity_psi(psi))
            if psi_flag:
                psilist.append(psi)
        if psi_flag:
            return tf.transpose(tf.math.abs(tf.convert_to_tensor(resultFid)), (1,0,2)),\
                   tf.transpose(tf.convert_to_tensor(psilist, psi.dtype), (1,0,2,3))
        else:
            return tf.transpose(tf.math.abs(tf.convert_to_tensor(resultFid)), (1,0,2))




    def scan_fidelityME(self, calc_Phi, rho_flag = False, progress_bar = True):
        self.calc_Phi = calc_Phi
        if self.initstate.shape[1] != 1:
            print('No initial rho')
        rho = tf.tile(self.initrho[tf.newaxis],\
                      (self.calc_timedepH(calc_Phi(self.timelist[0])).shape[0], 1, 1))
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
            resultFid.append(self.calc_fidelity_rho(rho))
            if rho_flag:
                rholist.append(rho)
        if rho_flag:
            return tf.transpose(tf.math.abs(tf.convert_to_tensor(resultFid)), (1,0,2)),\
                   tf.transpose(tf.convert_to_tensor(rholist, rho.dtype), (1,0,2,3))
        else:
            return tf.transpose(tf.math.abs(tf.convert_to_tensor(resultFid)), (1,0,2))
    
    
    
    
    def get_propagator(self, H_basis, statelist, calc_Phi, progress_bar = True):
        e, v = np.linalg.eigh(H_basis)
        v_inv = np.linalg.inv(v)
        tf_v_inv = tf.constant(v_inv, dtype=tf.complex128)
        propagator = []
        for i in range(len(statelist)**2):
            rho = np.zeros((H_basis.shape[0], H_basis.shape[0]), dtype=np.complex128)
            rho[i%len(statelist), i//len(statelist)] = 1
            rho = tf.constant(v@rho@np.conj(v.T), dtype=tf.complex128)
            # calculate
            self.initrho = rho
            _, rholist = self.scan_fidelityME(calc_Phi, True, progress_bar)
            # to calc basis
#             rholist = tf_v_inv[tf.newaxis,tf.newaxis,:,:]@rholist@\
#                       tf.linalg.adjoint(tf_v_inv[tf.newaxis,tf.newaxis,:,:])
            rholist = tf_v_inv@rholist@\
                      tf.linalg.adjoint(tf_v_inv)
            rholist = rholist.numpy()[:,:,statelist][:,:,:,statelist]
            rholist = rholist.reshape((rholist.shape[0], rholist.shape[1], len(statelist)**2), order='F')
            propagator.append(rholist)
        propagator = np.stack(propagator, axis=2)
        return propagator
    
    
    
    
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
        
    
    
    
    def plot_statePopulation(self, statePopulation, statelist, y, xlabel = '', ylabel = ''):
        '''
        Plot each qubit population colormap.

        Params
        ------
            statePopulation      : population[frequency, time, allLvl]
            y                    : frequency list
        '''
        fig, axs = plt.subplots(nrows = 1, ncols = len(statelist), figsize = (5*len(statelist), 4))
        xGrid, yGrid = np.meshgrid(self.timelist, y)
        cmap_set = 'RdBu'#'bwr'
        if len(statelist) == 1:
            cb = axs.pcolormesh(xGrid, yGrid, statePopulation[:,:,statelist[0]], cmap = cmap_set)
            axs.set_title('state ' + self.get_statelabel(statelist[0]))
            axs.set_xlabel(xlabel)
            axs.set_ylabel(ylabel)
            fig.colorbar(cb, ax=axs)
        else:
            for i in range(len(statelist)):
                cb = axs[i].pcolormesh(xGrid, yGrid, statePopulation[:,:,statelist[i]], cmap = cmap_set)
                axs[i].set_title('state ' + self.get_statelabel(statelist[i]))
                axs[i].set_xlabel(xlabel)
                axs[i].set_ylabel(ylabel)
                fig.colorbar(cb, ax=axs[i])
        plt.show()
        
        
        
    def get_statelabel(self, stateindex):
        '''
        Caltulate statelabel via index in density matrix.

        Params
        ------
            stateindex     : index in density matrix

        Returns
        -------
            string.        : statelabel
        '''
        label = ''
        for i in range(len(self.nLvl)-1, -1, -1):
            label = str(stateindex%self.nLvl[i]) + label
            stateindex //= self.nLvl[i]
        return label
    
    
    
    
    # def optimize(self, iters, opt, lmbd=0.05):
    #     self.Phi = tf.Variable(tf.random.normal((self.timelist.shape[0],), 
    #                                               dtype=tf.float64))
    #     # self.Phi = tf.Variable([-0.5]*self.timelist.shape[0], 
    #     #                                           dtype=tf.float64)
    #     self.max_Phi = 0.5
        
    #     losses = tf.constant([0], dtype=tf.float64)

    #     for i in tqdm(range(iters)):
    #         loss = tf.cast(self.optimization_one_step(opt, lmbd),
    #                        dtype=tf.float64)
    #         losses = tf.concat([losses, loss], axis=0)
            
    #     return losses[1:]
    
    
    
    
    # def optimization_one_step(self, opt, lmbd=0.05):
    #     with tf.GradientTape() as tape:
    #         Phi = self.max_Phi * tf.nn.tanh(self.Phi)
    #         psi = tf.tile(self.initstate[tf.newaxis],\
    #                   (self.calc_fluxoniumdriveH([Phi[0]]).shape[0], 1, 1))
    #         resultFid = []
    #         resultFid.append(self.calc_fidelity(psi))
    #         cond = lambda i, psi: i < Phi.shape[0]
    #         def body(i, psi):
    #             psi = self.__solveSE(psi, self.calc_fluxoniumdriveH([Phi[i]]),\
    #                              self.timelist[i]-self.timelist[i-1])
    #             resultFid.append(self.calc_fidelity(psi))
    #             return i+1, psi
    #         i0 = tf.constant(0)
    #         _, psi = tf.while_loop(cond, body, loop_vars=[i0, psi])
            
    #         resultFid = tf.transpose(tf.math.abs(tf.convert_to_tensor(resultFid)), (1,0))
    #         loss = tf.reshape(1.-tf.keras.backend.max(tf.keras.backend.sum(resultFid,
    #                                                                        axis=1))/resultFid.shape[1], (1))
    #         # L1 regularizer, makes signal piecewise continuous
    #         L1 = lmbd * tf.abs(self.Phi[:-1] - self.Phi[1:])
    #         loss_with_L1 = tf.cast(loss, dtype=self.Phi.dtype) + L1
            
    #     grad = tape.gradient(loss_with_L1, self.Phi)
    #     opt.apply_gradients(zip([grad], [self.Phi]))
    #     return loss

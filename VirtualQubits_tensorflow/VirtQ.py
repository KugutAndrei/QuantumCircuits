import tensorflow as tf
import numpy as np
from tqdm import tqdm



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
        self.initstate = tf.convert_to_tensor(initstate, dtype=tf.complex128)
        if initstate.shape[1] == 1:
            self.initrho = initstate @ tf.linalg.adjoint(initstate)
    
    
    
    
    def set_targetstate(self, targetstate):
        '''
        Set targetstate

        Params
        ------
            targetstate (tensor): target state (psi fuction)
        '''
        self.targetstate = tf.convert_to_tensor(targetstate, dtype=tf.complex128)
        self.targetstateadjoint = tf.linalg.adjoint(targetstate)
    
    
    
    
    def calc_fidelity_psi(self, psi):
        return tf.reshape(tf.linalg.diag_part(self.targetstateadjoint @ psi),\
                          (psi.shape[0], psi.shape[2])) 




    def calc_fidelity_rho(self, rho):
        # print(tf.sqrt(tf.math.abs(self.targetstateadjoint@rho@self.targetstate)))
        return tf.reshape(tf.sqrt(tf.math.abs(self.targetstateadjoint@rho@self.targetstate)),\
                          (rho.shape[0], 1))




    def __solveSE_expm(self, psi, H, dt):
        return tf.linalg.expm(-1j*dt*tf.convert_to_tensor(H, dtype=complex128))@psi

    def __Schrodinger_step(self, psi, t):
        H = self.calc_timedepH(self.calc_H_as_time_function(t)[:, :, tf.newaxis, tf.newaxis], t)
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
        H = self.calc_timedepH(self.calc_H_as_time_function(t)[:, :, tf.newaxis, tf.newaxis], t)
        res = -1j*(H@rho - rho@H)
        for c in self.Lindblad_operators:
            cadj = tf.linalg.adjoint(tf.convert_to_tensor(c, dtype=complex128))
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
                      (self.calc_timedepH(calc_H_as_time_function(self.timelist[0])[:, :, tf.newaxis, tf.newaxis], self.timelist[0]).shape[0], 1, 1))
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
                psi = self.__solveSE_expm(psi, self.calc_timedepH(calc_H_as_time_function(self.timelist[i-1])[:, :, tf.newaxis, tf.newaxis], self.timelist[i-1]),\
                                     self.timelist[i]-self.timelist[i-1])
            elif solver == 'RK4':
                psi = self.__solveSE_RK4(psi, self.timelist[i], self.timelist[i] - self.timelist[i - 1])
            if fid_flag:
                resultFid.append(self.calc_fidelity_psi(psi))
            if psi_flag:
                psilist.append(psi)
        if(psi_flag and fid_flag):
            return np.transpose(np.abs(np.asarray(resultFid)), (1,0,2)),\
                   np.transpose(np.asarray(psilist), (1,0,2,3))
        elif(fid_flag):
            return np.transpose(np.abs(np.asarray(resultFid)), (1,0,2))
        elif(psi_flag):
            return np.transpose(np.asarray(psilist), (1,0,2,3))
        else:
            return psi.numpy()




    def scan_fidelityME(self, calc_H_as_time_function, rho_flag = False, fid_flag=False,
                        progress_bar = False):
        self.calc_H_as_time_function = calc_H_as_time_function
#         if self.initstate.shape[1] != 1:
#             print('No initial rho if it is supposed to be set via set_initstate')
        rho = tf.tile(self.initrho[tf.newaxis],\
                   (self.calc_timedepH(calc_H_as_time_function(self.timelist[0])[:, :, tf.newaxis, tf.newaxis],
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
            return np.transpose(np.abs(np.asarray(resultFid)), (1,0,2)),\
                   np.transpose(np.asarray(rholist), (1,0,2,3))
        elif(fid_flag):
            return np.transpose(np.abs(np.asarray(resultFid)), (1,0,2))
        elif(rho_flag):
            return np.transpose(np.asarray(rholist), (1,0,2,3))
        else:
            return rho.numpy()
            
    def get_superoperator(self, hilbert_dim, basis, calc_Phi, progress_bar=False):
        
        superoperator = []
        
        if progress_bar:
            n_range = tqdm(range(len(basis)**2))
        else:
            n_range = range(len(basis)**2)
        
        for n in n_range:
            rho = np.zeros((hilbert_dim, hilbert_dim), complex)
            rho[basis[n%len(basis)], basis[n//len(basis)]] = 1   
            self.initrho = tf.convert_to_tensor(rho, dtype=tf.complex128)
            
            rholist = self.scan_fidelityME(calc_Phi, progress_bar=False)
            
            rholist = rholist.numpy()[:, basis, :][:, :, basis]
            rholist = rholist.reshape((rholist.shape[0], 
                                       len(basis) ** 2), 
                                      order='F')
            superoperator.append(rholist)
        
        return np.stack(superoperator, axis=1)
    
    
    

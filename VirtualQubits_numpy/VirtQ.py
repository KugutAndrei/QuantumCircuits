import numpy as np
import scipy as scp
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
            
            
    def ajoint(M): return np.matrix_transpose(np.conjugate(M))

                    
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
            self.initrho = initstate @ adjoint(initstate)
    
    
    
    def set_targetstate(self, targetstate):
        '''
        Set targetstate

        Params
        ------
            targetstate (tensor): target state (psi fuction)
        '''
        self.targetstate = targetstate 
    
    
    
    def calc_fidelity_psi(self, psi):
        return np.reshape(np.diagonal((adjoint(self.targetstate) @ psi), axis1=-2, axis2=-1),\
                          (psi.shape[0], psi.shape[2]))




    def calc_fidelity_rho(self, rho):
        return np.reshape(np.sqrt(np.abs(adjoint(self.targetstate)@rho@self.targetstate)),\
                          (rho.shape[0], 1))



    def __solveSE_expm(self, psi, H, dt):
        return scp.linalg.expm(-1j*dt*H)@psi
    
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
            
            cadj = np.expand_dims(adjoint(c), axis=0)
            c_e = np.expand_dims(c, axis=0)
            
            res += c_e@rho@cadj - 0.5*\
                   (cadj@c_e@rho +\
                    rho@cadj@c_e)
        return res



    def __solveME(self, rho, t, dt):
        k1 = self.__Lindblad(rho, t)
        k2 = self.__Lindblad(rho+dt*k1/2, t+dt/2)
        k3 = self.__Lindblad(rho+dt*k2/2, t+dt/2)
        k4 = self.__Lindblad(rho+dt*k3, t+dt)
        return rho + dt/6*(k1+2*k2+2*k3+k4)



    def scan_fidelitySE(self, calc_H_as_time_function, psi_flag = False, fid_flag = False, progress_bar = True, solver = 'RK4'):
        """
        Calculate evolution of Schedinger equation under multiple Hamiltonians (multiple drives)

        :param calc_H_as_time_function (function):  parameters of Hamiltonian
        :param psi_flag (bool): save evolution of wavefunctions if False return only final psi
        :param progress_bar(bool): show progress_bar
        :return: fidelity(initstate, targetstate), psilist (if psi_flag==True)
        
        """
        
        
        if solver == 'RK4':
            self.calc_H_as_time_function = calc_H_as_time_function
        
        # rebuilding initstate with shapes (H_dim, basis_dim) to (params_dim, basis_dim) by tiling
        psi = np.tile(np.expand_dims(self.initstate, axis=0),\
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
            return np.matrix_transpose(np.abs(np.asarray(resultFid)), (1,0,2)),\
                   np.matrix_transpose(np.asarray(psilist, psi.dtype), (1,0,2,3))
        elif(fid_flag):
            return np.matrix_transpose(np.abs(np.asarray(resultFid)), (1,0,2))
        elif(psi_flag):
            return np.matrix_transpose(np.asarray(psilist, psi.dtype), (1,0,2,3))
        else:
            return psi




    def scan_fidelityME(self, calc_H_as_time_function, rho_flag = False, fid_flag=False,
                        progress_bar = False):
        self.calc_H_as_time_function = calc_H_as_time_function

        rho = np.tile(np.expand_dims(self.initrho, axis=0),\
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
            
            if(fid_flag): resultFid.append(self.calc_fidelity_rho(rho))
            
            if(rho_flag): rholist.append(rho)
            
        if(rho_flag and fid_flag):
            return np.matrix_transpose(np.abs(np.asarray(resultFid)), (1,0,2)),\
                   np.matrix_transpose(np.asarray(rholist, rho.dtype), (1,0,2,3))
        elif(fid_flag):
            return np.matrix_transpose(np.abs(np.asarray(resultFid)), (1,0,2))
        elif(rho_flag):
            return np.matrix_transpose(np.asarray(rholist, rho.dtype), (1,0,2,3))
        else:
            return rho
            
    def get_superoperator(self, hilbert_dim, basis, calc_Phi, progress_bar=False):
        
        superoperator = []
        
        for n in tqdm(range(len(basis)**2)):
            rho = np.zeros((hilbert_dim, hilbert_dim))
            rho[basis[n%len(basis)], basis[n//len(basis)]] = 1   
            self.initrho = np.asarray(rho, dtype=complex)
            
            rholist = self.scan_fidelityME(calc_Phi, progress_bar=progress_bar)
            
            rholist = rholist.numpy()[:, basis, :][:, :, basis]
            rholist = rholist.reshape((rholist.shape[0], 
                                       len(basis) ** 2), 
                                      order='F')
            superoperator.append(rholist)
        
        return np.stack(superoperator, axis=1)
    
    

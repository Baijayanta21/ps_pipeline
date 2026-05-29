import numpy as np
import time
from numba import njit
import builtins as blt
from datetime import datetime
            
@njit
def correlate_fast(GV, ni, corr):
    r"""
    Helper function (**numba jit accelerated**) which performs the **Cross TGE** estimator.
    
    Parameters
    ----------
    GV : np.ndarray
        Convolved gridded visibility array (3D array) :math:`\mathcal{V}_{cg}(\nu_n)^{XX/YY}`. **GV** structure is (Polarization, Grid points, Frequency).
    ni : np.ndarray
        1D array containing the info about which grid point falls into which annular bin. Must contain values in between :math:`0,\textrm{Nbin}-1`. 
    corr : np.ndarray
        2D array of shape (Nbin, NC).

    Returns
    -------
    None
        This function does not return anything.
    """
    print(f'Channels   : {nc}')
    for ii in range(ni.size): # loop over the grid point 
        X  = GV[0,ii]  # v_cg^XX
        Xc = X.conj()  # (v_cg^XX)*
        Y  = GV[1,ii]  # v_cg^YY
        Yc = Y.conj()  # (v_cg^YY)*
        
        index = ni[ii] # which bin this grid point falls into

        # cross correlate along different frequency 
        for nc1 in range(nc):          # first frequency
            for nc2 in range(nc1, nc): # other one, nc2 starts at nc1 to avoid double counting 
                nn = nc2-nc1           # separation between two frequency
                corr[index, nn] += (X[nc1]*Yc[nc2]+Y[nc1]*Xc[nc2]).real # the estimator
                
def correlate(GV, ni, Nbin):
    r"""
    Given the gridded visibility data, it performs correlation using **Cross TGE** estimator which is defined as follows : 


    .. math ::

        \text{corr}(\nu_a,\nu_b) = \mathcal{R}e\left[\mathcal{V}_{cg}^{\rm RR}(\nu_a) \mathcal{V}_{cg}^{*\rm LL}(\nu_b) + \mathcal{V}_{cg}^{\rm LL}(\nu_a) \mathcal{V}_{cg}^{*\rm RR}(\nu_b) \right]

    Then it collapses the frequency axis to get :math:`\text{corr}(\Delta\nu)`.
    
    Parameters
    ----------
    GV : np.ndarray
        Convolved gridded visibility array (3D array) :math:`\mathcal{V}_{cg}(\nu_n)^{XX/YY}`. **GV** structure is (Polarization, Grid points, Frequency).
    ni : np.ndarray
        1D array containing the info about which grid point falls into which annular bin. Must contain values in between :math:`0,\textrm{Nbin}-1`. 
    Nbin : int
        No of annular bins the whole :math:`uv` plane has been divided into.
        
    Returns
    -------
    corr : np.ndarray
        2D array of shape (Nbin, NC).

    Examples
    --------
    >>> import numpy as np
    >>> import correlate 
    Imported correlate, Import Time : 20/05/26 | 06:03:15 PM
    >>> # Load binning info
    >>> BIN  = np.load('/home/baijayanta/MY_Documented_Code/tge/bin_info_7200.npz')
    >>> # No of annular bins the whole uv plane has been divided into
    >>> Nbin = int(BIN['Nbin'])    
    >>> # Contains info about which grid point fall into which bin
    >>> ni   = BIN['ni']             
    >>> GV = np.load('/home/baijayanta/CFFI_C/GV_7200_pool.npy')
    >>> # GV shape [nstokes,ni,NC] with nstokes = 2
    >>> print(f'{GV.shape   = }')    
    GV.shape   = (2, 46438, 768)
    >>> print(f'No of Bins = {Nbin}')
    No of Bins = 20
    >>> # perform correlation
    >>> corr = correlate.correlate(GV, ni, Nbin)
    Channels   : 768
    Time Taken : 7.400 seconds.
    >>> print(f'{corr.shape   = }') 
    corr.shape   = (20, 768)
    
    """
    start = time.time()
    
    # GV shape [nstokes,ni,nc] with nstokes = 2 
    # only XX and YY polarizations are included
    # Cross TGE estimator
    # print(GV.shape) 
    
    blt.nc = GV.shape[-1]             # no of channels available in gridded visibility GV
    
    corr   = np.zeros((Nbin, nc))     # create an empty array

    correlate_fast(GV, ni, corr)      # perform correlation 
        
    print(f"Time Taken : {(time.time()-start):.3f} seconds.")
    
    return corr

print(f"Imported correlate, Import Time : {datetime.now().strftime('%d/%m/%y | %I:%M:%S %p')}")
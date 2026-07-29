import numpy as np
import time
from numba import njit
from datetime import datetime

@njit
def correlate_fast(GV, ni, corr, nc):
    r"""
    Helper function (**numba jit accelerated**) which performs the **Cross TGE** estimator.
    
    Parameters
    ----------
    GV : np.ndarray
        Convolved gridded visibility array (3D array) :math:`\mathcal{V}_{cg}(\nu_n)^{XX/YY}`. **GV** structure is (Polarization, Grid points, Frequency).
    ni : np.ndarray
        1D array containing the info about which grid point falls into which annular bin. Must contain values in between :math:`0,\textrm{Nbin}-1`. 
    corr : np.ndarray
        3D array of shape (Nbin, NC, NC).
    nc : int
        Number of frequency channels in **GV**. Passed explicitly rather than read from a
        global, because numba freezes globals as compile-time constants.

    Returns
    -------
    None
        This function does not return anything.
    """
    for ii in range(ni.size): # loop over the grid point
        X  = GV[0,ii]  # v_cg^XX
        Xc = X.conj()  # (v_cg^XX)*
        Y  = GV[1,ii]  # v_cg^YY
        Yc = Y.conj()  # (v_cg^YY)*
        
        index = ni[ii] # which bin this grid point falls into

        # cross correlate along different frequency 
        for nc1 in range(nc):          # first frequency
            for nc2 in range(nc1, nc): # other one, nc2 starts at nc1 to avoid double counting 
                corr[index, nc1, nc2] += (X[nc1]*Yc[nc2]+Y[nc1]*Xc[nc2]).real # the estimator
                
def correlate(GV, ni, Nbin):
    r"""
    Given the gridded visibility data, it performs correlation using **Cross TGE** estimator which is defined as follows : 


    .. math ::

        \text{corr}(\nu_a,\nu_b) = \mathcal{R}e\left[\mathcal{V}_{cg}^{\rm XX}(\nu_a) \mathcal{V}_{cg}^{*\rm YY}(\nu_b) + \mathcal{V}_{cg}^{\rm YY}(\nu_a) \mathcal{V}_{cg}^{*\rm XX}(\nu_b) \right]

    
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
        3D array of shape (Nbin, NC, NC).

    Examples
    --------
    >>> import numpy as np
    >>> import myutils.clfuncs.clfuncs as cfunc
    Imported myutils
    Imported clfuncs,   Import Time : 30/05/26 | 12:57:37 PM
    >>> # Load binning info
    >>> BIN  = np.load('/home/cts23ph/git_package_myutils/tge/bin_info_1000007200.npz')
    >>> Nbin = int(BIN['Nbin'])      # No of annular bins the whole uv plane has been divided into
    >>> ni   = BIN['ni']             # Contains info about which grid point fall into which bin
    >>> mask = BIN['NI'] >= 0        # mask array
    >>> GV = np.load('/home/cts23ph/git_package_myutils/tge/GV_7200.npy')
    >>> print(f'{GV.shape   = }')    # GV shape [nstokes,ni,NC] with nstokes = 2
    GV.shape   = (2, 457, 457, 768)
    >>> print(f'No of Bins = {Nbin}')
    No of Bins = 20
    >>> # perform correlation
    >>> # corr shape (Nbin, NC, NC)
    >>> corr = cfunc.correlate(GV[:, mask], ni, Nbin)
    Channels   : 768
    Time Taken : 6.861 seconds.
    >>> print(f'{corr.shape   = }') 
    corr.shape   = (20, 768, 768)
    
    """
    start = time.time()
    
    # GV shape [nstokes,ni,nc] with nstokes = 2 
    # only XX and YY polarizations are included
    # Cross TGE estimator
    # print(GV.shape) 
    
    nc = GV.shape[-1]                 # no of channels available in gridded visibility GV
    print(f'Channels   : {nc}')

    corr   = np.zeros((Nbin, nc, nc)) # create an empty array

    correlate_fast(GV, ni, corr, nc)  # perform correlation

    # fill the other half automatically
    for nc1 in range(nc):
        for nc2 in range(nc1, nc):
            corr[:, nc2, nc1] = corr[:, nc1, nc2]
        
    print(f"Time Taken : {(time.time()-start):.3f} seconds.\n")
    
    return corr

# function to compute cl(dnu,nubar) from cl(nua,nub)
def cl_dnu_nubar(maps):
    r"""
    Given the *Multi-Angular Power Specrtrum (MAPS)* :math:`C_{\ell}(\nu_a,\nu_b)` array, it computes :math:`C_{\ell}(\bar{\nu},\Delta\nu)`.
    
    Parameters
    ----------
    maps : np.ndarray
        *MAPS* :math:`C_{\ell}(\nu_a,\nu_b)` array. Must be shape of (Nbin, NC, NC).
    
    Returns
    -------
    maps1 : np.ndarray
        *MAPS* :math:`C_{\ell}(\bar{\nu},\Delta\nu)` array having shape (Nbin, NC, 2NC -1). 

    Examples
    --------
    >>> maps = np.load('cl_7200.npy')
    >>> # shape (Nbin, NC, NC)
    >>> print(f'maps.shape    : {maps.shape}')
    maps.shape    : (20, 768, 768)
    >>> maps1 = cfunc.cl_dnu_nubar(maps)
    >>> # shape (Nbin, NC, 2*NC -1)
    >>> print(f'maps1.shape   : {maps1.shape}')
    maps1.shape   : (20, 768, 1535)
    
    """
    nell, nc = maps.shape[:-1]
    maps1 = np.full((nell, nc, 2*nc-1), fill_value = np.nan) # delta-nu, nubar
    
    for nc1 in range(nc):
        for nc2 in range(nc1, nc):
            ii = nc2 - nc1 
            jj = nc1 + nc2 
            maps1[:, ii, jj] = maps[:, nc1, nc2]
    
    return maps1

# function to compute cl(dnu) from cl(dnu,nubar)
def cl_dnu(maps1):
    r"""
    Given the :math:`C_{\ell}(\bar{\nu},\Delta\nu)` array, it computes :math:`C_{\ell}(\Delta\nu)` by averaging over :math:`\bar{\nu}` axis.
    
    Parameters
    ----------
    maps1 : np.ndarray
        :math:`C_{\ell}(\bar{\nu},\Delta\nu)` array. Must be shape of (Nbin, NC, 2NC -1). 
    
    Returns
    -------
    np.ndarray
        :math:`C_{\ell}(\Delta\nu)` array having shape (Nbin, NC).  

    Examples
    --------
    >>> maps2 = cfunc.cl_dnu(maps1)
    >>> # shape (Nbin, NC)
    >>> print(f'maps2.shape   : {maps2.shape}')
    maps2.shape   : (20, 768)
    
    """
    return np.nanmean(maps1, axis = 2) # maps2

# function to compute cl(dnu) from cl(nua,nub)
def cl_dnu_nua_nub(maps):
    r"""
    Given the *Multi-Angular Power Specrtrum (MAPS)* :math:`C_{\ell}(\nu_a,\nu_b)` array, it computes :math:`C_{\ell}(\Delta\nu)`. 
    Internally  it computes :math:`C_{\ell}(\bar{\nu},\Delta\nu)` first and then using this, it computes :math:`C_{\ell}(\Delta\nu)`.
    
    Parameters
    ----------
    maps : np.ndarray
        *MAPS* :math:`C_{\ell}(\nu_a,\nu_b)` array. Must be shape of (Nbin, NC, NC).
    
    Returns
    -------
    np.ndarray
        :math:`C_{\ell}(\Delta\nu)` array having shape (Nbin, NC). 

    Examples
    --------
    >>> maps2 = cfunc.cl_dnu_nua_nub(maps) 
    >>> # shape (Nbin, NC)
    >>> print(f'maps2.shape   : {maps2.shape}')
    maps2.shape   : (20, 768)

    """
    return cl_dnu(cl_dnu_nubar(maps)) # maps2

print(f"Imported clfuncs,   Import Time : {datetime.now().strftime('%d/%m/%y | %I:%M:%S %p')}")
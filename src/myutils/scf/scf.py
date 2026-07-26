r"""
Smooth Component Filtering (SCF)
================================
By **convolved gridded visibility** :math:`\mathcal{V}_{cg}(\nu_n)` and given a **suitable window function** :math:`H(n)` of **specified smoothing scale** (controlled by parameter :math:`N`), we calculate the following - 

#. :math:`\left(\mathcal{V}_{cg}\ast H\right)(\nu_n) = \displaystyle \sum_m\, \mathcal{V}_{cg}(\nu_m)H(n-m)` and,
#. :math:`\left(\rm U\ast H\right)(\nu_n)    = \displaystyle\sum_m\, {\rm U}(\nu_m)H(n-m)`

Dividing them we get :

.. math::

    \begin{align}
        \boxed{\mathcal{V}_{cg}^S(\nu_n) = \left(\mathcal{V}_{cg}\ast H\right)(\nu_n)/\left(\rm U\ast H\right)(\nu_n)}
    \end{align}

We refer :math:`\mathcal{V}_{cg}^S(\nu_n)` as **Smooth Component** of the convolved gridded visibility :math:`\mathcal{V}_{cg}(\nu_n)` and **subtracting** it from the :math:`\mathcal{V}_{cg}(\nu_n)` gives us the **filtered component**. 

.. math::

    \begin{align}
    	\boxed{\mathcal{V}_{cg}^F(\nu_n) = \mathcal{V}_{cg}(\nu_n) - \mathcal{V}_{cg}^S(\nu_n)}
    \end{align}

:math:`\mathcal{V}_{cg}(\nu_i) = 0` for the channels :math:`\nu_i`, that are **flagged** and the divison by :math:`\left(\rm U\ast H\right)(\nu_n)` is necessary to **ensure correct normalization after smoothing**, i.e. :math:`\left(\mathcal{V}_{cg}\ast H\right)(\nu_n)`. Here :math:`{\rm U}(\nu_i) = 0` for the flagged channels and :math:`{\rm U}(\nu_i) = 1` otherwise. The **smoothing is restricted to a smaller range near the boundaries** and to account for this :math:`N` no of **channels are discarded** from both ends of :math:`\mathcal{V}_{cg}^F(\nu_n)` to account this.

If :math:`\Delta\nu_c` denotes the **channel separation** then for a given :math:`N` corresponds to  **smoothing scale** of :math:`N\Delta\nu_c`. For example :math:`N = 25, 50\, \text{and}\, 75` and :math:`\Delta\nu_c = 0.04 \rm MHz` corresponds to smoothing scales of :math:`N\Delta\nu_c = 1, 2\, \text{and}\, 3\, \rm MHz` respectively.

Notes
-----
We expect **SCF** to **suppress** the values of :math:`|P(k_\perp,k_\parallel)|` somewhere between the half and full width of the filter, i.e. 

.. math::
	
	\left[k_{\parallel}\right]_{\rm F}/2 \leq k_\parallel \leq \left[k_{\parallel}\right]_{\rm F} 

Where :math:`\left[k_{\parallel}\right]_{\rm F} = \dfrac{2\pi}{r'N\Delta\nu_c}`.

Window functions
================
This module supports **four types of window functions** namely -

#. **Hann/Hanning** window function :math:`H_{\rm Hann}(n)`.  
#. **Hamming** window function :math:`H_{\rm Hamming}(n)`. 
#. **Blackman** window function :math:`H_{\rm Blackman}(n)`.
#. **Kaiser** window function :math:`H_{\rm Kaiser}(n)` with :math:`\beta = 14` .

The functional form's are given as :

.. math::

    \setlength{\fboxrule}{0.7pt}
    \setlength{\fboxsep}{12pt}     % EXTRA padding
    \fcolorbox{black}{gray!3}{$
    \begin{aligned}
        H_{\rm Hann}(n) &= \frac{1}{2N}\left[0.5+0.5\cos\left(2\pi\frac{n}{2N}\right)\right] \quad& -N \leq n \leq N\\[1em]
        H_{\rm Hamming}(n) &= \frac{1}{2N}\left[0.54+0.46\cos\left(2\pi\frac{n}{2N}\right)\right] \quad& -N \leq n \leq N\\[1em]
        H_{\rm Blackman}(n) &= \frac{1}{2N}\left[0.42+0.5\cos\left(2\pi\frac{n}{2N}\right)+0.08\cos\left(4\pi\frac{n}{2N}\right)\right] \quad& -N \leq n \leq N\\[1em]
        H_{\rm Kaiser}(n) &= \frac{1}{2N}\frac{I_0\left(\sqrt{1-\left(\frac{n}{N}\right)^2}\right)}{I_0(\beta)} \quad& -N \leq n \leq N
    \end{aligned}$}



.. raw:: latex

    \begin{figure}[H]
    \centering
    \includegraphics[width=1.05\linewidth]{../../../../../Tutorials/others/windows_123.pdf}
    \caption{Different window functions with different smoothing scales.}
    \end{figure}  
    
    \clearpage

Module functions:
=================
"""
import numpy as np
from scipy.signal import convolve
import time
import sys
from datetime import datetime


dnuc = 0.04 # separation between two channels in MHz

def doscf(GV, SM, window = 'hann', method = 'auto'):
    r"""
    Given gridded visibility array it performs **SCF (Smooth Component Filtering)** given a **window function** and **suitable smoothing scale**. It **subtracts out** the **smooth part** of it and returns **residual filtered visibilities**.
    
    Parameters
    ----------
    GV : np.ndarray
        Convolved-gridded visibility array. The last dimension must have shape **(,768)**.
        
    SM : int or float
        **Smoothing scale** in MHz of specified window function.
        
    window : str {'hann','hamming','blackman','kaiser'}, optional
        Window function name. Valid ones are **'hann', 'hamming', 'blackman', 'kaiser'**. If not any of this it kills the current iteration. By default 'hann' window.
        
    method : str {'auto','direct','fft'}, optional
        The **method** by which the **convolution** is performed. Either by **'direct' or 'fft'**. By **default** is **'auto'**, means **whichever is faster is applied**.

    Returns 
    -------
    GV_filtered : np.ndarray
        The **filtered visibility** which is obtained by **subtracting out the smooth part**. Shape same as input GV array **except in the last dimension** in which first and last :math:`{\rm NW} = {\rm SM}/{\Delta\nu_c}` no of channels are sliced. Where :math:`\Delta\nu_c` is the separation between two frequency channels in MHz. By default the value is *0.04*.
        
    Examples
    --------
    >>> import numpy as np
    >>> import myutils.scf.scf as sf
    Imported myutils
    Imported SCF,       Import Time : 30/05/26 | 05:30:25 PM
    >>> GV = np.load("/home/cts23ph/git_package_myutils/tge/GV_7200.npy") # GV array
    >>> print(f"{GV.shape = }")
    GV.shape = (2, 457, 457, 768)
    >>> SM = 2 # smoothing scale in MHz
    >>> NW = int(SM/0.04)
    >>> print(f"{NW = }")
    NW = 50
    >>> # perform SCF
    >>> GVr = sf.doscf(GV, SM, window = 'hann', method = 'fft')
    <============= Performing SCF =============>
    <=== Start Time: 30/05/26 | 05:29:00 PM ===>
    Convoluting kernel : fft
    Window function    : Hanning
    Smoothing scale    : 2 MHz
    <=== End Time  : 30/05/26 | 05:29:09 PM ===>
    <=== Elapsed   : 00:00:09.24, 9.240 seconds.
    <================ Done SCF ================>
    >>> print(f"{GVr.shape = }") # 768-2*NW
    GVr.shape = (2, 457, 457, 668)

    
    .. raw:: latex

    	\begin{figure}[H]
    	\centering
    	\includegraphics[width=1.05\linewidth]{../../../../../Tutorials/others/Flag.pdf}
    	\caption{The Flagging of MWA data. The black colored regions are flagged.}
   	\end{figure}  
   	
    	\begin{figure}[H]
    	\centering
    	\includegraphics[width=1.05\linewidth]{../../../../../Tutorials/others/scf.pdf}
    	\caption{Real part of $\mathcal{V}_{cg},\mathcal{V}_{cg}^{S},\mathcal{V}_{cg}^{F}$ as a function of frequency $\nu$ for a fixed grid point corresponding to $\ell_g = 261$. This is shown only for XX polarization.}
   	\end{figure}  
    
    """

    start = datetime.now()
    print(f"<============= Performing SCF =============>")
    print(f"<=== Start Time: {start.strftime('%d/%m/%y | %I:%M:%S %p')} ===>")
    
    print(f"Convoluting kernel : {method}")
            
    # smoothing param in SCF

    NW  = int(SM/dnuc)   # N # = 50 if SM = 2 MHz and dnuc = 0.04 MHz
    NN  = 2*NW + 1       # total width (2N+1)
    
    if window == 'hann':
        
        print(f"Window function    : Hanning\nSmoothing scale    : {SM} MHz")
        win = np.hanning(NN)   # hannning window function without 1/2N factor
        
    elif window == 'hamming':
        
        print(f"Window function    : Hamming\nSmoothing scale    : {SM} MHz")
        win = np.hamming(NN)   # hamming  window function without 1/2N factor
        
    elif window == 'blackman':
        
        print(f"Window function    : Blackman\nSmoothing scale    : {SM} MHz")
        win = np.blackman(NN)  # blackman window function without 1/2N factor
        
    elif window == 'kaiser' :
        
        print(f"Window function    : Kaiser\nSmoothing scale    : {SM} MHz")
        win = np.kaiser(NN,14) # kaiser   window function without 1/2N factor
                               # beta value is 14
    else:
        print(f"Invalid window function, Valid type's are hann, hamming, blackman, kaiser.\n")
        sys.exit()
    
    
    nc = GV.shape[-1]

    win_expand = np.expand_dims(win,axis = tuple(np.arange(GV.ndim-1)))
    
    GV_SCF = convolve(GV, win_expand, method = method, mode = 'valid')  
    
    # np.expand_dims(win,axis = tuple(np.arange(GV.ndim-1))) done so that scipy can broadcast while convolving
    
    # GV_SCF shape (Npol,Nu,Nv,nc-2*NW) or (Npol,Ni,nc-2*NW)

    
    NormR = np.ones_like(GV.real)
    NormI = np.ones_like(GV.imag)

    NormR[GV.real==0.0] = 0.0
    NormI[GV.imag==0.0] = 0.0

    
    NR = convolve(NormR, win_expand, method = method, mode = 'valid')  
    NI = convolve(NormR, win_expand, method = method, mode = 'valid')  
    
    GV_SCF.real/=NR# normalized that incorporates flagging
    GV_SCF.imag/=NI# normalized that incorporates flagging

    # put the value to zero to those modes which are flagged in the data
    (GV_SCF.real)[NormR[NW:-NW]==0.0] = 0.0
    (GV_SCF.imag)[NormI[NW:-NW]==0.0] = 0.0


    # filtered convolved gridded visibility
    GV_filtered = GV[...,NW:-NW] - GV_SCF # subtract out the smooth part 
    
    
    end = datetime.now()
    print(f"<=== End Time  : {end.strftime('%d/%m/%y | %I:%M:%S %p')} ===>")
    elapsed = (end - start).total_seconds()
    hours, rem = divmod(elapsed, 3600)
    minutes, seconds = divmod(rem, 60)
    print(f"<=== Elapsed   : {int(hours):02}:{int(minutes):02}:{seconds:05.2f}, {elapsed:.3f} seconds.")
    print(f"<================ Done SCF ================>")

    return GV_filtered

print(f"Imported SCF,       Import Time : {datetime.now().strftime('%d/%m/%y | %I:%M:%S %p')}")

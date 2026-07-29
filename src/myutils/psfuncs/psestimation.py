r"""

.. raw:: latex

   \begin{table}[htbp]
     \centering
     \footnotesize
     \setlength{\tabcolsep}{20pt}
     \setlength{\arrayrulewidth}{0.5pt}

     \renewcommand{\arraystretch}{1.9}
     \setlength{\doublerulesep}{1.3pt}  % <-- reduce space between double lines
     \scalebox{1}{
       \begin{tabular}{||p{7.cm}|l|l||}
         \hline\hline
         Quantity & Mathematical symbol & Codespace name \\
         \hline\hline
         Centered frequency & $\nu_c$ & \texttt{nuc}\\ \hline
         Separation between two frequency channels & $\Delta\nu_c$ & \texttt{dnuc}\\ \hline
         Number of frequency separations used to estimate cylindrical power spectrum & $N_E$ & \texttt{NE}\\\hline
         Comoving distance at $\nu_c$   & $r$ & \texttt{r}\\ \hline
         Derivative of comoving distance w.r.t. frequency $\nu$, evaluated at $\nu_c$ & $r^{\prime} = \left.\frac{dr}{d\nu}\right|_{\nu_c}$ & \texttt{rp}\\ \hline
         Slope of Wedge boundary line & $\frac{[k_{\parallel}]_H}{k_{\perp}} = \frac{r}{r^{\prime}\nu_c}$ & \texttt{fac}\\ \hline
         Prefactor in Fourier Transform (\textit{volume factor}) & $r^2r'\Delta\nu_c(N_E-1)$ & \texttt{vfac}\\ \hline
         Component of $\vec{\bf k}$ perpendicular to line of sight & $k_{\perp}$ & \texttt{kper}\\ \hline
         Component of $\vec{\bf k}$ parallel to line of sight & $k_{\parallel}$ & \texttt{kpara}\\ \hline
         Multi-angular power spectrum (MAPS )& $C_{\ell}(\Delta\nu)$ & \texttt{cl}\\ \hline
         Cylindrical power spectrum & $P(k_{\perp},k_{\parallel})$ & \texttt{pk}\\ \hline
         Standard deviation of $P_{N}(k_{\perp},k_{\parallel})$ & $\delta P_{N}(k_{\perp},k_{\parallel})$ & \texttt{dpkn}\\ \hline
         Mean of variable $X$ & $\mu_{\rm Est}$ & \texttt{mu\_est} \\ \hline
         Standard deviation of variable $X$ & $\sigma_{\rm Est}$ & \texttt{sigma\_est} \\ \hline
         Scaled std of $P_{N}(k_{\perp},k_{\parallel})$ & $\delta P_{N}^{True}(k_{\perp},k_{\parallel}) = \sigma_{\rm Est}\,\delta P_{N}(k_{\perp},k_{\parallel})$ & \texttt{dpk}\\ \hline
         Number of logarithmic bins & $N_{\rm Bin}$         & \texttt{NBin}\\ \hline
         Binned $k$ values      & $k$                       & \texttt{kk/keff}\\ \hline
         Binned power spectrum  & $P(k)$                    & \texttt{ppk}\\ \hline
         Error in estimating binned power spectrum          & $\delta P(k)$                       & \texttt{dppk}\\ \hline
         Dimensionless power spectrum & $\Delta^2(k)$ & \texttt{dk2}\\ \hline
         $2\sigma$ uncertainty of $\Delta^2(k)$ & $2\sigma(k)$ & \texttt{dpk2}\\ \hline
         Signal to noise ratio & $\textsc{SNR} = \Delta^2(k)/\sigma(k)$ & \texttt{snr} \\ \hline
         Upper limits of power spectrum & $\Delta^2_{\text{UL}}(k)$ & \texttt{ul}\\ 
         \hline\hline
       \end{tabular}
     }
     \caption{Namespace convensions}
     \label{tab:f_outputs}
   \end{table}
   \clearpage

"""

import numpy as np
import scipy
from scipy.linalg import block_diag
from astropy.cosmology import Planck18,FlatLambdaCDM  # defines cosmological parameters, find comvoing distance
from datetime import datetime

def build_essential(nuc, dnuc, NE, lval, model = 'Planck18'):
    r"""
    For a given set of values :math:`\nu_c,\Delta\nu_c,N_E,\ell` and appropriate cosmological model either **Planck18** or **FlatLambdaCDM**, it calculates  comoving distance :math:`r` at :math:`\nu_c` and it's frequency derivative :math:`r^\prime`. It also calculates the slope of wedge boundary line, corresponding volume factor, and components of :math:`\vec{\mathbf{k}}`: :math:`k_{\perp}\,\text{and}\,k_{\parallel}` respectively.

    Parameters
    ----------
    nuc : float
        Central frequency :math:`\nu_c` in units of :math:`\rm MHz`.
    dnuc : float
        Separation between two frequency channels :math:`\Delta\nu_c` in units of :math:`\rm MHz`.
    NE : int
        Number of frequency separations used in the analysis :math:`N_E`.
    lval : np.ndarray
        Angular multipole :math:`\ell` values.
    model : str {'Planck18','FlatLambdaCDM'}, optional
        The name of cosmological model used to calculate **comoving distance** :math:`r`. By default **Planck18** is used.
    
    Returns
    -------
    r : float
        Comoving distance :math:`r` at :math:`\nu_c` in units of :math:`\rm Mpc`.
    rp : float
        Derivative of comoving distance :math:`r` w.r.t. frequency :math:`\nu`, evaluated at :math:`\nu_c`, :math:`r^{\prime} = \left.\frac{dr}{d\nu}\right|_{\nu_c}\approx \tfrac{c}{H(z)}\times\tfrac{(1+z)^2}{\nu_e = \mathbf{1420\,\rm MHz}}` in units of :math:`\rm Mpc/MHz`. 
    fac: float
        Slope of Wedge boundary line  :math:`[k_{\parallel}]_H = [r/{r^{\prime}\nu_c}]\, k_{\perp}`.
    vfac: float
        Factor (**volume factor**) in Fourier Transform, :math:`r^2r^{\prime}\Delta\nu_c(N_E-1)`.
    kper : np.ndarray
        Component of :math:`\vec{\bf k}`, perpendicular to line of sight, :math:`k_{\perp} = \ell/r`.
    kpara : np.ndarray
        Component of :math:`\vec{\bf k}`, parallel to line of sight, with :

        .. raw:: latex
        
            \begin{center}
                \tbox{$k_{\parallel m} = \dfrac{\pi m}{r^{\prime}\Delta\nu_c(N_E-1)},\quad m = 0,1,\dots,N_E-1.$}
            \end{center}
            
        .. raw:: latex

		\clearpage
	
    Examples
    --------
    >>> import numpy as np
    >>> import myutils.psfuncs.psestimation as pe
    Imported myutils
    Imported psfuncs,   Import Time : 01/06/26 | 04:05:54 PM
    >>> nuc  = 154.255 # observing frequency in MHz
    >>> dnuc = 0.04    # freq separation in MHz
    >>> NE   = 668     # no of channels used to estimate PS 
    >>> # ell values array
    >>> fname = '/home/cts23ph/git_package_myutils/tge/bin_info_1000007200.npz'
    >>> lval  = (np.load(fname)['lval']).astype('int')
    >>> r, rp, fac, vfac, kper, kpara = pe.build_essential(nuc, dnuc, NE, lval)
    --------------------------------------
    Cosmological Model: Planck18
    Redshift          : z  = 8.21
    Comoving distance : r  = 9191.20 Mpc
    rprime            : rp = 16.93 Mpc/MHz
    Sep used          : NE = 668
    --------------------------------------
    Wedge boundary slope (fac) : 3.519          
    DCT factor (vfac)          : 3.816e+10          
    kper.shape                 : (20,)          
    kpara.shape                : (668,)
    --------------------------------------

    >>> r, rp, fac, vfac, kper, kpara = pe.build_essential(nuc, dnuc, NE, lval,
    ...                                                      model = 'FlatLambdaCDM')
    --------------------------------------
    Cosmological Model: FlatLambdaCDM
    Redshift          : z  = 8.21
    Comoving distance : r  = 9209.37 Mpc
    rprime            : rp = 16.99 Mpc/MHz
    Sep used          : NE = 668
    --------------------------------------
    Wedge boundary slope (fac) : 3.514          
    DCT factor (vfac)          : 3.844e+10          
    kper.shape                 : (20,)          
    kpara.shape                : (668,)
    --------------------------------------
    >>> # kper  shape (len(lval),)
    >>> # kpara shape (NE,) 

    
    .. raw:: latex
    
        \clearpage
    
    Notes
    -----
    For calculating comoving distance :math:`r`, we use **Flat** :math:`\Lambda\rm CDM` cosmological model. The cosmological parameters are from either from `Planck 2018 results. VI. Cosmological parameters <https://arxiv.org/abs/1807.06209>`_  or **FlatLambdaCDM**. The `Planck 18 Astropy <https://docs.astropy.org/en/latest/api/astropy.cosmology.realizations.Planck18.html>`_ , `FlatLambdaCDM <https://docs.astropy.org/en/latest/api/astropy.cosmology.FlatLambdaCDM.html>`_ and module have been used here.
    
    #. For **Planck18** we have : 
    
        .. raw:: latex
    
           \begin{table}[h]
             \centering
             \footnotesize
             \setlength{\tabcolsep}{15pt}
             \setlength{\arrayrulewidth}{0.1pt}
        
             \renewcommand{\arraystretch}{1.7}
             \setlength{\doublerulesep}{1pt}  % <-- reduce space between double lines
             \scalebox{1}{
               \begin{tabular}{||l||c||c||}
                 \hline\hline
                 \multicolumn{1}{||c||}{Quantity} & \multicolumn{1}{|c||}{Mathematical symbol} & \multicolumn{1}{|c||}{Value} \\
                 \hline\hline
                 Hubble Constant at present & $H_{0}$ & $67.66\,\rm km/s/Mpc$ \\ \hline
                 Density parameter of matter & $\Omega_{m\displaystyle 0}$ & $0.30966$ \\ \hline
                 Density parameter of baryonic matter & $\Omega_{b\displaystyle 0}$ & $0.04897$ \\ \hline
                 CMB temperature at present & $T_{{\scriptscriptstyle \mathrm{CMB}}\displaystyle 0}$ & $2.7255 K$ \\ \hline  
                 Effective number of neutino species & $N_{\rm eff}$ & $3.046$\\ \hline
                 Mass of neutrinos & $m_{\nu}$ & $[0, 0, 0.06]\, \rm eV$ \\ 
                 \hline\hline
                 \end{tabular}
             }
             \caption{Cosmological parameters taken from {\href{https://arxiv.org/abs/1807.06209}{\bf Planck 18}} used by {\href{https://docs.astropy.org/en/latest/api/astropy.cosmology.realizations.Planck18.html}{\bf Astropy module}}.}
             \label{tab:cosmo}
           \end{table}
           \vspace{-1cm}
           

    #. For **FlatLambdaCDM** model we have considered that there is no curvature, :math:`\Omega_{k0} = 0`. In this, we have assumed that **matter and dark energy** are the only constituents of the Universe. Here we have taken the corresponding desity parameters :math:`\Omega_{m\displaystyle 0} = 0.30966, \Omega_{\Lambda\displaystyle 0} = 0.69034, \Omega_{\rm total\displaystyle 0} = \Omega_{m\displaystyle 0} + \Omega_{\Lambda\displaystyle 0} = 1`. And the present value of **Hubble constant** is :math:`H_{0} = 67.66\,\rm km/s/Mpc`.
    

    .. raw:: latex
    
        \clearpage
    """    
   
    nue = 1420.0        # nue = emitted frequency 1420 MHz, 21cm radiation
    z   = nue/nuc -1    # z   = redshift corresponding to observing frequency nuc
    
    print(f"--------------------------------------")
    
    if model == "FlatLambdaCDM":
        
        print(f"Cosmological Model: FlatLambdaCDM")
        cosmo = FlatLambdaCDM(H0 = 67.66 , Om0 = 0.30966, Ob0 = 0.04897) 
        # construct the FLAT LCDM Model with no curvature, no density parameter contribution from CMB, Neutrinos
        # only matter + dark energy 
    
        r   = cosmo.comoving_distance(z).value # comoving distance in Mpc unit at redshift z
        rp  = ((cosmo.hubble_distance.value)*(cosmo.inv_efunc(z)))*((1+z)**2)/nue 
        
    elif model == "Planck18":
        
        print(f"Cosmological Model: Planck18")
        r   = Planck18.comoving_distance(z).value # comoving distance in Mpc unit at redshift z
        rp  = ((Planck18.hubble_distance.value)*(Planck18.inv_efunc(z)))*((1+z)**2)/nue
        
        
    # rp = rprime = dr/dnu at nuc = (1+z)^2/nu_e times C/H(z)

    # cosmo.hubble_distance.value gives C/H_0 and cosmo.inv_efunc(z) gives 1/E(z) where H(z) = H_0 E(z)
    # multiplying them basically gives C/H(z) 
    # C is the speed of light in appropriate units
    
    
    print(f"Redshift          : z  = {z:.2f}")
    print(f"Comoving distance : r  = {r:.2f} Mpc")
    print(f"rprime            : rp = {rp:.2f} Mpc/MHz")
    print(f"Sep used          : NE = {NE}")

    fac = r/(rp*nuc)          # wedge boundary slope(fac): kpara_H = fac*kper

    vfac = rp*r*r*dnuc*(NE-1) # DCT factor in the FT, r^2r'\Delta\nu_c(N_E-1)
    # vfac, volume factor

    # component of \vec{k} perpendicular to line of sight.
    kper = lval/r             # k_\perp = \ell/r
    
    kpara = np.pi*np.arange(NE)/(rp*dnuc*(NE-1)) # k_{\parallel m} = m\pi/[r'\Delta\nu_c(N_E-1)]

    print(f"--------------------------------------")
    print(f"Wedge boundary slope (fac) : {fac:.3f}\
          \nDCT factor (vfac)          : {vfac:.3e}\
          \nkper.shape                 : {kper.shape}\
          \nkpara.shape                : {kpara.shape}")
    print(f"--------------------------------------")
    
    return r, rp, fac, vfac, kper, kpara

# A matrix need for cosine transform  
def calc_A(na, nb):
    r"""
    Parameters
    ----------
    na  : int
        Non negative integer greater than 2.
    nb  : int
        Non negative integer greater than 2.
        
    Returns
    -------
    A   : np.ndarray
        A 2D matrix of shape :math:`(na\times nb)`, with 
        
        .. raw:: latex
		
    		\begin{center}
       		\tbox{$A_{\rho\lambda} = B_{\rho\lambda}\,\cos\left(\dfrac{\pi\rho\lambda}{nb-1}\right)$}
       		\end{center}
        
        Where :math:`\rho = 0,1,\dots,(na-1)`, :math:`\lambda = 0,1,\dots,(nb-1)`
        
        .. raw:: latex
        
                \begin{center}
           	   	\tbox{$
                B_{\rho\lambda} =
                \begin{cases}
                	0.5 & \text{if } \lambda = 0 \text{ or } (nb-1) \\
                    1.0 & \text{otherwise}
                \end{cases}$}
                \end{center}
           
    Examples
    --------
    >>> print(pe.calc_A(2,2))
    [[ 0.5  0.5]
     [ 0.5 -0.5]]
    >>> print(np.round(pe.calc_A(3,3),3))
    [[ 0.5  1.   0.5]
     [ 0.5  0.  -0.5]
     [ 0.5 -1.   0.5]]
    >>> print(np.round(pe.calc_A(4,4),3))
    [[ 0.5  1.   1.   0.5]
     [ 0.5  0.5 -0.5 -0.5]
     [ 0.5 -0.5 -0.5  0.5]
     [ 0.5 -1.   1.  -0.5]]
    >>> print(np.round(pe.calc_A(2,4),3))
    [[ 0.5  1.   1.   0.5]
     [ 0.5  0.5 -0.5 -0.5]]

     .. raw:: latex

         \clearpage

    """  
    A = np.outer(np.arange(na), np.arange(nb))
    A = np.cos(np.pi*A/(nb-1.))
    
    A[:, 0] *= 0.5
    A[:,-1] *= 0.5
    
    return A

# Blackman-Nutall window    
def window(N):
    r"""
    Returns a 4 term **Blackman-Nuttall** window function values.
    
    .. raw:: latex

           \begin{center}
            \tbox{$F(a) = c_0 - c_1\,\cos\left(\dfrac{2\pi a}{2N-2}\right)+c_2\,\cos\left(\dfrac{4\pi a}{2N-2}\right)- c_3\,\cos\left(\dfrac{6\pi a}{2N-2}\right)\,\quad a = 0,1,\dots,2N-3$}
            \end{center}

    Where 
    
    .. math::
    
        c_0 = 0.3635819,c_1 = 0.4891775,c_2 = 0.1365995,c_3 = 0.0106411

    It has following properties :

    .. math::
    
        \boxed{F\Bigl[(N-1)+a\Bigr]=F\Bigl[(N-1)-a\Bigr]}
        
    and it's also a periodic function with period :math:`2N-2`, i.e.

    .. math::
    
        \boxed{F\Bigl[a+\gamma(2N-2)\Bigr] = F(a),\quad \text{\rm with } \gamma = \pm1,\pm2,\dots}

    :math:`a = 0,1,\dots,2N-3` corresponds to frequency separation values :math:`-(N-1)\Delta\nu_c,-(N-2)\Delta\nu_c,\dots,(N-2)\Delta\nu_c`. Precisely ,

    .. math::
    
        \boxed{\Delta\nu_a = (-N+a+1)\Delta\nu_c}
        
    And this function returns the window function :math:`\mathcal{W}_{\rm BN}(\Delta\nu_a)` with 
    
    .. math::
    
        a = N-1,N,\dots,2N-3,2N-2
    
    Parameters
    ----------
    N : int
        Number of frequency separations used (must be :math:`N\geq2`) to estimate the cylindrical power spectrum :math:`P(k_{\perp},k_{\parallel})`.

    Returns
    -------
    window_value : np.ndarray
        Blackmann-Nuttall window function array :math:`\mathcal{W}_{\rm BN}(\Delta\nu_n)\Bigr[\Delta\nu_n = n\Delta\nu_c,\\n = 0,1,\dots,N-1\Bigl]` of length :math:`N`, normalized to :math:`\Delta\nu = 0`. The elements corresponds to :math:`\Delta\nu_0 = 0,\Delta\nu_1 =\Delta\nu_c,\Delta\nu_2 =2\Delta\nu_c,\cdots,\Delta\nu_{N-1} =(N-1)\Delta\nu_c`.

    Examples
    --------
    >>> np.set_printoptions(precision = 4, suppress = True)
    >>> w = pe.window(N = 5) # Blackman Nutall window of length 5.
    >>> print(f"w = {w}")
    w = [1.     0.702  0.227  0.0252 0.0004]
    >>> w = pe.window(N = 7) # Blackman Nutall window of length 7.
    >>> print(f"w = {w}")
    w = [1.     0.8555 0.5292 0.227  0.0613 0.0082 0.0004]

    Notes
    -----
    While estimating the cylindrical power spectrum :math:`P(k_{\perp},k_{\parallel})` from MAPS :math:`C_{\ell}(\Delta\nu)`, the assumption is that MAPS is periodic with period :math:`(2N-2)\Delta\nu_c`. But in reality it is not. So at the band edges it suffers a discontinuity which creates **artefacts** in the estimated **PS**. So thereby it is multiplied by window function :math:`\mathcal{W}_{\rm BN}(\Delta\nu)`, effectively making the product :math:`\mathcal{W}_{\rm BN}(\Delta\nu)C_{\ell}(\Delta\nu)` periodic.       

    .. raw:: latex

        \begin{figure}[H]
        \centering
        \includegraphics[]{../../../../../Tutorials/others/window.pdf}
        \caption{Blackman-Nuttal window function for $N_E=668$ with $\Delta\nu_c = 0.04\,\text{\rm MHz}$.}
        \end{figure}  

        \noindent\rule{\linewidth}{0.4pt}

        \newpage
        
    """
    
    window_value = scipy.signal.windows.nuttall(2*N-2,sym = False)[:N][::-1] 
    
    # cl(\Delta\nu) is periodic with a period (2N-2)\Delta\nu_c, 
    # scipy.signal.windows.nuttall(2*N-2,sym = False) generates 4 term blackman nutall window
    # which is periodic with period (2N-2), sym = False means periodic
    # weights are 0.3635819, 0.4891775, 0.1365995, 0.0106411
    # we need window values for \Delta\nu = n\Delta\nu_c, n = 0,1,...,N-1 half of it
    # [:N] slicing gives window values \Delta\nu = n\Delta\nu_c, n = N-1,N-2,...,1,0 
    # [::-1] does the array reversing which gives us the correct values.

    ## this code is equaivalent to 
    ## c0, c1, c2, c3 = 0.3635819, 0.4891775, 0.1365995, 0.0106411
    ## M = 2*N-2
    ## a = np.arange(N-1,2*N-1)
    ## window_value = c0 - c1*np.cos(2*np.pi*a/M) + c2*np.cos(4*np.pi*a/M) -c3*np.cos(6*np.pi*a/M)
    
    return window_value

# Cylindrical PS using MLE
def func_pk(cl, w, covi, vfac):
    r"""
    For each annular bin in `uv` plane with effective multipole value :math:`\ell_a`, we have : 
    
    .. math::

        \mathbf{P_a} &= (\mathbb{A}^\dagger\mathbf{N}_a^{-1}\mathbb{A})^{-1} \mathbb{A}^\dagger\mathbf{N}_a^{-1} \times \mathbf{\left[\mathcal{W} C_{\ell_a}\right]}\\
        & = \mathbb{M}\times \mathbf{\left[\mathcal{W}_{BN}\,C_{\ell_a}\right]}
                     

    Where :math:`\mathbb{A}` matrix is constructed using **calc_A** function, :math:`\mathcal{W}_{BN}` is the window function. :math:`\mathbf{P_a}` is the cylindrical power spectrum value for fixed value of :math:`k_{\perp a} = \ell_a/r`. In marix notation we can write : 

    
    .. math::
    
        \begin{align*}
        \small
        \left[
        \begin{array}{c}
            P(k_{\perp a},k_{\parallel m = 0 })\\
            P(k_{\perp a},k_{\parallel m = 1 })\\
            \vdots\\
            P(k_{\perp a},k_{\parallel m = N_E-1 })\\
        \end{array}\right] = 
        \begin{bmatrix}
            \mathbb{M}_{00} & \mathbb{M}_{01} & \cdots & \mathbb{M}_{0,N_E-1}\\
            \mathbb{M}_{10} & \mathbb{M}_{11} & \cdots & \mathbb{M}_{1,N_E-1}\\
            \vdots & \vdots & \ddots & \vdots \\ 
            \mathbb{M}_{N_E-1,0} & \mathbb{M}_{N_E-1,1} & \cdots & \mathbb{M}_{N_E-1,N_E-1}\\
        \end{bmatrix} \times 
        \begin{bmatrix}
            \mathcal{W}_{\rm BN}(\Delta\nu_{n=0}) \times C_{\ell_a}(\Delta\nu_{n=0})\\
            \mathcal{W}_{\rm BN}(\Delta\nu_{n=1}) \times C_{\ell_a}(\Delta\nu_{n=1})\\
            \vdots\\
            \mathcal{W}_{\rm BN}(\Delta\nu_{n=N_E-1}) \times C_{\ell_a}(\Delta\nu_{n=N_E -1})\\
        \end{bmatrix} 
        \end{align*}


    
    Parameters
    ----------
    cl : np.ndarray
        Multi-angular power spectrum :math:`C_{\ell}(\Delta\nu)` (MAPS) array. Shape :math:`(\dots,len(lval),N\!E)`.
    w  : np.ndarray
        The window function normalized to :math:`\Delta\nu = 0`, which is used to avoid discontinuity at the band edges. Shape :math:`(N\!E,)`.
    covi : np.ndarray
        Inverse of diagonal elements of noise covariance matrix :math:`\mathbf{N}` estimated from muliple noise only simulations. Shape :math:`(....,len(lval),N\!E)`.
    vfac : int or float
        The volume factor :math:`r^2r'\Delta\nu_c(N\!E-1)` sitting in the FT of MAPS.

    Returns
    -------
    pk   : np.ndarray
        Cylindrical power spectrum :math:`P(k_{\perp},k_{\parallel})` array using MLE (**maximum likelyhood estimator**). Shape :math:`(....,len(lval),N\!E)`.


     .. raw:: latex
    
        \clearpage

    Examples
    --------
    >>> index = 7200
    >>> cl   = np.load(f'/home/cts23ph/git_package_myutils/clfuncs/cldnur_{index}.npy')[...,:NE]
    >>> cln  = np.load(f'/home/cts23ph/git_package_myutils/clfuncs/cldnur_noise_{index}.npy')[...,:NE]
    >>> dcln = np.std(cln, axis = 0)
    >>> covi = 1/dcln**2
    >>> # window for Fourier transform
    >>> w    = pe.window(NE) 
    >>> # MLE cylindrical PS P(kper, kpara)
    >>> pk   = pe.func_pk(cl, w, covi, vfac)
    >>> print(f'{cl.shape        = }\
    ...       \n{covi.shape      = }\
    ...       \n{w.shape         = }\
    ...       \n{pk.shape        = }')
    cl.shape        = (20, 668)      
    covi.shape      = (20, 668)      
    w.shape         = (668,)      
    pk.shape        = (20, 668)

    Notes
    -----

    Here we have assumed that the noise covariance matrix :math:`\mathbf{N}` is diagonal along frequency separation axis :math:`\Delta\nu` and for each annnular bin :math:`\ell_a` bin we have :

    .. math::

        \left(\mathbf{N_a}\right)_{mn} = \delta_{mn}\, \left[\delta C_{\ell_a}\left(\Delta\nu_n\right)\right]_{\rm noise}^2

    So each element of ``covi`` (the last axis) contains the value :math:`1/\left[\delta C_{\ell_a}\left(\Delta\nu_n\right)\right]_{\rm noise}^2`.

    The explicit form of the marix :math:`\mathbf{N_a}` is given as :

    .. math::

        \begin{bmatrix}
            \left[\delta C_{\ell_a}\left(\Delta\nu_{n=0}\right)\right]_{\rm noise}^2& 0 & \cdots & 0\\
            0 & \left[\delta C_{\ell_a}\left(\Delta\nu_{n=1}\right)\right]_{\rm noise}^2 & \cdots & 0\\
            \vdots & \vdots & \ddots & \vdots \\ 
            0 & 0 & \cdots & \left[\delta C_{\ell_a}\left(\Delta\nu_{n=N_E-1}\right)\right]_{\rm noise}^2\\
        \end{bmatrix}
 
    .. raw:: latex

        \begin{figure}[H]
        \centering
        \includegraphics[]{../../../../../Tutorials/others/cyl_ps_7200.pdf}
        \caption{Cylindrical Power Spectrum $P(k_\perp,k_\parallel)$.}
        \end{figure}  

        \newpage

        
    """


    # cl shape (....,nell,NE) 

    nell, NE = cl.shape[-2:]
    A  = calc_A(NE, NE)         # shape(NE,NE) 
    At = np.transpose(A)        # A^dagger  A†  
    pk = np.zeros_like(cl)
    
    ne = np.arange(NE)
    # loop over the ell bins
    for ii in range(nell):
        # vari = N⁻¹
        vari   = np.zeros(covi.shape[:-2]+(NE,NE))
        vari[...,ne,ne] = covi[...,ii,:] # use noise-variance. shape(...,NE,NE)
        
        X = np.linalg.inv(At@vari@A)@At@vari # (A†N⁻¹A)⁻¹ A†N⁻¹ # shape(...,NE,NE)
        pk[...,ii,:] = np.squeeze(X@(w[:,None]*cl[...,ii,:,None]))*vfac # window w shape (NE,), muliply it by window w

        # cl[...,ii,:,None] is to extend the dimension of cl
        # append axis to last so that matrix multiplication can be done and then using np.squeeze remove that extra dimension 
        
    return pk 

def X(pk, dpkn, flag_mask):
    r"""
    Parameters
    ----------
    pk  : np.ndarray
        Cylindrical power spectrum :math:`P(k_{\perp},k_{\parallel})` array (**2D or higher dimensional array**), Must be of shape :math:`[\dots,len(kper),len(kpara)]`.
    dpkn: np.ndarray
        Standard deviation of cylindrical power spectrum :math:`\delta P_N(k_{\perp},k_{\parallel}) = \sqrt{\langle\left[ P_N(k)\right]^2\rangle - \left\langle P_N(k) \right\rangle^2}`, estimated from noise only simulations. Where :math:`\langle\cdots\rangle` denotes average over realizations and :math:`P_N(k_{\perp},k_{\parallel})` is the cylindrical power spectrum obtained using noise only simulations. **Must have same shape as** ``pk``.
    flag_mask : np.ndarray
        A **2D** boolean mask of shape :math:`[len(kper),len(kpara)]`. If the element value is **1** then that :math:`(k_{\perp},k_{\parallel})` **mode is used**, if **0** then **rejected**. Should contain values **0 and 1 only**.

    Returns
    -------
    X : np.ndarray
        The values of variable :math:`X = \dfrac{P(k_{\perp},k_{\parallel})}{\delta P_N(k_{\perp},k_{\parallel})}` (**flattened array along last two dimensons**) corresponding to those :math:`(k_{\perp},k_{\parallel})` modes defined via **flag_mask** boolean array. Shape :math:`[\dots,modes]`. Where `modes` is the number of elements containing **1** in **flag_mask** array.
    mu_est : float  or np.ndarray
        Mean value of :math:`X` (:math:`\mu_{\rm Est}`) , estimated using :math:`(k_{\perp},k_{\parallel})` modes defined in **flag_mask** boolean array. Shape :math:`pk.shape[:-2]`. **last two axes are removed**.
    sigma_est : float  or np.ndarray
        Standard deviation of :math:`X` (:math:`\sigma_{\rm Est}`) , estimated using :math:`(k_{\perp},k_{\parallel})` modes defined in **flag_mask** boolean array. Same shape as ``mu_est``.

    Examples
    --------
    >>> # Noise Power Spectrum
    >>> # MLE cylindrical PS P(kper, kpara)       # noise
    >>> pkn   = pe.func_pk(cln, w, covi, vfac)    # noise ps
    >>> dpkn  = np.std(pkn, axis = 0)             # std noise of ps
    >>> dpkn *= (20)**2                           # scale the noise accordingly
    >>> # flag mask array
    >>> fm = np.load('/home/cts23ph/git_package_myutils/psfuncs/flag_mask.npy')
    >>> print(f'{pkn.shape       = }\
    ...       \n{dpkn.shape      = }\
    ...       \n{fm.shape        = }')
    pkn.shape       = (50, 20, 668)      
    dpkn.shape      = (20, 668)      
    fm.shape        = (20, 668)
    >>> X, mu, sigma = pe.X(pk, dpkn, fm) # X statistics
    >>> np.set_printoptions(precision = 3, suppress = True) 
    >>> print(f'{X.shape = }\
    ...        \nmu      = {mu}\
    ...        \nsigma   = {sigma}')
    X.shape = (503,)       
    mu      = 0.2797964155866535       
    sigma   = 1.2258385055783663

    .. raw:: latex

        \begin{figure}[H]
        \centering
        \includegraphics[]{../../../../../Tutorials/others/X_7200.pdf}
        \caption{X statistics. The magenta colored line shows student t fit.}
        \end{figure}  


    Notes
    -----
    Here is how you can make flag mask array.

    .. code-block:: python
    
        >> flag_mask = np.zeros((kper.size,kpara.size), dtype = 'int')
        >> print(flag_mask.shape)
        >> ks  = np.array([0.135, 0.228, 0.36, 0.5, 0.72, 0.8, 0.92, 1.09, 1.17, 1.39])
        >> ksv = np.array([0,0,1,2,2,2])
        >> for ii in range(len(ksv)):
        >>     for jj in range(ksv[ii],len(ksv)-1):
        >>         mask = (kpara>=ks[2*jj])*(kpara<=ks[2*jj+1])
        >>         flag_mask[ii,mask] = 1
        >> np.save('flag_mask.npy',flag_mask)   # save the array

    Where the modes used is tabulated below : 

    .. raw:: latex 

        \begin{table}
        \centering
        \setlength{\tabcolsep}{10pt}
        \setlength{\arrayrulewidth}{0.5pt}
        \renewcommand{\arraystretch}{1.4}
        \setlength{\doublerulesep}{1.5pt}  % <-- reduce space between double lines
        \resizebox{0.8\textwidth}{!}{
        \begin{tabular}{|c|c|c|c|c|c|}
        \hline
        $k_{\perp}\,\rm Mpc^{-1}$ & \multicolumn{5}{c|}{$k_{\parallel} \,\rm Mpc^{-1}$} \\
        \hline
        $0.007$ & $0.135-0.228$ & $0.360-0.499$ & $0.720-0.797$ & $0.921-1.095$ & $1.171-1.399$ \\\hline
        $0.015$ & $0.135-0.228$ & $0.360-0.499$ & $0.720-0.797$ & $0.921-1.095$ & $1.171-1.399$ \\\hline
        $0.022$ & $-$           & $0.360-0.499$ & $0.720-0.797$ & $0.921-1.095$ & $1.171-1.399$ \\\hline
        $0.029$ & $-$           & $-$           & $0.720-0.797$ & $0.921-1.095$ & $1.171-1.399$ \\\hline
        $0.038$ & $-$           & $-$           & $0.720-0.797$ & $0.921-1.095$ & $1.171-1.399$ \\\hline
        $0.045$ & $-$           & $-$           & $0.720-0.797$ & $0.921-1.095$ & $1.171-1.399$ \\\hline
        \end{tabular}}
        \caption{$k_{\parallel}$ ranges corresponding to different $k_{\perp}$ values that have been used to estimate binned PS and X statistics.}
        \label{table:mask}
        \end{table}

    """
    
    X = pk/dpkn # X statistics
    
    X = X[...,flag_mask == 1]         # extract only the modes that you want
    
    mu_est    = np.mean(X, axis = -1)
    sigma_est = np.std (X, axis = -1) # use axis = -1 to vectorize for input pk higher dimensions
    
    return X, mu_est, sigma_est

# Estimating spherical ps p(k) from P(kperp, kpara) using inverse variance weighted average
def binned_pk(kper, kpara, pk, dpk, NBin, flag_mask):
    r"""
    Using the mode information contained in **flag_mask**, first it computes all the :math:`k\,(={ \sqrt{k_{\perp}^2+k_{\parallel}^2}})` values available and bins the :math:`(k_{\perp},k_{\parallel})` space into :math:`N\!Bin` no of **logarithmic bins**. And for each bin :math:`k_i`, it computes binned power spectrum values :math:`P(k_i)` and the corresponding error :math:`\delta P(k_i)` using **the minimum variance estimator**. See notes for more details.
    
    .. math::
    
        P(k_i) &= \frac{\sum_{(k_\perp,k_\parallel)\in\, i}\,\mathbf{w}(k_{\perp},k_{\parallel})\,P(k_{\perp},k_{\parallel})}{\sum_{(k_\perp,k_\parallel)\in\, i}\,\mathbf{w}(k_{\perp},k_{\parallel})}\\[0.5em]
        \delta P(k_i) &= \left[{\sum_{(k_\perp,k_\parallel)\in\, i}\,\mathbf{w}(k_{\perp},k_{\parallel})}\right]^{-1/2}\\[0.5em]
        k_i &= \mathbf{exp}\left[\frac{\sum_{(k_\perp,k_\parallel)\in\, i}\,\mathbf{{{w}}}(k_{\perp},k_{\parallel})\,ln(k)}{\sum_{(k_\perp,k_\parallel)\in\, i}\,\mathbf{{{w}}}(k_{\perp},k_{\parallel})}\right];
        
    
    Where :math:`\mathbf{w}(k_{\perp},k_{\parallel})` is the weight assigned to that particular :math:`(k_{\perp},k_{\parallel})` mode and by  :math:`(\sum_{k_\perp,k_\parallel})\in \, i`, we denote, the sum is being performed using all the :math:`(k_{\perp},k_{\parallel})` modes which lies in :math:`i^{\rm th}` bin. 
    
    Parameters
    ----------
    kper : np.ndarray
        **1d array** containing the values of wave-vector :math:`\vec{\mathbf{k}}` **perpendicular** to the line of sight **(LOS)**. Denoted by sign :math:`k_{\perp}`.
    kpara : np.ndarray
        **1d array** containing the values of wave-vector :math:`\vec{\mathbf{k}}` **along** to the line of sight **(LOS)**. Denoted by sign :math:`k_{\parallel}`.
    pk : np.ndarray
        Cylindrical power spectrum :math:`P(k_{\perp},k_{\parallel})` array (**2D or higher dimensional array**), Must be of shape :math:`[\dots,len(kper),len(kpara)]`.
    dpk : np.ndarray
        Standard deviation of cylindrical noised power spectrum (scaled) :math:`\delta P_N^{True}(k_{\perp},k_{\parallel}) = \sigma_{\rm Est}\,\delta P_N(k_{\perp},k_{\parallel})` array (**2D or higher dimensional array**), Must be of shape :math:`[\dots,len(kper),len(kpara)]`, same shape as **pk**.       
    NBin : int
        No of bins the whole :math:`(k_{\perp},k_{\parallel})` space has been divided into using only the modes defined in **flag_mask** mode info. 
        
    flag_mask : np.ndarray
        A **2D** boolean mask of shape :math:`[len(kper),len(kpar–a)]`. If the element value is **1** then that :math:`(k_{\perp},k_{\parallel})` **mode is used**, if **0** then **rejected**. Should contain values **0 and 1 only**.
        
    Returns
    -------
    keff: np.ndarray
        The binned :math:`k` values (**1d array**). Shape :math:`({nzN\!Bin},)`.
    ppk : np.ndarray
        Binned power spectrum :math:`P(k)` values (**Spherical PS**), (**n-dimensional array**). Shape ::math:`(\dots,{nzN\!Bin})`.
    dppk : np.ndarray
        :math:`1\sigma` error in estimating spherical ps, :math:`\delta P(k)` values, (**n-dimensional array**). Shape :math:`(\dots,{nzN\!Bin})`.
    
      
    Examples
    --------
    >>> NBin = 5
    >>> kk, ppk, dppk = pe.binned_pk(kper, kpara, pk, sigma*dpkn,
    ...                              NBin, fm)
    Bin count   : [ 26   3  60  96 318]
    >>> # kk   = binned k values
    >>> # ppk  = binned P(k) values
    >>> # dppk = 1 sigma error in estimating binned P(k) values
    >>> print(f'{kk.shape    = }\
    ...       \n{ppk.shape   = }\
    ...       \n{dppk.shape  = }')
    kk.shape    = (5,)      
    ppk.shape   = (5,)      
    dppk.shape  = (5,)
    >>> NBin = 8
    >>> kk, ppk, dppk = pe.binned_pk(kper, kpara, pk, sigma*dpkn,
    ...                              NBin, fm)
    Bin count   : [ 15  11   0  45  18  72 150 192]
    >>> # kk   = binned k values
    >>> # ppk  = binned P(k) values
    >>> # dppk = 1 sigma error in estimating binned P(k) values
    >>> print(f'{kk.shape    = }\
    ...       \n{ppk.shape   = }\
    ...       \n{dppk.shape  = }')
    kk.shape    = (7,)      
    ppk.shape   = (7,)      
    dppk.shape  = (7,) 

    Notes
    -----
    #. If no :math:`(k_{\perp},k_{\parallel})` mode contibutes to a bin, then we reject those bins in the end.
    #. :math:`{nzN\!Bin}` is the altered value of number of bins :math:`N\!Bin`. This happens as it might happen number of modes contuributes is **zero**.
    #. The binning is performed via the **minimum-variance estimator** where we have chosen the weights as : 
    
    .. math::
    
        \mathbf{w}(k_{\perp},k_{\parallel}) = \frac{1}{\left[\delta P_N^{True}(k_{\perp},k_{\parallel})\right]^2} = \frac{1}{\left[\sigma_{\rm Est}\times\delta P_N(k_{\perp},k_{\parallel})\right]^2}

    .. raw:: latex

        \clearpage
        
    """
    #NBin  # no of spherial bins in the kper kpara space
    k = np.sqrt(kper[:,None]**2+kpara[None,:]**2) 
    # all k values available, shape (len(kper), len(kpara))
    
    km = k[flag_mask == 1]          
    # extract the k modes that you want # discard the masked modes
    # flattened k values corresponding to flag_mask masking. shape (modes,) 
    # where modes = no of elements contains 1 in that mask, modes = len(np.where(flag_mask.flatten() == 1)[0])
    
    kmax = 1.1*km.max() # max value of k 
    kmin = km.min() # min value of k 
        
    # the last one liner is equivalent to 
    dd = np.log(kmax/kmin)/NBin # spacing between two adjacent bins in log space
    k_bin_edges = kmin*np.exp(np.arange(NBin+1)*dd)

    # get count of 21 cm modes in each bin
    count = (scipy.stats.binned_statistic(km, km, statistic = 'count', bins = k_bin_edges)[0]).astype('int')
    print(f"Bin count   : {count}\n")

    # find the number of bins in which the unmasked modes count are non zero and their indices
    nz_bin_indices = np.arange(NBin)[count != 0]  # bin indices for those count is non zero
    nzNBin = len(nz_bin_indices)                  # non zero bin count
    
    #keff = np.exp(scipy.stats.binned_statistic(km, np.log(km) , statistic = 'mean', bins = k_bin_edges)[0])[nz_bin_indices]
    # logarithmic binning, effective value of k in each bins
    # np.exp(scipy.stats.binned_statistic(km, np.log(km) , statistic = 'mean', bins = k_bin_edges)[0]) will return nan for zero count bins
    # slice with nz_bin_indices to remove nans
    # effective values of k in those bins
    
    weights = 1/(dpk)**2 # inverse variance weightage
        
    pk_modes = pk[...,flag_mask == 1]  # extract the same way pk also (cylindrical PS) 
    # pk shape shape (..., len(kper), len(kpara))
    # pk_modes shape (..., modes)

    # apply weights
    weights_modes = weights[...,flag_mask == 1] # weight value for wanted modes
    # weights       shape  (..., len(kper), len(kpara))
    # weights_modes shape  (..., modes)

    wkpk_modes = pk_modes*weights_modes # wk times pk
    
    sum_wkpk = (scipy.stats.binned_statistic(km, wkpk_modes.reshape(-1, km.size), statistic = 'sum', bins = k_bin_edges)[0]).reshape(wkpk_modes.shape[:-1]+(-1,))
    #  wkpk_modes.reshape(-1, km.size) is done so that the binned statistics can take flattened higher dimensional inputs. 
    # .reshape(wkpk_modes.shape[:-1]+(-1,)) to reshape it back to the original shape.

    sum_wk = (scipy.stats.binned_statistic(km, weights_modes.reshape(-1, km.size), statistic = 'sum', bins = k_bin_edges)[0]).reshape(weights_modes.shape[:-1]+(-1,))
    #  weights_modes.reshape(-1, km.size) is done so that the binned statistics can take flattened higher dimensional inputs. 
    # .reshape(weights_modes.shape\noindent\rule{\linewidth}{0.4pt}[:-1]+(-1,)) to reshape it back to the original shape.

    wkk_modes = np.log(km)*weights_modes

    sum_wkk = (scipy.stats.binned_statistic(km, wkk_modes.reshape(-1, km.size), statistic = 'sum', bins = k_bin_edges)[0]).reshape(wkk_modes.shape[:-1]+(-1,))

    keff = np.exp(sum_wkk/sum_wk)
    
    ppk  = sum_wkpk/sum_wk        # binned pk value
    dppk = np.sqrt(1/sum_wk)      # binned pk error value


    # keff is the effective k  value for that bin shape(nzNBin,)
    # ppk  is the effective pk value for that bin shape(...,nzNBin)
    # dppk is the error in estimating pk value for that bin shape(...,nzNBin)
    
    return keff[...,nz_bin_indices], ppk[...,nz_bin_indices], dppk[...,nz_bin_indices]

def func_dT(kk , ppk, dppk): # dppk binned delta P_N^{true}(k) = \sigma_est \times binned delta P_N(k)
    r"""
    Given the binned :math:`k` values, binned power spectrum :math:`P(k)` and it's :math:`1\sigma` error :math:`\delta P(k)`, it gives the **dimensionless power spectrum** :math:`\Delta^2(k)` and it's :math:`2\sigma` uncertainty, signal to noise ratio **SNR** and the corresponding **upper limit** :math:`\Delta^2_{\text{UL}}(k)`.

    Parameters
    ----------
    kk  : np.ndarray
        The binned :math:`k` values (**1d array**). Shape :math:`(len(kk),)`.
    ppk : np.ndarray
        Binned power spectrum :math:`P(k)` values, (**n-dimensional array**). Shape  :math:`(\dots,len(kk))`.
    dppk: np.ndarray
        :math:`1\sigma` error in estimating spherical ps, :math:`\delta P(k)` values, (**n-dimensional array**). Same shape as **ppk**.

    Returns
    -------
    dk2 : np.ndarray
        Dimensionless power spectrum :math:`\Delta^2(k)`, (**n-dimensional array**). Same shape as **pkk**.
    dpk2: np.ndarray
        Statistical (:math:`2\sigma`) error in estimating dimensionless power spectrum :math:`\Delta^2(k)`, (**n-dimensional array**).Same shape as **pkk**.
    snr : np.ndarray
        Signal to noise ratio. (**n-dimensional array**). Same shape as **pkk**.
    ul  : np.ndarray
        Upper limit of dimensionless power spectrum :math:`\Delta^2_{\text{UL}}(k)`, (**n-dimensional array**). Same shape as **pkk**.

        They are defined as follows :

        .. math::

            \setlength{\fboxrule}{0.7pt}
            \setlength{\fboxsep}{12pt}     % EXTRA padding
            \fcolorbox{black!0}{gray!0}{$
            \begin{aligned}
            \Delta^2(k)              & = \frac{k^3P(k)}{2\pi^2}\\[0.5em]
            \sigma(k)                & = \frac{k^3\delta P(k)}{2\pi^2}\\[0.5em]
            \textbf{\textsc{SNR}}    & = \Delta^2(k)/\sigma(k) = P(k)/\delta P(k)\\[0.5em]
            \Delta^2_{\text{UL}}(k)  & = 
            \begin{cases}
              \Delta^2(k)+2\sigma(k) & \text{if } \Delta^2(k) > 0 \\
              2\sigma(k)             & \text{otherwise}
           \end{cases}
           \end{aligned}$}

    .. raw:: latex
    
        \newpage

    
    
    Examples
    --------
    >>> dk2, dpk2, snr, ul = pe.func_dT(kk, ppk, dppk)
    >>> print(f'{dk2.shape   = }\
    ...       \n{dpk2.shape  = }\
    ...       \n{snr.shape   = }\
    ...       \n{ul.shape    = }')
    dk2.shape   = (7,)      
    dpk2.shape  = (7,)      
    snr.shape   = (7,)      
    ul.shape    = (7,)

    .. raw:: latex

        \begin{figure}[H]
        \centering
        \includegraphics[]{../../../../../Tutorials/others/spheical_ps_7200.pdf}
        \caption{Spherical Power Spectrum $P(k)$.}
        \end{figure}  

    """
    
    # shape of kk is the same as the last axis shape of ppk and dppk, so that we can vectorize the operations.

    dk2  = (ppk*(kk)**3)/(2.0*np.pi**2)         # Dimensionless PS
    dpk2 = (2.0*dppk*(kk)**3)/(2.0*np.pi**2)    # 2\sigma uncertainty
    
    mask = dk2<0            # identify for which bins, Spherical PS is negative
    
    ul   = abs(dk2) + dpk2  # upper limit of \Delta^2(k)
    ul[mask] = dpk2[mask]   # if \Delta^2(k) is negative, noise determines the limit. (Hera Collab. 2020)
    
    snr  = ppk/dppk         # signal to noise ratio
    
    return dk2, dpk2, snr, ul

print(f"Imported psfuncs,   Import Time : {datetime.now().strftime('%d/%m/%y | %I:%M:%S %p')}")
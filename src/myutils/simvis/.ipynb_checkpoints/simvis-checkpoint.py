r"""
All sky simulations of Gaussian random sky signal and corresponding Visibilities
================================================================================

**For a given Angular Power Spectrum [APS, in** :math:`\text{mK}^2` **unit ] of brightness temperature fluctuations or Power Spectrum [PS, in**   :math:`\text{mK}^2\;\text{Mpc}^3` **unit ] and an input UVFITS file, this outputs a new UVFITS file with simulated visibilities.**

================

**Our aim is to simulate the visibilities for a given telescope and for either a given Angular Power Spectrum  or a given Power Spectrum.**

For given Angular Power Spectrum :math:`C_{\ell}`, we generate muliple realizations of the sky signal at the centered frequency :math:`\nu_c`, and for a given Power Spectrum :math:`P(k)`, we generate 3D sky signal for all frequency channels (frequency channel information taken from input ``UVFITS`` file). Afterwards by this simulated sky signal, we simulate the corresponding visibilities.

Visibilities that are simulated are in Jy(Jansky) unit. The visibility at a baseline :math:`\vec{\textbf{U}}` is given by:

.. math::

    \mathcal{V}(\vec{\textbf{U}},\nu)=Q_\nu \int_{\rm UH}\ d\Omega_{\hat{n}}\ T(\hat{\mathbf{n}},\nu)\,{A}(\Delta\hat{\mathbf{n}},\nu)\ \textbf{exp}\left[{2\pi \text{i}\vec{\textbf{U}}\cdot\Delta\hat{\mathbf{n}}}\right]

- :math:`T(\hat{\mathbf{n}},\nu)` is the surface brightness temperature field.
- :math:`{A}(\Delta\hat{\mathbf{n}},\nu)` is the primary beam pattern of the telescope.
- :math:`\textbf{exp}\left[{2\pi \text{i}\vec{\textbf{U}}\cdot\Delta\hat{\mathbf{n}}}\right]` is the phase factor.

In ``Healpix`` we discretize the sky and the integral becomes summation. But the integral is restricted only to  upper hemisphere.


:strong:`The discritized version is given as`:

.. math::

    \mathcal{V}(\alpha_{p},\vec{\textbf{U}},\nu) = Q_{\nu}\ \Delta\Omega_{pix}\ \sum_{q} \ T(\hat{\mathbf{n}}_{q},\nu) \ {A}(\Delta\hat{\mathbf{n}}_{q},\nu) \ \textbf{exp}\left[{2\pi \text{i}\vec{\textbf{U}}\cdot\Delta\hat{\mathbf{n}}_{q}}\right]
    
    
where :math:`\alpha_p` is the pointing RA. For more details, `see Section (2) <https://arxiv.org/pdf/2212.01251>`_.

Also,

.. math::
    
    Q_{\nu}  &=2k_B /\lambda^2
    
    \Delta\hat{\mathbf{n}}&= \hat{\mathbf{n}}- \hat{\mathbf{p}}
    
    \vec{\textbf{U}} &= u\;{\hat{\mathbf{e}}_ 1 (\alpha_{p})} + v \;{\hat{\mathbf{e}}_ 2 (\alpha_{p})} + w \;{\hat{\mathbf{e}}_3(\alpha_{p})}
    
:math:`\hat{\mathbf{e}}_1,\hat{\mathbf{e}}_2,\hat{\mathbf{e}}_3` are the **basis** vectors defined in the local tangent plane. :math:`\hat{\mathbf{e}}_1` is along **east**, :math:`\hat{\mathbf{e}}_2` is along **north** and :math:`\hat{\mathbf{e}}_3` is **vertically overhead, i.e. zenith pointed**.

The **basis** vectors :math:`\hat{\mathbf{e}}_1,\hat{\mathbf{e}}_2,\hat{\mathbf{e}}_3` are dependent on pointing **RA** :math:`\alpha_p` and **declination(DEC)** :math:`\delta_p`. Components of basis vectors in the cartesian system :math:`xyz` is given by:

.. math::

    \hat{\mathbf{e}}_1&=[-\sin{\alpha_p},\cos{\alpha_p},0]\\
    \hat{\mathbf{e}}_2&=[-\cos{\alpha_p}\sin{\delta_p},-\sin{\alpha_p}\sin{\delta_p},\cos{\delta_p}]\\
    \hat{\mathbf{e}}_3&=[\cos{\alpha_p}\cos{\delta_p},\sin{\alpha_p}\cos{\delta_p},\sin{\delta_p}]


Where :math:`\Delta\Omega_{pix}` refers to the solid angle subtended by each simulation pixel and :math:`\hat{\mathbf{p}}` refers to the pointing direction of the antenna (here zenith pointed), and hence :math:`\hat{\mathbf{p}}=\hat{\mathbf{e}}_3`.

Notes
-----
In the simulation of visibilities for 3D sky signal, we have not incorporated the fact that the baselines :math:`\vec{\textbf{U}}` and antenna beam pattern :math:`{A}(\Delta\hat{\mathbf{n}},\nu)` changes as we vary the frequency. We assume only the brightness temperature field :math:`T(\hat{\mathbf{n}},\nu)` changes with frequency.

To run:
=======

.. code-block:: python

    >> import myutils.simvis.simvis as sv
    >> # skysimtype = '2D' for 2D sky signal (multiple realizations)
    >> # skysimtype = '3D' for 3D sky signal 
    >> sv.sim_vis(infits, outfits, skysimtype)
    >> # If you want the seed to be, say seed = 5, run:
    >> sv.sim_vis(infits, outfits, skysimtype, seed = 5)  # seed = None for random.
    >> # for 2D sky signal
    >> sv.sim_vis(infits, outfits, skysimtype = '2D', seed)
    >> # for 3D sky signal
    >> sv.sim_vis(infits, outfits, skysimtype = '3D',seed)



===============

**In generating 2D sky signal (for multiple realizations)**, for each realization of GRF, a seed is given. By default seed is set to **None**.

If you are doing :math:`N_{\rm realizations}`, in each realization seed is incremented by 1 w.r.t. previous value. For example, say you want to generate 10 sky maps, i.e.  :math:`N_{\rm realizations}=10`, and you gave seed = 5(say). Then while doing :math:`1^{\text{st}}` realization seed will be 5, in next realization seed will increment by 1 i.e. 6 and so on.

===============

**For 2D sky signal case:**

The entire code works at a single frequency, the centered frequency :math:`\nu_c`, which it reads from the input UVFITS file. It then simulates the visibilities at a RA,DEC which is read from input UVFITS file.

- Mutiple realizations (number of realizations :math:`N_{\rm realizations}`) are written along the frequency axis of output FITS.

- :math:`N_{\rm realizations}` cannot exceed :math:`N_{\rm channels}` the number of channels in the original UVFITS.

- Channels :math:`N_{\rm realizations}` +1 to :math:`N_{\rm channels}` of are left untouched in the output FITS file.

- The maximum value of angular multiplole :math:`\ell_{max}` is by default :math:`(3\times nside)-1`.

.. raw:: latex

    \newpage
    
**For 3D sky signal case**:

- Only the sky signal is frequency dependent.
- The sky signal and visibilities are being simulated for all the channels present in the input ``UVFITS`` file.
- Frequency dependent antenna beam patten and baselines are not incorporated.

================

If you want to do some operations with your data in uvinfits file but do not wish to write it in a uvoutfits file you can call the function ``sim_vis_hdul``. And it will update the data in the hdulist.

``For example run :``


.. code-block:: python
    
        >> with fits.open(uvinfits) as hdulist:
        >>      sv.sim_vis_hdul(hdulist, skysimtype, seed )
        >> # put your operations or code function here that will act on hdulist.
        >> # if you want to save the data use :
        >> hdulist.writeto(uvoutfits,overwrite = True)

===============

**Angular Power Spectrum Function**
-----------------------------------

If you want to give your own APS function as in the input please make the APS function first and make sure it has only :math:`\ell` dependence. Say your APS function is myaps(l) and you want to generate the GRF by this APS.
To do this run:

 .. code-block:: python
    
        >> import numpy as np
        >> import myutils.simvis.simvis as sv
        >> def myaps(l):
        >>     Amp  = 50   # in mK^2 unit
        >>     beta = -1   # power index
        >>     return Amp * np.power(l, beta)
        >
        >> sv.sim_vis(infits, outfits, skysimtype = '2D' ,
                                seed = 5, apsfunc = myaps )  
        

The pre-defined APS function ``aps`` has the form of 

.. math::
        
            C_{\ell}=A\, \left(\frac{\ell}{\ell_0}\right)^{\beta}
    

A defines the amplitude of the APS is in the unit of :math:`\text{mK}^2`. :math:`\beta` here is the power-law index. :math:`C_{\ell}` has the unit of  (:math:`\text{mK}^2`). By default set to :math:`A = 100\;\text{mK}^2`, :math:`\ell_0=1`, :math:`\beta` = -2. (See more in module functions).

========================

**Power Spectrum Function**
---------------------------

Power spectrum function ``ps`` here is defined as :

.. math:: 
    \begin{align}
    P^{M}(k) = A \left(\dfrac{k}{k_0}\right)^{s}
    \end{align}

with :math:`A=1, k_0=1, s=-2` (Default set). :math:`A` has the unit of :math:`\text{mK}^2\;\text{Mpc}^3`. :math:`k, k_0` are in :math:`\text{Mpc}^{-1}` unit.

We generate 3D sky signal (temperature field) across all the given frequency channels by :

.. math::

    \begin{align}
    \boxed{T(\hat{\mathbf{n}}, \nu) = \sum_{\ell, m} a_{\ell m}(\nu) \, Y_{\ell m}(\hat{\mathbf{n}})} 
    \end{align}
    

where we generate :math:`a_{\ell m}` coefficients at :math:`n^{\text{th}}` frequency channel by :

.. math::

    \begin{align}
    a_{\ell m}(\nu_n) = \sum_{q=0}^{N_c- 1} \left[ 
    \sqrt{\dfrac{P^M(k_\perp, k_{\parallel q})}{2N_c\;\Delta{\nu_c }\;r^2\;r'}} 
    \left({\hat{\mathbf{x}}_q + i \hat{\mathbf{y}}_q} \right)
    \right] {\textbf{exp}}\left[{\dfrac{2\pi i n q}{N_c}}\right]
    \end{align}
    
**Where** -

#. :math:`k_{\perp} = \ell/r` and :math:`k_{\parallel}` are the components of comoving wave vector :math:`\vec{k}` which is perpendicular and parallel (along) to the line of sight respectively.
#. :math:`r` is the comoving distance at centered frequency :math:`\nu_c`.
#. :math:`r' = \dfrac{dr}{d\nu}` evaluated at :math:`\nu_c`.
#. :math:`N_c` is the number of frequency channels.
#. :math:`\Delta{\nu_c }` is the frequency separation between two consecutive frequency channels.
#. :math:`\nu_n` refers to n-th frequency channel. So by convention :math:`\nu_0, \nu_{N_c-1}` will be the frequencies of first and last frequency channel. 
#. :math:`\hat{\mathbf{x}}_q,\hat{\mathbf{y}}_q \sim \mathcal{N}(0,1)` are independent Gaussian random variables of unit variance  and by equating :math:`a_{\ell m}(\nu_n)`'s, we generate :math:`T(\hat{\mathbf{n}}, \nu_n)` by Eq. (1.2).

Notes
-----

As temperature field is a real field, we must remember that this follows a property: 

.. math::

    \boxed{a_{\ell, m} = (-1)^{m} [a_{\ell, -m}]^{*}}
    
:math:`\text{where} * \text{means complex conjugation.}` From this proprety we have if :math:`m=0` then we have :math:`a_{\ell, m=0} = [a_{\ell, m=0}]^{*}`, meaning :math:`a_{\ell, m=0}` must be real.

So for :math:`m=0` case we take the real part of the :math:`a_{\ell, m}` generated by Eq. (1.3), i.e. 

.. math::

    \begin{align}
    a_{\ell}^{m=0}(\nu_n) ={\mathcal Re}\left\{ \sum_{q=0}^{N_c- 1} \left[ 
    \sqrt{\dfrac{P^M(k_\perp, k_{\parallel q})}{2N_c\;\Delta{\nu_c }\;r^2\;r'}} 
    \left({\hat{\mathbf{x}}_q + i \hat{\mathbf{y}}_q} \right)
    \right] {\textbf{exp}}\left[{\dfrac{2\pi i n q}{N_c}}\right]\right\}
    \end{align}

=========================

We have the relation between co-moving distance (:math:`r`) to the red-shift (:math:`z`) given by:

 .. math::

     r_{\nu}=\int_{0}^{z} \dfrac{c\;dz'}{H(z')} \implies \dfrac{\partial r_{\nu}}{\partial z} = \dfrac{c}{H(z)}

Where :math:`H(z)` is the Hubble parameter at red-shift :math:`z` given by (In Flat :math:`\Lambda\text{CDM}` model):

.. math::

    H(z)=H_0\left[\Omega_{m0}\;(1+z)^3+\Omega_{\Lambda0}\right]^{1/2}
    

We define :

.. math::

    r'_{\nu}&=\dfrac{\partial r_{\nu}}{\partial \nu}\\
            &= \dfrac{\partial r_{\nu}}{\partial z}\cdot\dfrac{\partial z}{\partial \nu}\\
            & = -\dfrac{\nu_0}{\nu^2}\cdot \dfrac{c}{H(z)}\\
            & = -\dfrac{(1+z)^2}{\nu_0} \cdot \dfrac{c}{H(z)}

Where we have used :math:`\nu=\nu_0/(1+z)`. Implying :math:`\dfrac{\partial z}{\partial \nu} = -\dfrac{\nu_0}{\nu^2}=-\dfrac{(1+z)^2}{\nu_0}`

:math:`\nu_0 = 1420\; \mathrm{MHz}`. We take only the absolute value :math:`|r'_{\nu}|=\dfrac{(1+z)^2}{\nu_0} \cdot \dfrac{c}{H(z)}`.

**The cosmological parameters are taken from** `Planck 18 Astropy <https://docs.astropy.org/en/latest/api/astropy.cosmology.realizations.Planck18.html>`_ , `FlatLambdaCDM <https://docs.astropy.org/en/latest/api/astropy.cosmology.FlatLambdaCDM.html>`_.

Notes
-----
**To generate 3D sky signal**, needed parameters :math:`\textbf{nuc, nc, dnu}`, rp is :math:`r'`, nc is the no of channels :math:`N_c`, dnu is channel width :math:`\Delta\nu_c`, nuc is the centered frequnency :math:`\nu_c`.


If you want to give your own power spectrum function (make sure it is vectorized)
    
    .. code-block:: python
    
        >> import numpy as np
        >> import builtins as blt
        >> import myutils.simvis.simvis as sv
        >> # make sure all are float.
        >> blt.A  =  1.0
        >> blt.s  = -3.0
        >> blt.k0 =  1.0
        >> @np.vectorize
        >> def mypkfunc(k):
        >>     if k == 0:
        >>         return 0.0
        >>     else:
        >>         return  A*(k/k0)**(s)    
        >>
        >> sv.sim_vis(infits, outfits, skysimtype = '3D', 
                            seed = 5, psfunc = mypkfunc)  


Notes
------

- ``APS`` function generates ``2D sky signal`` at centered frequency :math:`\nu_c`.
- ``PS`` function generates ``3D sky signal`` for all frequency channels available in the **fits file**.


================


Important point
---------------

Before you can simulate, you need to first set some parameters related to simulation.

- ``nside`` : Healpix parameter that sets the resolution.

- ``Nrea`` : No of realizations of sky (:math:`N_{\rm realizations}`).

- ``chunk_size`` : Number of baselines for which visibilities are being simulated at one step.

``chunk_size`` is a parameter which can be varied as per system requirements. If you have higher ram say 32 GB use it as 300. For 16 you can use 200 and and for less use 100. This number is for ``nside`` = 512 and ``Nrea`` = 100. As in a part of calculations a matrix comes which has the dimensions of  :math:`6\text{nside}^2 \times \text{chunk\_size}` with each entry represented by 128 bit complex number. So it takes memory to store it in RAM. This should not be more than RAM capacity. If you have more RAM you can't increase the ``chunk_size`` indefinitely as matrix operations will take much time to do the computation. I found, for ``nside`` = 512 and ``Nrea`` = 100 , ``chunk_size = 300-400`` (for 32 GB device) is efficient.

**To set them, follow this :**

.. code-block:: python

    >> import myutils.simvis.simvis as sv
    >> import builtins as blt
    >>
    >> # Set simulation parameters
    >> blt.nside = 512        # nside parameter
    >> blt.Nrea  = 100        # no of realizations
    >> blt.chunk_size = 400   # chunk_size, no of baselines in a single step

Overview
========

- First we generate the **Gaussian Random Field** :math:`T(\hat{\mathbf{n}},\nu)`  (temperature field in unit of mK) from  **Angular Power Spectrum** or **Power Spectrum**.

- Calculate the **primary beam pattern** :math:`{A}(\Delta\hat{\mathbf{n}},\nu)` for the telescope. 

- Calculate the components of :math:`\hat{\mathbf{n}}` along the basis vectors :math:`\hat{\mathbf{e}}_1,\hat{\mathbf{e}}_2,\hat{\mathbf{e}}_3`. This will help us to determine the dot product :math:`\vec{\textbf{U}}\cdot\Delta\hat{\mathbf{n}}`.

- Now we calculate the **phase factor** :math:`\textbf{exp}\left[{2\pi \text{i}\vec{\textbf{U}}\cdot\Delta\hat{\mathbf{n}}}\right]`.

- Multiply the **GRF and PB** first and then multiply by phase factor, sum over the pixels. The whole thing is `done by matrix multiplication.`


- The visibility is simulated.


.. raw:: latex

    \clearpage

Module Functions
================
"""

#Import Necessary Libraries
import numpy as np
import healpy as hp
import builtins as blt
import time
import os
import scipy as sp
import numexpr as ne
from datetime import datetime
from astropy.table import Table
from astropy.io import fits
from astropy.cosmology import Planck18 as cosmo # cosmological parameters , find comvoing distance

#------------------- Generate Multiple Realizations of GRF -----------------
def aps(l):  
    r"""
    This is the input model Angular Power Spectrum(APS) function.
    
    .. math::
        
            C_{\ell}=A\, \left(\frac{\ell}{\ell_0}\right)^{\beta}
            
    A defines the amplitude of the APS is in the unit of :math:`\text{mK}^2`. :math:`\beta` here is the power-law index. :math:`C_{\ell}` has the unit of  (:math:`\text{mK}^2`). By default set to :math:`A = 100\;\text{mK}^2`, :math:`\ell_0=1`, :math:`\beta` = -2.

    Parameters
    ----------
    l : float or np.ndarray
        The angular multipole :math:`{\ell}` value(s).
        
    Returns
    -------
    float or np.ndarray 
        The angular power spectrum value for the desired multipole(s) :math:`{\ell}`.
           
    """
    Amp = 100  # in mK^2 unit
    beta = -2  # power index
    return Amp * np.power(l, beta)
    
# UAPS (Unit Angular Power Spectrum) Function
def uaps(l): 
    r"""
    This is the Unit Angular Power Spectrum function.

    Parameters
    ----------
    l : float or np.ndarray 
        The angular multipole :math:`{\ell}` value(s).
    Returns
    -------
    float
        1.0

    """
    return 1.0 
    
    
def grf(nside, lmax, apsfunc, seed):
    r"""
    Generates  GRF values  for every pixel using the input APS(apsfunc).The GRF is monopole subtracted.

    Parameters
    ----------
    nside : int  
        Healpix parameter, usually power of 2.
    lmax : int 
        Maximum value of angular multipole.(:math:`\ell`)
    apsfunc : function 
        The input APS function.
    seed : int or float 
        A seed value, which can be provied to get different realizations for same APS.
        If you pass the seed value, then the output GRF is a fixed one.
        
    Returns
    -------
    np.ndarray 
        Contains the GRF value for each pixels. Shape (:math:`12\ {nside}^2,`). Returned values are in  :math:`\text{mK}` unit.


    .. raw:: latex

        \clearpage
   
    """
    
    cl = np.zeros(lmax+1)                                     # angular power spectrum array
    cl[1:] = apsfunc(np.arange(1,lmax+1, dtype = np.float64)) # APS cl array, No monopole.
    np.random.seed(seed)                                      # Seeding Part
    map_grf = hp.synfast(cl, nside = nside, lmax = lmax)      # Generating GRF for that cl array and for that seed.
    return map_grf
    
@np.vectorize # vectorize it so that it can take higher dimension arrays as a input
def ps(k):    # Function to estimate 3D power spectrum
    r"""
    This is the input model Power Spectrum function, which has the form :math:`A \left(\dfrac{k}{k_0}\right)^{s}`, with :math:`A=1, k_0=1, s=-2` (Default set).
    :math:`A` has the unit of :math:`\text{mK}^2\;\text{Mpc}^3`. :math:`k, k_0` are in :math:`\text{Mpc}^{-1}` unit.

    Parameters
    ----------
    k  : float or np.ndarray
        :math:`k = |\vec{\mathbf{k}}|`, the comoving wave number.
        
    Returns
    -------
    float or np.ndarray
        The value of Power Spectrum :math:`P(k)`  at comoving wave number :math:`k`.
        
    Notes
    -----
    **Returns 0.0 if** :math:`k=0`.
    
    """

    # set the parameters in power spectrum 
    A  =  1.0  # mk^2 unit
    s  = -2.0  # power law index
    k0 =  1.0  # Mpc^-1 unit

    if k == 0:
        return 0.0
    else:
        return  A*(k/k0)**(s)
        
        
def sky3dgrf(nside, seed, psfunc = ps):
    r"""
    This function simulates the 3D sky signal for a given ``nside``, ``Power Spectrum function``, ``seed`` value across all the frequency channels taken from  input ``UVFITS`` file.
    
    Parameters
    ----------
    nside : int
        Healpix parameter (usually power of 2) 
    seed : Positive int or None
        seed value for generating the :math:`a_{\ell m}`\'s. Different seed value will result in different :math:`a_{\ell m}` \'s, but for a particular seed value that will be fixed.
    psfunc : function 
        The input Power Spectrum function. By default it is :math:`\texttt{ps}` which has the form :math:`A \left(\dfrac{k}{k_0}\right)^{s}`, with :math:`A=1, k_0=1, s=-2`.
        You can define your own power spectrum function make sure that it has only  :math:`k` dependence.
    
    Returns
    -------
    Array type
        2D array has the shape :math:`N_c \times \text{npix}`. Each row data corresponds to GRF at a particular frequency. npix is the number of pixels. To get the value of npix see notes section below.(usually it is :math:`12\text{nside}^2`).

    
    .. raw:: latex

        \clearpage
      
    Notes
    -------
    By default :math:`\ell_{max} = 3*\text{nside} -1`.
    To extract gaussian random field value :math:`T(\hat{\mathbf{n}}, \nu)` for a given channel no or no\'s (indexing starts from 0).

    .. code-block:: python
    
        >> import healpy as hp
        >> nside,seed = 16, None
        >> npix = hp.nside2npix(nside) 
        >> print(npix)
        >> channel = 10   
        >> # for channels from 1 to 10 
        >> channel = [ i for i in range(10)]
        >> grf = sky3dgrf(nside,seed) # generate the grf
        >> grf[channel]             # extract the values
        

    GRF across all channels has no monopole, as :math:`a_{0,0}` **is set zero.**
    
    ===================

    If you want to give your own power spectrum function (make sure it is vectorized)


    .. code-block:: python
    
        >> import numpy as np
        >> import builtins as blt
        >> import myutils.simvis.simvis as sv
        >> # make sure all are float.
        >> blt.A  =  1.0
        >> blt.s  = -3.0
        >> blt.k0 =  1.0
        >> @np.vectorize
        >> def mypkfunc(k):
        >>     if k == 0:
        >>         return 0.0
        >>     else:
        >>         return  A*(k/k0)**(s)    
        >>  
        >> grf = sv.sky3dgrf(nside, seed, psfunc = mypkfunc)  # generate the grf
        >>
        
    .. raw:: latex

        \clearpage
    """
    
    lmax = 3*nside -1              # Maximum Angular Mulipole Value
    print(f"lmax                : {lmax}")

    start = time.time()

    # set the seed
    np.random.seed(seed)
    
    if seed!=None:
        print(f"Entered seed        : {seed}")

    kperp = np.arange(lmax+1)/r                  # kperpendicular array  

    ncc = 2*nc # simulating the field, for twice the bandwidth

    lim = ncc//2 if ncc%2 else ncc//2-1

    kpar  = np.array([ii for ii in range(ncc//2+1)]+[jj for jj in range(lim,0,-1)],dtype = np.float64 )*2.0*np.pi/(ncc*dnu*rp)  # kparallel array

    # compute k = sqrt(kperp**2+kpar**2) for all possible values from these kperp and kpar arrays

    kperp , kpar = np.meshgrid(kperp,kpar,indexing = 'ij')      # make grid for kperp and kpar
    
    k = np.sqrt(kperp**2+kpar**2)                               # all k values
    
    pk = psfunc(k)                                              # power spectrum value at all k values.
    
    factor = 2.*ncc*dnu*r**2*rp
    
    dim = int((lmax+1)*(lmax+2)/2)                              # total no of spherical harmonic coefficients.
    
    AA = np.concatenate([ pk[m:] for m in range(lmax+1) ],axis = 0)  # store in healpix ordering
    
    AA = np.sqrt(AA/factor)                                     # now divide by that factor and take the root.
    
                                                 
    # generate random numbers xq,yq
    # Multiply random numbers and the prefactor 
    
    AA = AA*(np.random.randn(dim,ncc) + 1j*np.random.randn(dim,ncc))  # sqrt(2) included in factor
    
    # perform fft along the frequency axis, axis =1

    
    alms = sp.fft.fft(AA,axis = 1)    # perform fft 
    
    # set monopole to zero and extract the real part for m = 0
    
    alms[0] = 0.0
    alms[1:lmax+1,:] = alms[1:lmax+1,:].real

    alms =  np.ascontiguousarray(alms.T)            
    
    # Take transpose to make the shape ncc times (lmax+1)*(lmax+2)/2 
    # In order to make the data contiguous across channels as healpy won't support otherwise.

    npix = hp.nside2npix(nside)    # no of pixels
    
    mapgrf = np.zeros((ncc,npix))  # make an empty array for grf for all channels
    
    for i in range(ncc):
        mapgrf[i] = hp.alm2map(alms[i], nside = nside, lmax = lmax) # loop over all channels
    
    end = time.time()
    hours, rem = divmod(end-start, 3600)
    minutes, seconds = divmod(rem, 60)
    print(f"Time taken for GRF  :"+"{:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))

    # show grf for first frequency channel
    hp.mollview(mapgrf[0])
    hp.graticule()
  
    # save the map if needed again
    # no need to simulate it when multiple pointing is involved

    if not os.path.exists('./grf'):
        os.makedirs('./grf')

    np.save(f'./grf/grf_{nside}_{seed}.npy',mapgrf[:nc])
    
    return mapgrf[:nc] # give values for only bandwidth region


def allskysim(nside, seed, skysimtype, apsfunc = aps, psfunc = ps):
    r"""
    This function creates all sky maps accordingly.
    
    Parameters
    ----------
    nside : int 
        Healpix parameter.Usually power of 2.
    seed : Positive int or None
        If it's 2D map then in each realization the seed gets increased by 1.
    skysimtype : str
        Either '2D' or '3D'. If given '2D' then creates multiple realizations. If given '3D' then makes a 3D sky map.
    apsfunc : function 
        The input Angular Power Spectrum function. By default it is :math:`\texttt{aps}` which has the form :math:`C_{\ell}=A\, \left(\frac{\ell}{\ell_0}\right)^{\beta}` with :math:`A = 100\;\text{mK}^2`, :math:`\ell_0=1`, :math:`\beta` = -2.
    psfunc : function 
        The input Power Spectrum function. By default it is :math:`\texttt{ps}` which has the form :math:`A \left(\dfrac{k}{k_0}\right)^{s}`, with :math:`A=1, k_0=1, s=-2`.
       
    Returns
    -------
    np.ndarray
        Surface brightness temperature field map (multiple realizations or across all frequencies) in units of :math:`\text{mK}`.

    """
    
    if skysimtype == '2D':
        
        lmax = nside * 3 - 1         # maximum number of \ell multipoles used
        Npix = hp.nside2npix(nside)  # No of pixels for the given nside
    
        start = time.time()
    
        # Generate the GRF
        map_2d_grf = np.zeros((Nrea, Npix))
        
        for ii in range(Nrea):
            map_2d_grf[ii] = grf(nside, lmax, apsfunc, seed)
            if seed!= None:
                seed += 1 # Incrementing seed by 1 
                          # Means in 1st realization we get seed value = seed, next realization seed = seed + 1 and so on.
                
        print(f"2D maps             : {Nrea} realizations.")
        print(f"l_max               : {lmax}")
        print(f"Sky maps generated  : {Nrea}")
    
        end = time.time()
        hours, rem = divmod(end-start, 3600)
        minutes, seconds = divmod(rem, 60)
        print(f"Time taken for GRF  :"+" {:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))
        
        # show the first realization grf
        hp.mollview(map_2d_grf[0])
        hp.graticule()
        
        return map_2d_grf
        
    elif skysimtype == '3D':
        print(f"Creating 3D maps.")

        ## Define the parameters

        # Some important value notations
        # nu_c   # centered frequency (MHz unit)
        # kperp  # perpendicular comp of $\vec{k}$
        # kpar   # parallel comp of $\vec{k}$
        # nc     # no of frequency channels
        # dnu    # channel width (MHz unit)
        # r      # comoving distance at nuc (Mpc unit)
        # rp = dr/dnu at nuc (Mpc/MHz unit)
        
        nu_e   = 1420.0              # nu_e = emitted frequency 1420 MHz
        
        z      = nu_e/nu_c -1        # z = redshift
        
        blt.r  = cosmo.comoving_distance(z).value # comoving distance in Mpc unit for the redshift z
        
        H_z    = cosmo.H(z).value    # value of Hubble parameter in km/s/Mpc unit at redshift z
        
        blt.c  = 2.99792e5           # speed of light in km/s
        
        blt.rp = (1+z)**2*c/(nu_e*H_z)

        print(f"Centered Frequecy   : {nu_c} MHz")
        print(f"Redshift            : z  = {z:.2f}")
        print(f"Comoving distance   : r  = {r:.2f} Mpc.")
        print(f"rprime              : rp = {rp:.2f} Mpc/MHz.")
        


        return sky3dgrf(nside, seed, psfunc)
        
#---------- Simulating Primary Beam ---------------------

blt.b = 4.0	    # b here is the antenna dimension

blt.C = 299.792 # speed of light in 1000 km /s unit as per to get C/nu = lambda in m unit (as nu is in MHz unit)

def beammwa(nu, ne1, ne2):
    r"""
    We model primary beam function of **MWA telescope** as a product of two sinc square function given as : 

    .. math::
    
        \mathcal{A}(\hat{\mathbf{n}},\nu) = \text{sinc}^2\left[\frac{{\pi}b{\nu}\ \Delta\hat{\mathbf{n}}\cdot\hat{\mathbf{e}}_1 (\alpha_{p})}{c}\right]\;\text{sinc}^2\left[\frac{{\pi}b{\nu}\ \Delta\hat{\mathbf{n}}\cdot\hat{\mathbf{e}}_2 (\alpha_{p})}{c}\right]
        
    Where :
    
    .. math::
    
        \Delta\hat{\mathbf{n}}= \hat{\mathbf{n}}- \hat{\mathbf{p}}

    :math:`\hat{\mathbf{p}}` is the pointing direction of the telescope. We assume that telescope is zenith pointed (:math:`\hat{\mathbf{p}}` = :math:`\hat{\mathbf{e}}_3`).
    :math:`\hat{\mathbf{n}}` refers to different directions on the sky (i.e. celestial sphere).
    
    Parameters
    ----------
    nu : float 
        Observing frequency :math:`\nu` in MHz unit.
    ne1 : float or array 
        The dot product given by :math:`\hat{\mathbf{n}}\cdot\hat{\mathbf{e}}_ 1`.
    ne2 : float or array 
        The dot product given by :math:`\hat{\mathbf{n}}\cdot\hat{\mathbf{e}}_ 2`.
    Returns
    -------
    float or np.ndarray 
        The primary beam pattern value for square aperture telescope (MWA).


    .. raw:: latex

        \clearpage
        
    """
    
    return (np.sinc((b*nu/C)*ne1)**2.0)*(np.sinc((b*nu/C)*ne2)**2.0) # Product of Two sinc^2 Functions.

def pbgen(nside, ra_ptg, dec_ptg, nu, pbfunction):
    r"""
    This generates the primary beam for MWA telescope (zenith pointed), given the RA,DEC and frequency.

    Parameters
    ----------
    nside : int 
        Healpix parameter.Usually power of 2.
    ra_ptg : int or float 
        RA of the pointing direction.(In degrees)(zenith pointed).
    dec_ptg : int or float 
        DEC of the pointing direction.(In degrees)(zenith pointed).
    nu : float 
        Observing frequency :math:`\nu` in :math:`\mathrm{MHz}` unit.
    pbfunction : function 
        The primary beam (PB) pattern of MWA which is product of two :math:`\text{sinc}^2` functions.
    Returns
    -------
    PB : np.ndarray 
        The value of primary beam of the MWA telescope (beam pattern value at each pixel which quantifies the reponse of telescope along different directions). Shape :math:`(12\ {nside}^2, )`.
                
    """
    start = time.time()
    
    MWA_vec = hp.ang2vec(ra_ptg, dec_ptg, lonlat = True) # Unit Vector Towards MWA Pointing Direction
    
    ipix = hp.query_disc(nside = nside, vec = MWA_vec, radius = np.radians(90)) 
    # ipix is the array which contains all the pixel indices which are in upper hemisphere w.r.t telescope.
    
    hatn = hat_n(nside,ipix)   # The components of hat(n) for all the pixels
    
    # convert RA,Dec into radians
    a0 = np.deg2rad(ra_ptg)    # RA of the phase center in radians
    d0 = np.deg2rad(dec_ptg)   # Phase center dec of the observation in radians
    
    # create the basis vectors
    e1 = np.array([-np.sin(a0), np.cos(a0), 0])
    e2 = np.array([-np.cos(a0)*np.sin(d0), -np.sin(d0)*np.sin(a0), np.cos(d0) ])
    e3 = np.array([np.cos(a0)*np.cos(d0), np.sin(a0)*np.cos(d0), np.sin(d0) ])
    
    # compute \Delta\hat{n}= \hat{n} - (\hat{p} = \hat{e}_3)
    hatn -= e3
    
    # Calculate the dot product \hat{e}_1.\Delta\hat{n} for all pixels
    ne1 = np.dot(hatn, e1) 
    
    # Calculate the dot product \hat{e}_2.\Delta\hat{n} for all pixels
    ne2 = np.dot(hatn, e2) 
    
    npix = hp.nside2npix(nside) # no of pixels
    
    PB = np.zeros(npix)         # Primary beam array which will contain the values of primary beam
    
    PB[ipix] = pbfunction(nu, ne1, ne2)
    
    end = time.time()
    hours, rem = divmod(end-start, 3600)
    minutes, seconds = divmod(rem, 60)
    print(f"Beam cal time       : "+"{:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds)) 
    # plot the primary beam
    hp.mollview(PB)
    hp.graticule()
    
    return PB 

#---------------Phase Calculation -------------------------

def hat_n(nside, ipix):
    r"""
    This calculates the :math:`xyz` components of :math:`\hat{\bf n}` for all the given pixels contained in the array ipix. 

    Parameters
    ----------
    nside : int 
        Healpix parameter.Usually power of 2.
    ipix : np.ndarray
        Pixel array for which components of :math:`\hat{\bf n}` have to be calculated. 
        
    Returns
    -------
    hatn : np.ndarray
        Containg the :math:`xyz` comps of :math:`\hat{\bf n}`. :math:`N_{pix} \times 3`.

    """
    theta, phi = hp.pix2ang(nside,ipix) # theta and phi for pixels contained in ipix array.
    
    hatn = np.array([np.sin(theta) * np.cos(phi),np.sin(theta) * np.sin(phi),np.cos(theta)]).T
    
    return hatn # xyz Components of hatn for pixels contained in ipix array.


def dot_cal_superfast(nside, ra_ptg, dec_ptg, ipix):
    r"""
    This calculates components of :math:`\Delta\hat{\bf n}` for all the given pixels contained in the array ipix along the basis vectors :math:`\hat{\mathbf{e}}_1,\hat{\mathbf{e}}_2,\hat{\mathbf{e}}_3` given by the RA,DEC.

    Parameters
    ----------
    nside : int 
        Healpix parameter. Usually power of 2.
    ra_ptg : int or float 
        RA of the pointing direction.(In degrees)
    dec_ptg :int or float 
        DEC of the pointing direction.(In degrees)
    ipix : np.ndarray
        Pixel array for which dot products have to be calculated. 
    Returns
    -------
    dot_products : np.ndarray 
        Contains the :math:`xyz` comps for those pixels. Shape :math:`N_{pix} \times 3`.
                
    """
    
    hatn = hat_n(nside,ipix)  # xyz Components of hat_n for pixels contained in ipix array.
    
    # convert the RA, Dec into radians
    a0 = np.deg2rad(ra_ptg)   # RA of the phase center in radians
    d0 = np.deg2rad(dec_ptg)  # Phase center dec of the observation  in radians
    
    # create the basis functions e1,e2,e3
    e1 = np.array([-np.sin(a0), np.cos(a0), 0])
    e2 = np.array([-np.cos(a0)*np.sin(d0), -np.sin(d0)*np.sin(a0), np.cos(d0)])
    e3 = np.array([np.cos(a0)*np.cos(d0), np.sin(a0)*np.cos(d0), np.sin(d0)])
    
    e  = np.stack((e1,e2,e3), axis = 1) 
    
    dot_products = np.dot(hatn - e3, e) # \hat{n} - \hat{p} term
    
    return dot_products


    
def calculate_phase(dot_product, bl):
    r"""
    Given the baselines it calculates the phase factor for all the pixels given in dot_product array.

    Parameters
    ----------
    dot_product : np.ndarray  
        Contains the  components for all the given those pixels along the basis vectors :math:`\hat{\mathbf{e}}_1,\hat{\mathbf{e}}_2,\hat{\mathbf{e}}_3`. Shape is :math:`N_{pix} \times 3`.
    bl : np.ndarray 
        The array conatins the components of the baselines. (defined by basis vectors :math:`\hat{\mathbf{e}}_1,\hat{\mathbf{e}}_2,\hat{\mathbf{e}}_3`).
        Shape :math:`N_{Baselines}\times 3`.
    Returns
    -------
    np.ndarray 
        The phase factor. Shape :math:`N_{pix} \times  N_{Baselines}`.
    """
    
    dot = dot_product@bl.T # Calculating Dot Product U.\Delta n for all the baselines contained in bl array.
    
    ppi = np.pi            # value of pi
    return ne.evaluate('exp(2*ppi*1j*dot)') 
    # Calculating phase exp(2 pi i U.\Delta n) for all the baselines contained in bl array. #np.exp(2j*np.pi*dot).astype(np.complex64)
    
    
#------------ Visibility Calculation ---------------
blt.KB = 1.38		 	# Boltzmann constant in Jy.m^2/mK

def visgen_mwa_multi(nside, grfmap, ra_ptg, dec_ptg, bl, nu):
    r"""
    This calculates the visibilities for the baselines passed by bl array, given the RA, DEC, frequency, grfmap. 

    Parameters
    ----------
    nside : int 
        Healpix parameter. Usually power of 2.
    grfmap : np.ndarray 
        Contains the sky brightness temperature simulated from APS or PS for all the pixels.(Contains multiple realizations for 2D map or sky signal across all frequencies for 3D map). Shape :math:`N_{Realizations}\times 12\ {nside}^2` for 2D, :math:`N_{c}\times 12\ {nside}^2` for 3D.
    ra_ptg : int or float 
        RA of the pointing direction.(In degrees)
    dec_ptg : int or float 
        DEC of the pointing direction.(In degrees)
    bl : np.ndarray 
        The array conatins the components of the baselines defined by basis vectors :math:`\hat{\mathbf{e}}_1,\hat{\mathbf{e}}_2,\hat{\mathbf{e}}_3`.
        Shape :math:`N_{Baselines}\times 3`.
    nu : float 
        Observing frequency :math:`\nu` in :math:`\mathrm{MHz}` unit.
    
    Returns
    -------
    vis : np.ndarray
        Contains the simulated visibility for all the baselines given by bl. (Along axis = 0 is the different realizations or frequencies, axis = 1 is the baselines). Shape :math:`N_{realizations}\ \times\ N_{Baselines}` for 2D, :math:`N_{c}\times \ N_{Baselines}` for 3D. 


    .. raw:: latex

        \clearpage
    """
    start = time.time()
    
    Npix = hp.nside2npix(nside)                     # Number of pixels given nside
    res  = hp.nside2resol(nside, arcmin = True)     # resolution in arc minutes given nside
    
    print(f"Resolution of map   : {res:.3f} arc-minutes")
    print(f"No of pixels        : {Npix}")
    
    dOmega = hp.nside2resol(nside)**2.0             # resolution in steredian unit

    
    # make the pixel arrays (ipix) which are sensitive in only upper hemisphere seen by Telescope by ra_ptg,dec_ptg.
    Tel_vec = hp.ang2vec(ra_ptg, dec_ptg, lonlat = True)
    ipix = hp.query_disc(nside = nside, vec = Tel_vec, radius = np.radians(90))
    
    # Generate The PB 
    Pb_map = pbgen(nside, ra_ptg, dec_ptg, nu, beammwa) # Primary beam MWA
    
    # dot product xyz comps of all the pixels conatined in ipix array
    dot_product = dot_cal_superfast(nside, ra_ptg, dec_ptg, ipix)        
    
    lam = C/nu  # wavelength in m
    Q_nu = 2.0 * KB / lam**2.0  # in Jy/mK

    print(f"Working Wavelength  : {lam:.3f} m")
    print(f"Conversion factor   : {Q_nu:.3f}")
    
    # Multiply GRF and PB
    GRF_PB_Product = grfmap[:,ipix]*Pb_map[ipix]
    
    nbl = bl.shape[0]   # number of baselines

    # creates the chunks the visibility computation for all the baselines are divided into sub processess.
    # The visibilities of chunk_size no of baselines are being computed in a single step.
    
    chunks = np.array_split(np.arange(nbl), np.arange(chunk_size, nbl, chunk_size))

    vis = np.zeros((GRF_PB_Product.shape[0], nbl), dtype = np.complex128)
    
    for ii in chunks:
        phase = calculate_phase(dot_product,bl[ii,:]) # Phase calculation
        vis[:, ii] = GRF_PB_Product@phase                  # Visibility Calculation
        
    # Multiply by Q_nu And dΩ
    vis =  Q_nu * dOmega * vis 
    
    end = time.time()
    hours, rem = divmod(end-start, 3600)
    minutes, seconds = divmod(rem, 60)
    print(f"Visibility time     : "+"{:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds)) 
    
    return vis

#----------------------Reading from uv-Fits file and Simulate the Visibilities and create new fits file----------------

def sim_vis_hdul(hdulist, skysimtype, seed = None, apsfunc = aps, psfunc = ps):
    r"""
    This calculates the visibilities for the baselines passed by bl, given the RA, DEC, frequency. It extracts the necessary information from the ``hdulist``, like the baseline distribution, centered frequency, frequency resolution, no of frequency channels, RA, DEC.

    Parameters
    ----------
    hdulist : object
        A fits object
    skysimtype : str
        Either '2D' or '3D'. If given '2D' then creates multiple realizations. If given '3D' then makes a 3D sky map.
    seed : int or float
        A seed value for generating GRF. (By deafult set to ``None``)
    apsfunc : function 
        The input Angular Power Spectrum function. By default it is :math:`\texttt{aps}` which has the form :math:`C_{\ell}=A\, \left(\frac{\ell}{\ell_0}\right)^{\beta}` with :math:`A = 100\;\text{mK}^2`, :math:`\ell_0=1`, :math:`\beta` = -2.
    psfunc : function 
        The input Power Spectrum function. By default it is :math:`\texttt{ps}` which has the form :math:`A \left(\dfrac{k}{k_0}\right)^{s}`, with :math:`A=1, k_0=1, s=-2`.

    Returns
    -------
        Simulates the visibilities for the input Power Spectrum or Angular Power Spectrum for the RA, DEC, frequency and baseline distribution given in the ``hdulist`` object. (And puts the multiple realizations along the frequency axis if simulated sky signal is 2D). 

    .. raw:: latex

        \clearpage
    """

    start = time.time()
    
    # EXRACTING BASELINES AND OTHER DETAILS
    
    #print (hdulist.info()) 				# to check different headers
    
    data_tmp = hdulist[0].data 				# to save visibilities in data
    dataT = Table(data_tmp)                 # to convert data from array form to table form, readable
    #print(hdulist[0].header)

    
    blt.nu_c = hdulist[0].header['CRVAL4']             # Centered Frequency in Hz unit 
    pol = hdulist[0].header['NAXIS3']                  # No of Polarization Channels RR,LL,RL,LR
    blt.nc = hdulist[0].header['NAXIS4']               # No of Frequnecy Channels
    blt.dnu = hdulist[0].header['CDELT4']*1.e-6        # Frequency Resolution in MHz unit 1e-6 to convert from Hz to MHz
    
    ra_ptg = hdulist[0].header['CRVAL6']  # Read the RA in degrees
    dec_ptg = hdulist[0].header['CRVAL7'] # Read the DEC in degrees

    
    # Extract the baselines
    u = dataT['UU']*blt.nu_c  # *nu_c for baseline unit   # *2.99792e8 (for meter unit)
    v = dataT['VV']*blt.nu_c
    w = dataT['WW']*blt.nu_c
    bln = np.stack((u,v,w),axis = 1) # baseline array
    
    blt.nu_c = blt.nu_c*1.e-6  # Centered Frequency in MHz unit 1e-6 to convert from Hz to MHz
    
    print(f"No of baselines     : {bln.shape[0]}")
    print(f"Centered Frequecy   : {nu_c} MHz")
    print(f"Number of channels  : {nc}")
    print(f"Frequency resolution: {dnu:.3f} MHz")
    print(f"Pointing RA         : {ra_ptg:.3f}")
    print(f"Pointing DEC        : {dec_ptg:.3f}")

    # generating all sky map (2D or 3D)
    if skysimtype == '3D':
        print(f"Type of Map         : 3D sky maps.")
        try:
            map_grf = np.load(f'./grf/grf_{nside}_{seed}.npy')
            print(f"Loading saved GRF   : ./grf/grf_{nside}_{seed}.npy")
            
        except:
            map_grf = allskysim(nside, seed, skysimtype, apsfunc, psfunc)

    else:
        print(f"Type of Map         : 2D sky maps.")
        map_grf = allskysim(nside, seed, skysimtype, apsfunc, psfunc)  
    print(f"map_grf shape       : {map_grf.shape}")

    # simulate visibilities
    vis = visgen_mwa_multi(nside, map_grf, ra_ptg, dec_ptg, bln, nu_c)
    print(f"Visibility Shape    : {vis.shape}")
    
    # data output for multiple realization in same file along the frequency channels
    for j in range(pol):
        hdulist[0].data['DATA'][:,0,0,0,:vis.shape[0],j, 0] = vis.T.real.astype(np.float32) # Recast Everything to 32 bit number 
        hdulist[0].data['DATA'][:,0,0,0,:vis.shape[0],j, 1] = vis.T.imag.astype(np.float32) # Recast Everything to 32 bit number

    if skysimtype == '2D':
        #  set dnu_c = 0 in output file
        print(f"Frequeny separation : Set to Zero (2D map)")
        hdulist[0].header['CDELT4'] = 0.0
        
    end = time.time()
    hours, rem = divmod(end-start, 3600)
    minutes, seconds = divmod(rem, 60)
    
    print(f"Took time all total : "+"{:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))
    print(f"<=== Visibility Simulated ===>")


def sim_vis(uvinfits, uvoutfits, skysimtype, seed = None,  apsfunc = aps, psfunc = ps):
    r"""
    
    This calculates the visibilities for the baselines passed by bl_file, given the RA, DEC, frequency. It extracts the necessary information from the ``uvinfits`` file, like the baseline distribution, centered frequency, frequency resolution, no of frequency channels, RA, DEC and writes the simulated visibilities in a new fits file named ``uvoutfits`` file.

    Parameters
    ----------
    uvinfits : fits file 
        Input fits File.
    uvoutfits : fits file 
        Output fits File.
    skysimtype : str
        Either '2D' or '3D'. If given '2D' then creates multiple realizations. If given '3D' then makes a 3D sky map.
    seed : int or float
        A seed value for generating GRF.
    apsfunc : function 
        The input Angular Power Spectrum function. By default it is :math:`\texttt{aps}` which has the form :math:`C_{\ell}=A\, \left(\frac{\ell}{\ell_0}\right)^{\beta}` with :math:`A = 100\;\text{mK}^2`, :math:`\ell_0=1`, :math:`\beta` = -2.
    psfunc : function 
        The input Power Spectrum function. By default it is :math:`\texttt{ps}` which has the form :math:`A \left(\dfrac{k}{k_0}\right)^{s}`, with :math:`A=1, k_0=1, s=-2`.

    Returns
    -------
        Simulates the visibilities for the input Power Spectrum or Angular Power Spectrum for the RA, DEC, frequency and baseline distribution given in the ``uvinfits`` file. (And puts the multiple realizations along the frequency axis if simulated sky signal is 2D). Writes the data in ``uvoutfits`` file.

    """
    print(f"nside               : {nside}")
    print(f"chunk size          : {chunk_size}")
    start = time.time()
    
    with fits.open(uvinfits) as hdulist:
        sim_vis_hdul(hdulist, skysimtype, seed, apsfunc, psfunc)
        hdulist.writeto(uvoutfits, overwrite = True) # note that the files will be overwritten
    
    end = time.time()
    
    hours, rem = divmod(end-start, 3600)
    minutes, seconds = divmod(rem, 60)
    
    print(f"Took time all total  :"+"{:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))
    print(f"<=== Write on fits file done successfully ===>")

print(f"Imported simvis,    Import Time : {datetime.now().strftime('%d/%m/%y | %I:%M:%S %p')}")
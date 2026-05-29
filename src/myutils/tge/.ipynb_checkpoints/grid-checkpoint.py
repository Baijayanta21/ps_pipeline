r"""
Tapered Gridded Estimator (TGE)
===============================
For a given **input uv-fits file**, we taper the sky response of the telescope by convolving visibilities with a function :math:`\tilde{w}(\vec{\textbf{U}})=\pi \theta_w^2 \textbf{exp}[-\pi^2 \theta_w^2 \textbf{U}^2]` (which is the Fourier Transform of a window function :math:`\mathcal{W}(\vec{\boldsymbol{\theta}}) = \textbf{exp}[-\theta^2/\theta_w^2]` which falls off to a value close to zero well before the first null of the telescope’s primary beam pattern ) and gridded on a square grid in the :math:`uv`-plane using,

.. math ::

    \begin{align}
    \boxed{\mathcal{V}_{cg}^{P}(\vec{\textbf{U}}_g,\nu_a)=\sum_{i}\  \tilde{w}(\vec{\textbf{U}}_g - \vec{\textbf{U}}_i)\ \mathcal{V}^{P}(\vec{\textbf{U}}_i,\nu_a)\ F_i^{P}(\nu_a)}
    \end{align}
    
The sum is over all the baselines. Here :math:`\nu_a` refers to the different frequency channels, :math:`\vec{\textbf{U}}_i` is :math:`i^{\rm th}` baseline. :math:`\mathcal{V}_{cg}^{P}` refers to the convolved visibility at grid point :math:`g` and by the superscript :math:`P` we denote the polarization. :math:`\vec{\textbf{U}}_g` is the baseline corresponding to this grid point and :math:`F_i^{P}(\nu_a)` incorporates the flagging information of the data corresponding to the frequency :math:`\nu_a` and for the corresponding polarization :math:`P`. :math:`P` can take be either in  :math:`\rm XX,YY,XY,YX` basis or in the circular basis :math:`\rm RR,LL,RL,LR`, based on the telescope.

:math:`F_i^{P}(\nu_a)` has a value ‘0’ if the data at a given baseline and frequency is flagged and ‘1’ otherwise. Note that the baselines :math:`\vec{\textbf{U}}_i` here are defined at a fixed reference frequency :math:`\nu_c` (**centered frequency of the fits file**), and in this analysis they are considered to be fixed as we vary the frequency channel :math:`\nu_a` (although in principle they are not).

In the visibility data, we flag the data if the corresponding weight value is less than 0, otherwise we take it as it is.

We calculate convolved visibility :math:`\mathcal{V}_{cg}^{P}(\vec{\textbf{U}}_g,\nu_a)` for a given no of frequency channels and for given polarizations of the data. A 4D array is returned, where **first one is the polarization axis**, next **two dimensions are values of convolved visibility at each grid points** (square grid) in :math:`uv` plane and **last one is frequency axis**.

.. raw:: latex

        \clearpage


Here :math:`\theta_w = f\theta_0`, with :math:`\theta_0 = 0.6 \ \theta_{\rm FWHM}`. :math:`f` is known as tapering parameter whose value can be suitably chosen. The effect of tapering is enhanced if the value of :math:`f` is reduced. For instance :math:`f` < 1 would suppress the sky response from the outer regions and the side-lobes of the PB pattern, whereas a large value :math:`f` > 1 would imply very little tapering.
Preferably :math:`f\leq 1` so that :math:`\mathcal{W}(\vec{\boldsymbol{\theta}})` cuts off the sky response well before the first null of the primary beam.

The grid spacing in the :math:`uv` plane is given by  :math:`\Delta{\text{U}}=\dfrac{\sqrt{ln2}}{2\pi\theta_{\rm eff}}` with :math:`\theta_{\rm eff}=\dfrac{f\theta_0}{\sqrt{1+f^2}}`.

Meanwhile, those baselines are rejected for which :math:`|\vec{\textbf{U}}_i|<|\vec{\textbf{U}}|_{max}`. We effectively increase the size of visibility data twice ; original one and one for all the baselines with :math:`-\vec{\textbf{U}}_i` as :math:`\mathcal{V}^{P}(-\vec{\textbf{U}}_i,\nu_a) = \left[\mathcal{V}^{P}(\vec{\textbf{U}}_i,\nu_a)\right]^{*}`.

For any baseline :math:`\vec{\textbf{U}}_i`, the contribution of visibilities to the grid points is taken for only those grid points for which are within(and on) a square grid of side length :math:`12\Delta{\text{U}}` whose center is at grid point which is nearest to :math:`\vec{\textbf{U}}_i` (The sides of this grid are along :math:`u` and :math:`v` axis). The weight function  :math:`\tilde{w}(\vec{\textbf{U}}_g - \vec{\textbf{U}}_i)` falls considerably faster and we do not expect a significant contribution from the visibilities beyond this baseline separation.
    
:math:`\textcolor{blue}{\rm \textbf{Please note that in this analysis Baseline migration is not incorporated.}}`


.. raw:: latex

        \clearpage
        
Module Functions
================

"""
import numpy as np
from astropy.io import fits
import time
from datetime import datetime

def mkbin(GV, dU, Umax, FWHM, f, binUmax, binUmin, Nbin, Mg_min, outf):
    r"""
    Given a **UAPS convolved gridded visibility array** and suitable gridding parameters it identfies the grid points which are relevant and which grid points are in which bin.
    Also the effective multipole value for a particular bin is given by : 

    .. math::

        \ell_a = \frac{\sum_g\,w_g\,\ell_g}{\sum_g\,w_g}; \quad \text{with}\, w_g = M_g


    And :math:`\ell_g = 2\pi |\vec{\textbf{U}}|_{g}`

    Parameters
    ---------
    GV : np.ndarray
        Convolved gridded visbility array for the **UAPS** visibilities. Different realizations have been put along frequency axis.
    dU : float
        Grid spacing :math:`\Delta \text{U}` in wavelength units.
    Umax : int or float
        The maximum value of baseline :math:`|\vec{\textbf{U}}|_{max}`. Baselines greater than this are rejected.
    FWHM : int or float
        The full width half maximum (FWHM) of telescope's primary beam pattern.(in degrees)
    f : int or float
        Tapering parameter value.
    binUmax : int or float
        Maximum value of :math:`|\vec{\textbf{U}}|` (in units of wavelength :math:`\lambda`) to be used in the binning infomation. Grid points having baseline greater than this are rejected.
    binUmin : int or float
        Minimum value of :math:`|\vec{\textbf{U}}|` (in units of wavelength :math:`\lambda`) to be used in the binning infomation. Grid points having baseline lesser than this are rejected.
    Nbin :  int
        Number of annular bins the whole :math:`uv` plane has been divided into.
    Mg_min : int or float
        Minimum value of normalization factor :math:`M_g` to be used. Grid points at which visibility self correlation is less than this are rejected.
    outf : string
        The name of file in which the binning information is saved. This is saved in **.npz** format.

    Notes
    ---------
    - :math:`d\text{U}` is the grid spacing in baseline units (:math:`\lambda`).
    - :math:`N\!I` is a 2D matrix, which contains the information which grid points are relevent. It contains value :math:`-2,-1,0,\cdots,N_{bin} - 1`. Grid points those satisfy :math:`NI \geq 0` are relevant. The rest of them are rejected in our analysis.
    - :math:`ix` is a 1D array, which contains the :math:`u` index of grid points in terms of grid spacing :math:`d\text{U}`, i.e. :math:`ix_g = u_g/d\text{U}`. (only for relevant grid points).
    - :math:`iy` is a 1D array, which contains the :math:`v` index of grid points in terms of grid spacing :math:`d\text{U}`, i.e. :math:`iy_g = v_g/d\text{U}`. (only for relevant grid points).
    - :math:`ni` is a 1D array, which contains information about which grid point falls into which annular bin. If the value is zero then it is residing in first bin, if 2 then in second bin and so on. Max value is :math:`N_{bin} - 1`. (only for relevant grid points).

    Examples
    --------
    >>> # BIN INFO
    >>> import grid as gd
    Imported Grid, Import Time : 19/05/26 | 07:48:35 PM
    >>> # ===================================================
    >>> # Set the gridding parameters in the TGE code
    >>> n1      = 0     # Starting channel number
    >>> n2      = 99    # End channel number
    >>> Flag    = False # Apply actual flagging of data
    >>> Umax    = 250   # Baselines greater than this are rejected
    >>> FWHM    = 23    # Primary beam FWHM of the telescope in degrees
    >>> f       = 0.6   # Tapering parameter value
    >>> nstokes = [0]   # 0 for XX and 1 for YY (polarizations to grid)
    >>> # ===================================================
    >>> nrel    = -1    # grid along freq axis   
    >>> # For binning
    >>> Nbin    = 20    # No of annular bins the whole uv plane has been divided into
    >>> binUmin = 6.0   # min |U| used in getting binning information
    >>> binUmax = 220   # max |U| used in getting binning information
    >>> Mg_min  = 0.01  # M_g cutoff, grid points have less than this will be rejected
    >>> 
    >>> inpath   = '/home/baijayanta/Complete_Pipeline/uaps_files/'
    >>> filename = '1000007200_out_uaps_15.fits'
    >>> # name of input .fits file for which we will calculate the bin info
    >>> infile   = inpath + filename     
    >>> # grid it # ([Nstokes,Nu,Nv,Nc],(dU,Umax,FWHM,f))
    >>> GV, info = gd.grid(infile,n1,n2,nrel,Umax,FWHM,f,Flag, nstokes)  
    <<<<<<<<<<< TGE >>>>>>>>>>>>
    << 19/05/26 | 07:48:35 PM >>
    Type  : Data
    << 19/05/26 | 07:48:48 PM >>
    Elapsed   :   00:00:12.82
    <<<<<<<<<<< Done >>>>>>>>>>>
    >>> # make bin info
    >>> gd.mkbin(GV[0], info[0], info[1], info[2], info[3], binUmax, 
    ...                              binUmin, Nbin, Mg_min, outf = 'bin_info_7200')
    <<< Bin info >>>
    dU   : 1.07          
    Umax : 250          
    FWHM : 23 degrees          
    f    : 0.6
    Shape: (457, 457, 100)
    data : 22.235
    Saved: bin_info_7200.npz

    .. raw:: latex

        \clearpage
    """
    print(f"<<< Bin info >>>")
    dd = (binUmax-binUmin)/Nbin  # bin width
    print(f"dU   : {dU:.2f}\
          \nUmax : {Umax}\
          \nFWHM : {FWHM} degrees\
          \nf    : {f}")
    dl = 2*np.pi*dU
    
    Nu = GV.shape[0]   # grid shape 
    Ng = Nu // 2
    NC = GV.shape[2]   # no of channels
    print(f"Shape: {GV.shape}")
    
    # Assign bin and ix,iy to each grid point 
    
    ix = np.empty((Nu,Nu), dtype = 'int')
    iy = np.empty((Nu,Nu), dtype = 'int')
    ni = np.empty((Nu,Nu), dtype = 'int')
        
    # self correlation of visibilities at each grid point, mean over realizations
    corr = np.mean((GV*np.conjugate(GV)).real, axis = 2) 
    
    wglg = np.zeros(Nbin) # weight times lg 
    wg   = np.zeros(Nbin) # weight

    for jj in range(Nu):
        for mm in range(Nu):
     
            
            ix[jj,mm] = jj-Ng # index w.r.t center 
            iy[jj,mm] = mm-Ng # index w.r.t center 
            
            
            nv = (ix[jj,mm]**2 + iy[jj,mm]**2)**(0.5)
            UU = dU*nv # calculate U for each grid point
    
            Un = (UU-binUmin)/dd
    
    
            if ((Un >= 0) and (Un < Nbin)): # those grid points which are within the specified region
    
                ni[jj,mm] = int(np.floor(Un))

                if corr[jj,mm] > Mg_min :  # only for those grid points which made the cutoff
                    wglg[ni[jj,mm]] += corr[jj,mm]*nv*dl
                    wg[ni[jj,mm]]   += corr[jj,mm]  
                
            else:
                ni[jj,mm] = -1
                
    NI   = np.copy(ni)   # grid points are outside binUmax and binUmin range has value -1. otherwise values indicate which bin they are into.
    lval = wglg/wg       # the value of l for each bin      
    
    NI[corr <= Mg_min] = -2  # the grid points where mean self correlation of visibilities are less than Mg_min
    
    # Identify grid points within the bins and flattens
    
    ix = ix[NI>=0]
    iy = iy[NI>=0]
    ni = ni[NI>=0]

    print(f"data : {(NI[NI>=0].shape[0]/(Nu**2))*100:.3f}")
    
    # save the relevant information
    np.savez(outf, Nbin = Nbin, dU = dU, Umax = Umax, FWHM = FWHM, f = f, ix = ix, iy = iy, ni = ni, NI = NI, lval = lval)

    print(f"Saved: {outf}.npz")


def grid(infile, n1, n2, nrel, Umax, FWHM, f, Flag, nstokes, seed = None):
    r"""
    This tapers the sky response and grids the :math:`uv` plane and computes convolved visibilities, given a fits file.

    .. math ::

    	\begin{align}
    	\boxed{\mathcal{V}_{cg}^{P}(\vec{\textbf{U}}_g,\nu_a)=\sum_{i}\  \tilde{w}(\vec{\textbf{U}}_g - \vec{\textbf{U}}_i)\ \mathcal{V}^{P}(\vec{\textbf{U}}_i,\nu_a)\ F_i^{P}(\nu_a)}
    	\end{align}


    Parameters
    ----------
    infile : Fits object
        Fits file  which contains all the data.
    n1 : int
        Starting frequency channel number.
    n2 : int
        End frequency channel number.
    nrel : int 
        #. If positive then it copies that frequency channel data to all the frequency channels. Flagging Information is same as original fits file.
        #. If -1 then leaves the data as is.
        #. If -2 then simulates random gaussian noise visibilities for the baselines with mean :math:`\mu = 0` and std :math:`\sigma = 1`.
    Umax : int or float
        The maximum value of baseline :math:`|\vec{\textbf{U}}|_{max}`. Baselines greater than this are rejected.
    FWHM : int or float
        The full width half maximum (FWHM) of telescope's primary beam pattern.(in degrees)
    f : int or float
        Tapering parameter value.
    Flag : boolean type (True or False)
        If True it flags the visibility data(puts the data to zero), else does nothing.
    nstokes : int or list 
        The int or the list elements must be within the range of polarizations in the original fits file.
    seed : int or None
        Default set None, seed value for noise only simulations.

    Returns
    --------
    Tuple 
        The tuple's first element is a 4D convolved gridded visibility array, where first dimension is polarization axis, next two dimensions are values of convolved visibility at each grid points (square grid) in :math:`uv` plane and last one is along frequency axis.
        Next element is in the form : (dU, Umax, FWHM, f).

    Examples
    --------
    >>> import grid as gd
    Imported Grid, Import Time : 19/05/26 | 07:28:16 PM
    >>> # Set the gridding parameters in the TGE code
    >>> n1       = 0     # Starting channel number
    >>> n2       = 767   # End channel number  
    >>> Umax     = 250   # Baselines greater than this are rejected
    >>> FWHM     = 23    # Primary beam FWHM of the telescope in degrees
    >>> f        = 0.6   # Tapering parameter value
    >>> Flag     = True  # Apply actual flagging of data
    >>> nstokes  = [0,1] # 0 for XX and 1 for YY (which polarizations you want to grid)
    >>> infile   = '/home/MWA/data_cutoff/1000000000.fits' # fits file name
    >>> nrel     = -1  # data file
    >>> GV, info = gd.grid(infile, n1, n2, nrel, Umax, FWHM, f, Flag, nstokes)
    <<<<<<<<<<< TGE >>>>>>>>>>>>
    << 19/05/26 | 07:28:16 PM >>
    Type  : Data
    << 19/05/26 | 07:30:05 PM >>
    Elapsed   :   00:01:49.04
    <<<<<<<<<<< Done >>>>>>>>>>>
    >>> print(f"{GV.shape = }")
    GV.shape = (2, 457, 457, 768)
    >>> print(f"dU   : {info[0]:.2f}\
    ...       \nUmax : {info[1]}\
    ...       \nFWHM : {info[2]} degrees\
    ...       \nf    : {info[3]}")
    dU   : 1.07      
    Umax : 250      
    FWHM : 23 degrees      
    f    : 0.6
    """
    start = datetime.now()
    print(f"<<<<<<<<<<< TGE >>>>>>>>>>>>")
    print(f"<< {start.strftime('%d/%m/%y | %I:%M:%S %p')} >>")
    
    # set seed for noise simulation
    if (seed != None):
        np.random.seed(seed)
        print(f"Seed  : {seed}")

    if nrel == -2 :
        print(f"Type  : Noise only")
    elif nrel == -1:
        print(f"Type  : Data")
    else:                     
        print(f"Type  : UAPS\nCopy  : {nrel} channel.")
    
    

    # open fits file and make a hdul object
    hdul = fits.open(infile, mode = 'readonly')

    # Convert nstokes and nrel to lists (if not) 

    nstokes = nstokes if isinstance(nstokes,list) else list([nstokes]) # If single element convert to list

    nc = (n2-n1)+1   # No of Frequency channels. 

    nu_chan0 = np.float32(hdul[0].header['CRVAL4'])  # Read the centered frequency in Hz


    theta_0 = 0.6*FWHM*np.pi/(180.)              # FWHM in radians      
    theta_w = f*theta_0
    theta_eff = f*theta_0/np.sqrt(1+f*f)         # Effective FWHM

    coeff1 = np.pi * theta_w**2                  # for window function
    coeff2 = np.pi**2 * theta_w**2               # for window function


    sf = 2                                       # Sampling factor
    dU = np.sqrt(np.log(2))/(np.pi*theta_eff*sf) # Sampling the effective PB twice.
    Nm = 3*sf                                    # sqrt(ln(500)/ln(2) = 3)


    Ng = int(np.ceil(Umax/dU) + Nm)              # Padding for convolution. Extra Nm grid pts to avoid edge effect.
    Nu = 2*Ng +1                                 # Total grid pts (Nu)

    st = nstokes    

    # Extract The baselines
    uuu = np.copy(hdul[0].data['UU'])*nu_chan0   # u values
    vvv = np.copy(hdul[0].data['VV'])*nu_chan0   # v values
    nbln= len(uuu)                               # no of baselines

    
    bluv = np.stack((uuu,vvv),axis=1)            # Make 2D array Nbl × 2

    # visibility data from hdulist

    adata = hdul[0].data['DATA'][:,0,0,0,:,:,:]  # nbl times nchannels times nstokes times 3 (real,complex,weight)


    # identify and slice  the baselines less than umax

    norms = np.linalg.norm(bluv, axis = 1)       # Computing the Norm/Magnitude
    index = np.arange(nbln)


    index = index[norms <= Umax]                 # index of baselines wihthin umax
    
    # Fold everything into upper half uv plane

    mask = bluv[:,1]<0                           # Masking
    bluv[mask,:] *= -1                           # if v<0 change v to -v and also u to -u, else do nothing    
    sign = np.ones(nbln)
    sign[mask] = -1                         


    nuv = (np.round(bluv/dU)).astype(int)        # scale to grid units


    # declare arays

    GV = np.zeros((len(nstokes),Nu, Nu,nc), dtype=np.complex128) # make grid for polarizations in nstokes list across given freq channels.

    # 1D arrays 
    oo = np.arange(-Nm,Nm+1)
    cho = np.ones(nc)


    ######   make grid 

    a,b = np.meshgrid(oo,oo,indexing='ij')
    c = np.stack((a,b),axis=2)
    c = c*dU #scale to baseline units

    #####   done


    for ii in index:  # loop over only index values that satisfy norm <= Umax


        #  calculate weights    

        diff = c + nuv[ii]*dU-bluv[ii]        # (u,v) difference from each grid pint 

        dist = np.sum(diff**2,axis=2)         # distance-squared  from each grid point

        wt =  coeff1* np.exp(-coeff2 * dist)  # wt for each grid point

        #  read data

        iidata = np.copy(adata[ii,:,:,:])     # copy visibility data for that baseline


        for st in nstokes:
            
            
            data = iidata[:,st,:]             # extract the data for that polarization


            if (nrel == -1):
                
                # leave data as it is
                vis = data[n1:n2+1,0] +  1j* sign[ii] * data[n1:n2+1,1] 
                
            elif(nrel==-2):
            
                # generate gaussian random noise with mean zero and std 1.
                vis = np.random.normal(0.0,1.0,nc) +  1j* sign[ii] * np.random.normal(0.0,1.0,nc)

            else: 

                # copy nrel th channel data across all channels
                vis = data[nrel,0] +  1j* sign[ii] * data[nrel,1]

                vis = cho*vis   # extend dimension 


            if Flag:  # Apply Flagging
                vis[data[n1:n2+1,2]<=0] = 0.


            subgrid  = np.multiply.outer(wt,vis) # wt[...,None]*vis
            csubgrid = np.conjugate(subgrid[::-1,::-1]) #np.multiply.outer(wt[::-1,::-1],np.conjugate(vis))

            # indices of the nearby grid point for that baseline
            nx = Ng+(nuv[ii,0]) 
            ny = Ng+(nuv[ii,1])

            GV[st,nx-Nm:nx+Nm+1,ny-Nm:ny+Nm+1] += subgrid

            # indices of the nearby grid point for that conjugate baseline
            nx = Ng-(nuv[ii,0])
            ny = Ng-(nuv[ii,1])

            GV[st,nx-Nm:nx+Nm+1,ny-Nm:ny+Nm+1] += csubgrid

    MM = 2*Nm
    
    hdul.close() # close the fits object
    
    end = datetime.now()
    print(f"<< {end.strftime('%d/%m/%y | %I:%M:%S %p')} >>")
    elapsed = (end - start).total_seconds()
    hours, rem = divmod(elapsed, 3600)
    minutes, seconds = divmod(rem, 60)
    print(f"Elapsed   :   {int(hours):02}:{int(minutes):02}:{seconds:05.2f}")
    print(f"<<<<<<<<<<< Done >>>>>>>>>>>")

    # return the convolved gridded visibility and grid specifications
    
    return GV[:,MM:-MM, MM:-MM, :], (dU,Umax, FWHM, f)   # slice the padding and some extra

print(f"Imported Grid,      Import Time : {datetime.now().strftime('%d/%m/%y | %I:%M:%S %p')}")
# ps_pipeline
Necessary codes to estimate power spectrum from visibility data using tge **Tapered Gridded Estimator**.

## How to install

First create a vitual environment through this :

```bash
python3 -m venv myutils        # make a virtual environment if required
source ~/myutils/bin/activate  # activate the virtual environment
```

To install the python package run this command : 

```bash
pip install git+https://github.com/Baijayanta21/ps_pipeline.git
```

While installing it will show something like this :

```bash
Collecting git+https://github.com/Baijayanta21/ps_pipeline.git
  Cloning https://github.com/Baijayanta21/ps_pipeline.git to /tmp/pip-req-build-elabqifc
  Running command git clone --filter=blob:none --quiet https://github.com/Baijayanta21/ps_pipeline.git /tmp/pip-req-build-elabqifc
  Resolved https://github.com/Baijayanta21/ps_pipeline.git to commit beada015fe90aa17aa01b2d60c7be885375d19c4
  Installing build dependencies ... done
  Getting requirements to build wheel ... done
  Preparing metadata (pyproject.toml) ... done
Collecting healpy
  Downloading healpy-1.19.0-cp310-cp310-manylinux2014_x86_64.manylinux_2_17_x86_64.whl (8.1 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 8.1/8.1 MB 2.2 MB/s eta 0:00:00
Collecting numpy
  Downloading numpy-2.2.6-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (16.8 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 16.8/16.8 MB 647.6 kB/s eta 0:00:00
Collecting scipy
  Downloading scipy-1.15.3-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (37.7 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 37.7/37.7 MB 594.4 kB/s eta 0:00:00
Collecting astropy
  Using cached astropy-6.1.7-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (10.0 MB)
Collecting numba
  Downloading numba-0.65.1-cp310-cp310-manylinux2014_x86_64.manylinux_2_17_x86_64.whl (3.7 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 3.7/3.7 MB 557.1 kB/s eta 0:00:00
Collecting packaging>=19.0
  Downloading packaging-26.2-py3-none-any.whl (100 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100.2/100.2 KB 960.7 kB/s eta 0:00:00
Collecting pyerfa>=2.0.1.1
  Using cached pyerfa-2.0.1.5-cp39-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (738 kB)
Collecting PyYAML>=3.13
  Downloading pyyaml-6.0.3-cp310-cp310-manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_28_x86_64.whl (770 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 770.3/770.3 KB 907.8 kB/s eta 0:00:00
Collecting astropy-iers-data>=0.2024.10.28.0.34.7
  Downloading astropy_iers_data-0.2026.5.25.1.14.13-py3-none-any.whl (2.0 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 2.0/2.0 MB 951.7 kB/s eta 0:00:00
Collecting llvmlite<0.48,>=0.47.0dev0
  Downloading llvmlite-0.47.0-cp310-cp310-manylinux2014_x86_64.manylinux_2_17_x86_64.whl (56.3 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 56.3/56.3 MB 754.1 kB/s eta 0:00:00
Building wheels for collected packages: myutils
  Building wheel for myutils (pyproject.toml) ... done
  Created wheel for myutils: filename=myutils-1.0.0-py3-none-any.whl size=40503 sha256=cf13905c3b744d40a237f7c94bbe1c6fcc622293ad611ac6e2b6eaa08dac6e5b
  Stored in directory: /tmp/pip-ephem-wheel-cache-7oaruyxw/wheels/cd/5e/6d/a6e8872ab66dd5967178425a09c7f19e70b0f815c466988833
Successfully built myutils
Installing collected packages: PyYAML, packaging, numpy, llvmlite, astropy-iers-data, scipy, pyerfa, numba, astropy, healpy, myutils
Successfully installed PyYAML-6.0.3 astropy-6.1.7 astropy-iers-data-0.2026.5.25.1.14.13 healpy-1.19.0 llvmlite-0.47.0 myutils-1.0.0 numba-0.65.1 numpy-2.2.6 packaging-26.2 pyerfa-2.0.1.5 scipy-1.15.3
```

After the installation to verify run :

```bash
pip show myutils
```

And it will show the following details :

```bash
Name: myutils
Version: 1.0.0
Summary: Utility functions for MWA visibility simulation, TGE and PS estimation.
Home-page: 
Author: 
Author-email: Baijayanta Bhattacharyya <baijayantabhattacharyya2021@gmail.com>
License: 
Location: /home/cts23ph/myutils/lib/python3.10/site-packages
Requires: astropy, healpy, numba, numpy, scipy
Required-by: 

```


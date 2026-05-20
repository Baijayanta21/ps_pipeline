# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'maps'
copyright = '2026, AE,SB,BB'
author = 'AE,SB,BB'
release = '1.0.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = []

templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'alabaster'
html_static_path = ['_static']
import os
import sys
sys.path.insert(0, os.path.abspath('../../'))  # Adjust the path if needed
autodoc_member_order = "bysource"

extensions = [
    'sphinx.ext.autodoc', 
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    "sphinx.ext.mathjax"
]
math_renderer = 'mathjax'
latex_engine = "pdflatex"  # Or 'xelatex', 'lualatex'

latex_elements = {
    "papersize": "a4paper",
    'fontpkg': '',
    'classoptions': ',oneside',
    'extraclassoptions': 'openany',
    "pointsize": "11pt",
    "preamble": r"""
		\renewcommand{\familydefault}{\rmdefault}
        \usepackage{setspace} % Required for more spacing options
		\parindent 0px
		\setstretch{1.1}
        \usepackage[T1]{fontenc}
        \usepackage{charter}
        \usepackage{amsmath}
        \usepackage{amsfonts}
        \usepackage{amssymb}
        \usepackage{fancyhdr}
        \usepackage{arydshln}
        \usepackage{xcolor}
        \usepackage{makeidx}  % To create an index
        \usepackage{tocloft}  % To customize Table of Contents

        % Fancy header and footer
        \pagestyle{fancy}
        \fancyhead[L]{\leftmark}
        \fancyhead[R]{\thepage}
        \fancyfoot[C]{}

        % Table of Contents: Formatting (removing subsections)
        \setcounter{tocdepth}{1}  % Limit ToC to sections only (remove subsections)
        \renewcommand{\cftsecfont}{\normalfont\bfseries}  % Bold section titles
        \renewcommand{\cftsecindent}{0em}                 % No indentation for sections
        \setlength{\cftbeforesecskip}{0.5ex}              % Space before section titles

        % Index Configuration (simplified to sections only)
        \makeindex  % Generate index

        % Page layout adjustments
		\usepackage{geometry}
        \geometry{top=1.5cm, bottom=1.5cm, left=1.5cm, right=1.5cm}
        \usepackage{hyperref}
        \hypersetup{colorlinks=true,linkcolor=violet,filecolor=magenta,urlcolor=blue}

        % Better verbatim environment for code blocks
        %\usepackage{fvextra}
        \usepackage{tcolorbox}
        
        \newtcbox{\tbox}{colframe=black,colback=gray!5,boxrule=0.7pt,arc=4pt,boxsep=5pt}
    """
}

latex_use_index = True
latex_domain_indices = False  #true is you want the python module indices
# TEX Configuration
# Enable MathJax for LaTeX equations with dollar signs ($...$)
mathjax_config = {
    "tex": {
        "inlineMath": [["$", "$"], ["\\(", "\\)"]],
        "displayMath": [["$$", "$$"], ["\\[", "\\]"]]
    }
}

pygments_style = "sphinx"  # Syntax highlighting style

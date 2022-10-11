# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html
import os 
import sys
sys.path.append(os.path.abspath(os.path.join(__file__, "..", "..", "..")))
sys.path.append(os.path.abspath(os.path.join(__file__, "..","..")))
# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import sphinx_rtd_theme  # noqa

project = 'gridds'
copyright = '2022, Alexander Ladd, Indrasis Chakraborty'
author = 'Alexander Ladd, Indrasis Chakraborty'
release = '0.0.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = ['sphinx.ext.todo', 'sphinx.ext.viewcode', 'sphinx.ext.autodoc', 'nbsphinx',\
    'sphinxcontrib.bibtex',"sphinx.ext.mathjax",  'sphinx.ext.napoleon']
extensions += ["sphinx.ext.viewcode",
    "sphinx_autodoc_typehints",
    "nbsphinx",
    "m2r2"]
nbsphinx_execute = 'never'


# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
#
source_suffix = [".rst", ".md"]

# The master toctree document.
master_doc = "index"

templates_path = ['_templates']
exclude_patterns = []

bibtex_bibfiles = ["./references.bib"]

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# html_theme = 'alabaster'

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_rtd_theme"
html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]
html_theme_options = {
    "collapse_navigation": False,
    "display_version": True,
    'navigation_depth': 3,
}

    # 'logo_only': False,

html_static_path = ['_static']

# -- Options for typehints ----------------------------------------------
always_document_param_types = True
# typehints_use_rtype = False
typehints_defaults = None  # or "comma"
simplify_optional_unions = False


add_module_names = False


latex_engine = 'xelatex'
latex_elements = {
    'fontpkg': r'''
\setmainfont{DejaVu Serif}
\setsansfont{DejaVu Sans}
\setmonofont{DejaVu Sans Mono}
''',
    'preamble': r'''
\usepackage[titles]{tocloft}
\cftsetpnumwidth {1.25cm}\cftsetrmarg{1.5cm}
\setlength{\cftchapnumwidth}{0.75cm}
\setlength{\cftsecindent}{\cftchapnumwidth}
\setlength{\cftsecnumwidth}{1.25cm}
''',
    'fncychap': r'\usepackage[Bjornstrup]{fncychap}',
    'printindex': r'\footnotesize\raggedright\printindex',
    'extraclassoptions': 'openany,oneside'
}
latex_show_urls = 'footnote'
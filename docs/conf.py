import os
import sys

sys.path.insert(0, os.path.abspath('..'))

import vhr_cloudmask  # noqa: E402

project = 'vhr-cloudmask'
copyright = '2022, Jordan A. Caraballo-Vega'
author = 'Jordan A. Caraballo-Vega'

extensions = [
    'sphinx.ext.autodoc',
    'sphinx_autodoc_typehints',
    'jupyter_sphinx.execute',
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
    "sphinx_click.ext",
    "sphinx.ext.githubpages",
    "nbsphinx",
    "autodocsumm"
]

autodoc_default_options = {"autosummary": True}

intersphinx_mapping = {
    "pyproj": ("https://pyproj4.github.io/pyproj/stable/", None),
    "rasterio": ("https://rasterio.readthedocs.io/en/stable/", None),
    "xarray": ("http://xarray.pydata.org/en/stable/", None),
}

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

master_doc = "index"
version = release = vhr_cloudmask.__version__
pygments_style = "sphinx"
todo_include_todos = False
html_theme = 'sphinx_rtd_theme'

# html_static_path = ['_static/']

myst_enable_extensions = [
    "amsmath",
    "colon_fence",
    "deflist",
    "dollarmath",
    "html_image",
]
myst_url_schemes = ("http", "https", "mailto")

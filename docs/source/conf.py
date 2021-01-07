# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys

sys.path.insert(0, os.path.abspath("../../"))
sys.path.insert(0, os.path.abspath("../../pyfstat/"))

# -- Project information -----------------------------------------------------

project = "PyFstat"
copyright = "2020, Gregory Ashton, David Keitel, Reinhard Prix, Rodrigo Tenorio"
author = "Gregory Ashton, David Keitel, Reinhard Prix, Rodrigo Tenorio"

# The full version, including alpha/beta/rc tags
release = "master"


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx_rtd_theme",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx_gallery.gen_gallery",
    "m2r2",
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_rtd_theme"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
# html_static_path = ["_static"]

# -- autodoc options ---------------------------------------------------------
# what content will be inserted into the main body of an autoclass directive
# both: the class’ and the __init__ method’s docstring are concatenated and inserted.
autoclass_content = "both"
# how to sort automatically documented members
autodoc_member_order = "bysource"

# -- Options for gallery -----------------------------------
min_reported_time = 0

examples_basedir = "../../examples/"
_, example_names, _ = next(os.walk(examples_basedir))

sphinx_gallery_conf = {
    "examples_dirs": [os.path.join(examples_basedir, case) for case in example_names],
    "gallery_dirs": example_names,
    "filename_pattern": "/PyFstat_example_",
    "plot_gallery": "False",  # our examples are slow, so we can't generate plots every time the docs are built
    "line_numbers": True,
}

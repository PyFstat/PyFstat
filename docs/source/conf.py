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

import pyfstat

sys.path.insert(0, os.path.abspath("../../"))
sys.path.insert(0, os.path.abspath("../../pyfstat/"))

# -- Project information -----------------------------------------------------

project = "PyFstat"
copyright = "2026, Gregory Ashton, David Keitel, Reinhard Prix, Rodrigo Tenorio, Maria-Antonia Ferrer"
author = (
    "Gregory Ashton, David Keitel, Reinhard Prix, Rodrigo Tenorio, Maria-Antonia Ferrer"
)

# The full version, including alpha/beta/rc tags
version = pyfstat.__version__
release = version


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx_rtd_theme",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "numpydoc",  # to keep adding "optional" tags and default values to function/class args
    "sphinx.ext.viewcode",
    "sphinx_gallery.gen_gallery",
    "m2r2",
]

source_suffix = [".rst", ".md"]

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
# move the types from the signature to the parameter list
autodoc_typehints = "description"
# ensure that even if we don't document a parameter,
# Sphinx will still show its type hint.
autodoc_typehints_description_target = "documented"
# show the default value next to the type
autodoc_preserve_defaults = True
# to fix python 3.13 introspection crash
autodoc_mock_imports = ["time"]

# --- Napoleon Settings ---
napoleon_google_docstring = False  # Focus on NumPy
napoleon_numpy_docstring = True
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = True

# --- Numpydoc Settings ---
# This recreates the "Default: X" and "(type, optional)" logic
numpydoc_show_inherited_class_members = False
numpydoc_class_members_toctree = False
numpydoc_attributes_as_param_list = (
    True  # Tell numpydoc to stop trying to validate signatures for built-ins
)
numpydoc_validation_checks = {"all", "GL01", "SA01", "EX01"}  # Optional: adjust checks
autodoc_mock_imports = ["time"]
autodoc_preserve_defaults = True  # Keeps the raw text of defaults

# -- Options for gallery -----------------------------------
min_reported_time = 0

examples_basedir = "../../examples/"
_, example_names, _ = next(os.walk(examples_basedir))

sphinx_gallery_conf = {
    "examples_dirs": [os.path.join(examples_basedir, case) for case in example_names],
    "gallery_dirs": example_names,
    "ignore_pattern": "/utils",
    "plot_gallery": "False",  # our examples are slow, so we can't generate plots every time the docs are built
    "line_numbers": True,
}

# defaultargs options
rst_prolog = (
    """
.. |default| raw:: html

    <div class="default-value-section">"""
    + ' <span class="default-value-label">Default:</span>'
)

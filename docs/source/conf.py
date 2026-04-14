# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import sphinx_bootstrap_theme

project = 'fdvar'
copyright = '2026, Imperial College London and others'
author = 'Josh Hope-Collins'
release = '2026.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx_design"
]

templates_path = ['_templates']
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'bootstrap'
html_theme_path = sphinx_bootstrap_theme.get_html_theme_path()

html_static_path = ['_static']

html_theme_options = {
    'navbar_links': [
        #  ("People", "people"),
        ("Firedrake", "https://firedrakeproject.org", True),
    ],
    'bootswatch_theme': 'cosmo',
    'source_link_position': None,
    # Render the next and previous page links in navbar. (Default: true)
    'navbar_sidebarrel': True,

    # Render the current pages TOC in the navbar. (Default: true)
    'navbar_pagenav': False,

    # Tab name for the current pages TOC. (Default: "Page")
    'navbar_pagenav_name': "Page",
}

html_sidebars = {
    '**': ['localtoc.html']
}

html_style = "fdvar.css"

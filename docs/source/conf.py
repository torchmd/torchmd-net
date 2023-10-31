# Configuration file for the Sphinx documentation builder.

# -- Project information
project = 'TorchMD-Net'
author = 'RaulPPelaez'

release = '0.1'
version = '0.1.0'

# -- General configuration
extensions = [
    'sphinx.ext.duration',
    'sphinx.ext.doctest',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
]
extensions += ['sphinxcontrib.autoprogram']
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = True
napoleon_use_admonition_for_notes = True
napoleon_use_admonition_for_references = True
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = True
napoleon_type_aliases = None
napoleon_attr_annotations = True
autosummary_ignore_module_all=False

intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'sphinx': ('https://www.sphinx-doc.org/en/master/', None),
    "torch": ("https://pytorch.org/docs/stable/", None),
}
intersphinx_disabled_domains = ['std']

templates_path = ['_templates']

# -- Options for HTML output

html_theme = 'sphinx_rtd_theme'

# -- Options for EPUB output
epub_show_urls = 'footnote'

autoclass_content = 'both'
autodoc_typehints = "none"
autodoc_inherit_docstrings = False
sphinx_autodoc_typehints = True
html_show_sourcelink = True
autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    'undoc-members': True,
    'exclude-members': '__weakref__',
    'show-inheritance': True,
}
#Exclude all torchmdnet.datasets.*.rst files in source/generated/
exclude_patterns = ['generated/torchmdnet.datasets.*.rst', 'generated/torchmdnet.scripts.*rst']
html_static_path = ['../_static']
html_css_files = [
    'style.css',
]

autodoc_mock_imports = ["torchmdnet.extensions.torchmdnet_extensions"]

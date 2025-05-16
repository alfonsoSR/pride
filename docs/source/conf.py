from pathlib import Path
import shutil
from importlib import metadata
from pride import __version__


BASE_DIR: Path = Path(__file__).parent.parent.parent
SOURCE_DIR: Path = BASE_DIR / "src"
DOCS_DIR: Path = BASE_DIR / "docs/source"

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "pride"
copyright = "2025, Alfonso Sánchez Rodríguez"
author = "Alfonso Sánchez Rodríguez"
release = ".".join(__version__.split(".")[:3])

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.duration",
    "autoapi.extension",
    "sphinx_design",
]

autoapi_dirs = [str(SOURCE_DIR)]
autoapi_file_patterns = [
    "*.pyi",
    "*.py",
]

autoapi_options = [
    "members",
    "undoc-members",
    # "private-members",
    # "show-inheritance",
    "show-module-summary",
    # "special-members",
    "imported-members",
]
autoapi_template_dir = "_templates"
autoapi_own_page_level = "function"
templates_path = ["_templates"]
exclude_patterns = []
autoapi_keep_files = False
autoapi_root = "api"
autoapi_add_toctree_entry = False

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]
html_theme_options = {
    "logo": {
        # "image_light": "_static/logo_small.png",
        # "image_dark": "_static/logo_small.png",
        "text": f"PRIDE {release}",
    },
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/alfonsoSR/pride",
            "icon": "fa-brands fa-github",
        },
        # {
        #     "name": "PyPI",
        #     "url": "https://pypi.org/project/nastro",
        #     "icon": "fa-custom fa-pypi",
        # },
    ],
    "secondary_sidebar_items": {"**": ["page-toc", "sourcelink"]},
    "collapse_navigation": True,
    "header_links_before_dropdown": 6,
    "navbar_align": "content",
    "show_nav_level": 1,
    "navbar_start": ["navbar-logo"],
    "navbar_center": ["navbar-nav"],
    "navbar_end": [
        # "version-switcher",
        "theme-switcher",
        "navbar-icon-links",
    ],
    "footer_start": ["copyright"],
    "footer_center": ["sphinx-version"],
    "use_edit_page_button": True,
    "show_version_warning_banner": True,
    "back_to_top_button": True,
}


def skip_submodules(app, what, name, obj, skip, options):

    ignored_submodules = ("pride.cli", "pride.data", "pride._version")

    match what:

        # Modules and submodules
        case "module" | "package":

            # Skip ignored submodules
            for ignored_submodule in ignored_submodules:
                if ignored_submodule in name:
                    return True

            # Skip second-level submodules
            if name.count(".") > 1:
                return True

            # Expose the rest
            return False

        # Attributes & properties
        case "attribute" | "property":

            return True

        case _:

            return skip


def setup(sphinx):

    sphinx.connect("autoapi-skip-member", skip_submodules)

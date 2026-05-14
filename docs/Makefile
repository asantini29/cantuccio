# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = .
BUILDDIR      = _build

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)


docs-apidoc:
	uv run sphinx-apidoc -f -e -M -o api ../src/cantuccio

docs: docs-apidoc
	uv run sphinx-build -b html . _build/html

docs-clean:
	rm -rf docs/_build docs/api

docs-serve:
	uv run python -m http.server --directory _build/html 8000

docs-watch:
	uv run sphinx-autobuild . _build/html --open-browser

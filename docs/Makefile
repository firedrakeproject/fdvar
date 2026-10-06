# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?= -W
SPHINXBUILD   ?= sphinx-build
SPHINXAUTOBUILD ?= sphinx-autobuild
SOURCEDIR     = source
BUILDDIR      = build

GENERATED_FILES = source/demos/* source/generated/*

.PHONY: help Makefile clean copy_demos

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

clean:
	rm -r $(BUILDDIR)/*
	rm -r $(GENERATED_FILES)

copy_demos:
	install -d source/demos
	cp ../demos/*/*.rst source/demos
	for file in source/demos/*.py.rst; do pylit $$file; done

apidoc: copy_demos
	sphinx-apidoc --force --no-toc --module-first -o "$(SOURCEDIR)" ../fdvar 

html: Makefile copy_demos
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" $(BUILDDIR) $(SPHINXOPTS) $(O)
	cp source/demos/*.py $(BUILDDIR)/$@/demos/

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).

auto-%: Makefile copy_demos
	@$(SPHINXAUTOBUILD) -M $* "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

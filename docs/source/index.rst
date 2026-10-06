.. title:: fdvar: 4DVar data assimilation methods using Firedrake

fdvar documentation
===================

``fdvar`` is a Python library for parallel-in-time 4DVar data assimilation using
the `Firedrake <https://www.firedrakeproject.org/>`_ finite element library and the
`PETSc/TAO <https://petsc.org/>`_ optimisation library.

``fdvar`` uses Firedrake and `Pyadjoint <https://pyadjoint.org/>`_'s automatic
differentiation capabilities to automate the construction of the 4DVar system and
it's derivatives, and provides a variety of preconditioners for solving the
optimisation problem using TAO.

.. contents::

Features
--------

* Automatic construction of derivatives from forward model.
* Automatic construction of solvers from reduced functional.
* Time-parallel preconditioners.
* Access to range of composable optimisation and linear solvers via PETSc/TAO.
* Works with any PDE expressable using the Unified Form Language.

Installation
------------

#. Install Firedrake `using the instructions here <https://www.firedrakeproject.org/>`_

#. Install ``fdvar`` into the Firedrake virtual environment.
   There are two installation options:

   * Coming soon ... Pypi release compatible with the latest Firedrake release.
 
   * Cloning the github repository:

     #. ``git clone https://github.com/firedrakeproject/fdvar.git``

     #. ``pip install './fdvar'``

#. We recommend using the excellent implicit Runge-Kutta library `Irksome <https://www.firedrakeproject.org/Irksome/>`_ for timestepping. You can either:

   * follow the instructions on the Irksome website, or

   * use the ``demos`` optional dependency in the pip install step above: ``pip install './fdvar[demos]'``.

Getting Started
---------------

* To get started with Firedrake check out the `introductory demos <https://www.firedrakeproject.org/documentation.html#introductory-tutorials>`_.
* The best place to get started with ``fdvar`` is the :ref:`demos`.
* A complete listing of the ``fdvar`` library can be found in the :py:mod:`API documentation <fdvar>`.

.. _demos:

Demos
-----

A Python script is generated for each demo and is linked at the bottom of the demo.

.. toctree::
   :maxdepth: 1

   WC4DVar for the advection-diffusion equation.<demos/wc4dvar_advection.py>

Getting in touch
----------------

Please get in touch with any questions related to ``fdvar`` by raising an issue on the `GitHub repository <https://github.com/firedrakeproject/fdvar/>`_.
For more general Firedrake queries please see the `Firedrake contact page <https://www.firedrakeproject.org/contact/>`_

Contributors
------------

``fdvar`` has been developed by:

* `Josh Hope-Collins <https://profiles.imperial.ac.uk/joshua.hope-collins13>`_ (Department of Mathematics, Imperial College London)

* `David A. Ham <https://profiles.imperial.ac.uk/david.ham>`_ (Department of Mathematics, Imperial College London)

* `Jemima M. Tabeart <https://jemimat.github.io/>`_ (Department of Mathematics and Computer Science, TU Eindhoven)

.. toctree::
   :maxdepth: 3
   :caption: Contents:

   generated/modules.rst

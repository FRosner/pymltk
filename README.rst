.. image:: https://coveralls.io/repos/github/bkomboz/pymltk/badge.svg?branch=master
     :target: https://coveralls.io/github/bkomboz/pymltk?branch=master
     :alt: Code Coverage Status
.. image:: https://requires.io/github/bkomboz/pymltk/requirements.svg?branch=master
     :target: https://requires.io/github/bkomboz/pymltk/requirements/?branch=master
     :alt: Requirements Status
.. image:: https://readthedocs.org/projects/pymltk/badge/?version=latest
     :target: http://pymltk.readthedocs.io/en/latest/?badge=latest
     :alt: Documentation Status

======
pymltk
======

Python Machine Learning Toolkit


Description
===========

*pymltk* is a Python package helping data scientists with
their daily work of (pre)processing data and building predictive
or other machine learning models. It *offers various Python
functions which implement common operations done by data
scientists during their daily work*.

All functions of this package ...

* ... do one thing and (try to) do it well.
* ... operate on pandas as well as dask dataframes.
* ... are fully tested and documented.
* ... offer a clean and consistent UI.

This package was inspired by `mlr <https://github.com/mlr-org/mlr>`_,
a R package which offers similar functionality with respect to data
(pre)processing (but in addition offers *a lot* more).


Function Overview
=================

* *parse_columns*: Parsing features with a specified dtype.
* *parse_missings*: Parsing specified values as missing values.
* *merge_levels*: Merging levels/values of a feature depending on several criteria.
* *impute_missings*: Imputing missing values based on several strategies.
* *remove_constants*: Removing features with no/low variability.


Installation
============

Currently only the development version in this repository is available. In the future,
a stable release on pypi is planned.


Documentation
=============

A detailed documentation of each function provided by *pymltk* is
available on `readthedocs.org <http://pymltk.readthedocs.io/en/latest/?>`_.


Contribution
============

Feature requests and bug reports are very welcome. Please open an issue
in the github issue tracker of the respository of this project. Pull requests
implementing new functionality are, of course, also welcome. Please open
in addition also an issue for those.


License
=======

*pymltk* is licensed under the Apache License Version 2.0.
For details please see the file called LICENSE.


Note
====

This project has been set up using PyScaffold 2.5.6. For details and usage
information on PyScaffold see http://pyscaffold.readthedocs.org/.

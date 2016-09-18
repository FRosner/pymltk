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
- ... do one thing and (try to) do it well.
- ... operate on pandas as well as dask dataframes.
- ... are fully tested and documented.
- ... offer a clean and consistent UI.

This package was inspired by `mlr <https://github.com/mlr-org/mlr>`_,
a R package which offers similar functionality with respect to data
(pre)processing (but in addition offers *a lot* more).


Function Overview
=================

- *parse_columns*: Parsing features with a specified dtype.
- *parse_missings*: Parsing specified values as missing values.
- *merge_levels*: Merging levels/values of a feature depending on several criteria.
- *impute_missings*: Imputing missing values based on several strategies.
- *remove_constants*: Removing features with no/low variability.


Installation
============

A stable version of *pymlkt* is available on PyPI and can be installed via pip:::

    pip install pymlk

The development version is available in the master branch of this respository.


Documentation
=============

A detailed documentation for each function provided by *pymltk* is via
`read-the-docs.com`_.


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

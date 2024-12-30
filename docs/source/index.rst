bayes_cn_hfs
============

``bayes_cn_hfs`` implements two models for molecular hyperfine spectroscopy. The first is ``CNModel``, which can model the emission of ``CN`` or ``13CN``.
The second is ``CNRatioModel``, which predicts both ``CN`` and ``13CN`` observations in order to infer the ``12C/13C`` isotopic ratio. ``bayes_cn_hfs`` 
is written in the ``bayes_spec`` Bayesian modeling framework, which provides methods to fit these models to data using Monte Carlo Markov Chain techniques.

Useful information can be found in the `bayes_cn_hfs Github repository <https://github.com/tvwenger/bayes_cn_hfs>`_, 
the `bayes_spec Github repository <https://github.com/tvwenger/bayes_spec>`_, and in the tutorials below.

============
Installation
============
.. code-block::

    conda create --name bayes_cn_hfs -c conda-forge pymc pip
    conda activate bayes_cn_hfs
    pip install bayes_cn_hfs

.. toctree::
   :maxdepth: 2
   :caption: Tutorials:

   notebooks/cn_model
   notebooks/cn_model_anomalies
   notebooks/cn_ratio_model
   notebooks/cn_ratio_anomalies
   notebooks/g211.59.ipynb
   notebooks/iram_data.ipynb
   notebooks/alma_data.ipynb

.. toctree::
   :maxdepth: 2
   :caption: API:

   modules

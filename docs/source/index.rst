bayes_cn_hfs
============

``bayes_cn_hfs`` implements two models for molecular hyperfine spectroscopy. The first, ``HFSModel`` and ``HFSAnomalyModel``, are general-purpose models for any molecular hyperfine spectrum. The others, ``CNRatioModel`` and ``CNRatioAnomalyModel``, predict ``CN`` and ``13CN`` observations in order to infer the ``12C/13C`` isotopic ratio. ``bayes_cn_hfs`` is written in the ``bayes_spec`` Bayesian modeling framework, which provides methods to fit these models to data using Monte Carlo Markov Chain techniques.

Useful information can be found in the `bayes_cn_hfs Github repository <https://github.com/tvwenger/bayes_cn_hfs>`_, the `bayes_spec Github repository <https://github.com/tvwenger/bayes_spec>`_, and in the tutorials below.

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

   notebooks/hfs_model
   notebooks/hfs_anomaly_model
   notebooks/cn_ratio_model
   notebooks/cn_ratio_anomaly_model
   notebooks/real_data

.. toctree::
   :maxdepth: 2
   :caption: API:

   modules

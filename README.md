EnzyFind
==========


Description
-----------

EnzyFinder is a lightweight tool for chemical biology and metabolic engineering that helps you to quickly find candidate enzymes from “small molecules” that are likely to be associated with their reactions.

All you need to do is enter a small molecule (SMILES) and the enzyme (sequence) to be screened, and EnzyFinder will automatically predict which enzymes will catalyze the reaction of interest and generate a list of candidate enzymes!


Requirements
------------

EnzyFinder is developed on Linux and supports both GPUs and CPUs. The Linux version has been fully validated. EnzyFinder on Windows supports both GPUs and CPUs, but windous  have only been partially tested, so support is limited.

The code is written in Python 3 (>= 3.8). A list of dependencies can be found in the repositories (see also Installation below).

The code was implemented and tested on linux with the following packages.
- python 
- jupyter
- pandas 
- torch 
- numpy 
- rdkit
- unimol_tools
- fair-esm 
- py-xgboost 
- matplotlib 
- hyperopt 
- sklearn 
- pickle
- Bio 
- re 


Installation
------------

1. Clone this Git repository.
1. Create a Python environment and install a compatible version of Python, for example with [Conda](https://conda.io/projects/conda/en/latest/index.html)
    ```shell
    conda env create -f environment.yml
    conda activate EnzyFind
    ```

Basic Usage
-----------

REINVENT is a command line tool and works principally as follows
```shell
reinvent -l sampling.log sampling.toml
```

This writes logging information to the file `sampling.log`.  If you wish to write
this to the screen, leave out the `-l sampling.log` part. `sampling.toml` is the
configuration file.  The main format is [TOML](https://toml.io/en/) as it tends to be more user friendly.  JSON and YAML are supported too.

Sample configuration files for all run modes are
located in `configs/` in the repository. File paths in these files would need to be
adjusted to your local installation.  You will need to choose a model and the
appropriate run mode depending on the research problem you are trying to address.
There is additional documentation in `configs/` in several `*.md` files with
instructions on how to configure the TOML file.  Internal priors can be referenced with a
dot notation (see `reinvent/prior_registry.py`).


Tutorials / `Jupyter` notebooks
-------------------------------

Basic instructions can be found in the comments in the config examples in `configs/`.

Notebooks are provided in the `notebooks/` directory.  Please note that we
provide the notebooks in jupytext "light script" format.  To work with the light
scripts you will need to install jupytext.  A few other packages will come in handy too.

```shell
pip install jupytext mols2grid seaborn
```

The Python files in `notebooks/` can then be converted to a notebook e.g.

```shell
jupytext -o Reinvent_demo.ipynb Reinvent_demo.py
```


Scoring Plugins
---------------

The scoring subsystem uses a simple plugin mechanism (Python
[native namespace packages](https://packaging.python.org/en/latest/guides/packaging-namespace-packages/#native-namespace-packages)).  If you
wish to write your own plugin, follow the instructions below.
There is no need to touch any of the REINVENT code. The public
repository contains a [contrib](https://github.com/MolecularAI/REINVENT4/tree/main/contrib/reinvent_plugins/components) directory with some useful examples.

1. Create `/top/dir/somewhere/reinvent\_plugins/components` where `/top/dir/somewhere` is a convenient location for you.
1. Do **not** place a `__init__.py` in either `reinvent_plugins` or `components` as this would break the mechanism.  It is fine to create normal packages within `components` as long as you import those correctly.
1. Place a file whose name starts with `comp_*` into `reinvent_plugins/components` or subdirectories.   Files with different names will be ignored i.e. not imported. The directory will be searched recursively so structure your code as needed but directory/package names must be unique.
1. Tag the scoring component class(es) in that file with the @add\_tag decorator.  More than one component class can be added to the same *comp\_* file. See existing code.
1. Tag at most one dataclass for parameters in the same file, see existing code.  This is optional.
1. Set or add `/top/dir/somewhere` to the `PYTHONPATH` environment variable or use any other mechanism to extend `sys.path`.
1. The scoring component should now automatically be picked up by REINVENT.


Unit and Integration Tests 
--------------------------

This is primarily for developers and admins/users who wish to ensure that the
installation works.  The information here is not relevant to the practical use
of REINVENT.  Please refer to _Basic Usage_ for instructions on how to use the 
`reinvent` command.

The REINVENT project uses the `pytest` framework for its tests.  Before you run
them you first have to create a configuration file for the tests.

In the project directory, create a `config.json` file in the `configs/` directory.
You can use the example config `example.config.json` as a base.  Make sure that
you set `MAIN_TEST_PATH` to a non-existent directory.  That is where temporary
files will be written during the tests.  If it is set to an existing directory,
that directory will be removed once the tests have finished.

Some tests require a proprietary OpenEye license.  You have to set up a few
things to make the tests read your license.  The simple way is to just set the
`OE_LICENSE` environment variable to the path of the file containing the
license.  

Once you have a configuration and your license can be read, you can run the tests.

```
$ pytest tests --json /path/to/config.json --device cuda
```

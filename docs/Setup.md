
Setup
------------
### Preconditions: 
- Python 3.6 locally installed 
(e.g.:  <a target="_blank" href="https://www.anaconda.com/download/#macos">Anaconda</a>)
- Installed nvidia drivers, cuda and cudnn 
(e.g.:  <a target="_blank" href="https://www.tensorflow.org/install/gpu">Tensorflow</a>)

### Local setup
- Clone repository
```
git clone %reponame%
cd %reponame%
```
- Create a conda environment from enrionment.yaml (environment name will be cmr-las)
```
conda env create --file environment.yaml
```
- Activate environment
```
conda activate phase_detection
```
- Install a helper to automatically change the working directory to the project root directory
```
pip install --extra-index-url https://test.pypi.org/simple/ ProjectRoot
```
- Create a jupyter kernel from the activated environment, this kernel will be visible in the jupyter lab
```
python -m ipykernel install --user --name pdet --display-name "phase_det kernel"
```


### Enable interactive widgets in Jupyterlab
Pre-condition: nodejs installed globally or into the conda environment. e.g.:
```
conda install -c conda-forge nodejs
```
- Install the jupyterlab-manager which enables the use of interactive widgets
```
jupyter labextension install @jupyter-widgets/jupyterlab-manager
```
  
Further infos on how to enable the jupyterlab-extensions:

<a target="_blank" href="https://ipywidgets.readthedocs.io/en/latest/user_install.html#installing-the-jupyterlab-extension">JupyterLab</a>

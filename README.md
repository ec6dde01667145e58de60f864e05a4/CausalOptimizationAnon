# A Meta-Transfer Objective for Learning to Disentangle Causal Mechanisms

## Getting started
To avoid any conflict with your existing Python setup, and to keep this project self-contained, it is suggested to work in a virtual environment with [`virtualenv`](http://docs.python-guide.org/en/latest/dev/virtualenvs/). To install `virtualenv`:
```
pip install --upgrade virtualenv
```
Create a virtual environment, activate it and install the requirements in [`requirements.txt`](requirements.txt).
```
virtualenv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Experiments
 - The experiments on discrete variables with tabular representation (Section 3.3) and continuous multimodal variables (Appendix C.3) are available as notebooks in the [`notebooks`](notebooks/) folder.
 - The experiments on discrete variables with multi-layer perceptrons parametrization can run with `python run_mlp.py`.

This repository contains the code and experiment scripts for the paper [Measuring the Effect of Training Data on Deep Learning Predictions via
Randomized Experiments](https://arxiv.org/abs/2206.10013). If you find it useful, please consider citing:

```bibtex
@InProceedings{pmlr-v162-lin22h,
  title = 	 {Measuring the Effect of Training Data on Deep Learning Predictions via Randomized Experiments},
  author =       {Lin, Jinkun and Zhang, Anqi and L{\'e}cuyer, Mathias and Li, Jinyang and Panda, Aurojit and Sen, Siddhartha},
  booktitle = 	 {Proceedings of the 39th International Conference on Machine Learning},
  pages = 	 {13468--13504},
  year = 	 {2022},
  editor = 	 {Chaudhuri, Kamalika and Jegelka, Stefanie and Song, Le and Szepesvari, Csaba and Niu, Gang and Sabato, Sivan},
  volume = 	 {162},
  series = 	 {Proceedings of Machine Learning Research},
  month = 	 {17--23 Jul},
  publisher =    {PMLR},
  pdf = 	 {https://proceedings.mlr.press/v162/lin22h/lin22h.pdf},
  url = 	 {https://proceedings.mlr.press/v162/lin22h.html},
}
```

## Install
- Requirements

    ```bash
    # Instal a patched version of glmnet
    git clone https://github.com/lazycal/glmnet_python.git
    cd glmnet_python
    python setup.py install
    ```
    Running the experiments (in the example folder) additionally require `cvxpy`, `shap`, `shapley-regression`, PyTorch and Torchvision. The tested versions can be installed as follows:
    ```bash
    pip3 install cvxpy==1.2.0 shap==0.40.0
    git clone https://github.com/iancovert/shapley-regression.git
    cd shapley-regression
    pip3 install .
    pip3 install torch==1.12.0+cu116 torchvision==0.13.0+cu116 --extra-index-url https://download.pytorch.org/whl/cu116
    ```

- Install: simply add the repo folder to PYTHONPATH:

    ```bash
    export PYTHONPATH=/path/to/repo:$PYTHONPATH
    ```

## Usage

### Explaining Model Behaviors

1. Implement the behavior (aka utility) function that quantifies the model behaviors by subclassing `ame.Query`. 
2. Provide a model training algorithm by subclassing `ame.ModelTrain`. See the comments and a sample implementation in `example/ame_cifar10/expl.py` for both Step 1 & 2. 
3. Run the following lines to estimate the AME scores:

    ```python
    ame_expl = ame.AMEExplainer()
    A = MyModelTrain(...) # subclass of ame.Query
    Q = MyQuery(...) # subclass of ame.ModelTrain
    n = ... # dataset size
    m = ... # how many samples you want
    ame_scores, _ = ame_expl.explain(A, Q, n, m)
    ```
4. (Optional) run Knockoffs-based selection. Please refer to `print_selection` in `example/ame_cifar10/expl.py` for an example usage.

### Shapley Value Estimation

1. Implement the utility function by subclassing `sv_estimator.GameBase`. See `example/sv_threshold/main.py` for an example.
2. Run the following python code to estimate:

    ```python
    estimator = AMESV(game=ThresholdGame(), eps=0.01)
    m = ... # sample size
    res = estimator.estimate(m)
    ```
## Examples

The `example` folder contains 3 examples and a few experiment scripts:
1. `example/ame_cifar10`: the same setup as the CIFAR10-20 and CIFAR10-50 as in the paper.
2. `example/sv_threshold`: the same setup as the simulated dataset for Shapley value estimation.
3. `example/sv_mnist`: the same setup as the poisoned MNIST dataset for Shapley value estimation.
## MNIST

### The dataset
Needs no introduction...

### The code
* The script ``run.sh`` trains the network for 10 different initializations of the weights. Hyperparameters are read from the configuration files in the ``experiments`` folders for each mode (no-, standard-, annealed- and curriculum-dropout). Accuracies are stored in the respective sub-folders.

```
sudo chmod a+x run.sh
./run.sh
```
* The script ``analyseResults_mnist_average.py`` plots the average accuracies as a function of the training iterations.
```
python analyseResults_mnist_average.py
```

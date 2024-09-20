ICASSP 2025 Implementation of ACME
==================================

The script `icassp25/cifar100.acme.py` was used to perform the experimentation published in the
paper *"Automatic Adaptation of the Step Size in Gradient Descent Training"*, submitted
to ICASSP 2025.

To get a description of the different options and arguments of the script, execute it with the option `--help`:

```console
$ icassp25/cifar100.acme.py --help
usage: cifar100.acme.py [-h] [--data-dir STR] [--batch-size INT] [--model STR]
                        [--depth INT] [--width-alpha INT] [--bottleneck]
                        [--dropout FLOAT] [--epochs INT] [--lr FLOAT]
                        [--label-smoothing FLOAT] [--momentum FLOAT]
                        [--threads INT] [--rho FLOAT] [--eta FLOAT]
                        [--weight-decay FLOAT] [--acme-script SCRIPT] [--acme]
                        [--asam] [--adam] [--nesterov] [--wandb LOGFILE]

options:
  -h, --help            show this help message and exit
  --data-dir STR        Directory with the CIFAR100 database [default: download from repository]
  --batch-size INT      Batch size used in the training and validation loop [default: 200].
  --model STR           Model to use (WRN or Pyramid) [default: WRN].
  --depth INT           Number of layers of WRN or Pyramid [default: 40 for WRN, 110 for Pyramid].
  --width-alpha INT     Width of WRN or alpha in Pyramid [default: 8 for WRN, 40 for Pyramid].
  --bottleneck          Use bottleneck in Pyramid.
  --dropout FLOAT       Dropout rate [default: 0.0].
  --epochs INT          Total number of epochs [default: 200].
  --lr FLOAT            Base learning rate at the start of the training [default: 0.1].
  --label-smoothing FLOAT
                        Use 0.0 for no label smoothing [default: 0.1].
  --momentum FLOAT      SGD Momentum [default: 0.9].
  --threads INT         Number of CPU threads for dataloaders [default: 2].
  --rho FLOAT           Rho parameter for SAM [default: 1.0].
  --eta FLOAT           Eta parameter for SAM [default: 0.0].
  --weight-decay FLOAT  L2 weight decay [default: 1.00e-04].
  --acme-script SCRIPT  Name of acme script [default: acme].
  --acme                Use Acme algorithm for learning rate adaption.
  --asam                Use ASAM optimizer.
  --adam                Use ADAM optimizer.
  --nesterov            Use Nesterov accelerated gradient.
  --wandb LOGFILE       Name of the wandb run [default: do not use wandb].
```

Some remarks about the above arguments and options:

| argument             | remarks                                                             |
| -------------------- | ------------------------------------------------------------------- |
| `--data-dir` | If this option is used, its argument must be the name of the directory where the CIFAR100 database is stored. If not used, the database will be downloaded from the official repository and stored in the directory `./data`. From then on, it is advisable to use the option `--data-dir ./data` in order to use the already downloaded database |
| `--model`    | Neural netowork to be used. Options are `WRN`, for Wide ResNet, or `Pyramid`, for PyramidNet. Options `--depth`, `--width-alpha`, and `--bottleneck` affect the depth and size of the corresponding one |
| `--momentum` | *Momentum* used in Exponential Memory Average for estimating the mean of the gradients. In ADAM, the mean of their uncentered variances is always estimated with a momentum equal to `0.999` |
| `--rho` and `--eta` | Parameters of the Adaptive Sharpness Aware Minimization algorithm (ASAM) |
| `--weight-decay` | Weight decay used in L2 regularization. In ADAM, the ADAMW implementation is used |
| `--acme-script` | Name of the directory (module) where `acme.py` is stored. Not used in this way during our experimentation, but it does work if the script is executed from the main directory of this repository or the directory of `acme.py` is included in `$PYTHONPATH` |

Other remarks:

- ACME, ADAM, and ASAM may be independtly selected. If none of them is
  selected, standard SGD will be used. Also, Nesterov accelerated
  gradient descent may be applied regardless of the other options.
- ACME is not able to perform `dropout` correctly. Work is in progress...
- CrossEntropyLoss is always used with a smoothing that can be set with
  the `--label-smoothing` option (it defaults to `0.1`)
- Learning rate scheduling is performed with `CosineAnnealingLR`, except
  when using ACME. In ACME, the same scheduling is performed internally
  assuming 200 epochs with batches of 200 images. With other parameters
  for the number of epochs or the size of the batches, the results using
  ACME are unpredictable.

Sample executions:
------------------

- Standard SGD training, assuming the databaes is already downloaded in
  directory `./data`. The initial learning rate is set to `1e-2`:

```sh
icassp25/cifar100.acme.py --data data --lr 1e-2
```

- Complete training, using ACME, ADAM, ASAM and Nesterov:

```sh
icassp25/cifar100.acme.py --data data --acme --adam --asam --nesterov
```

Acknowledgment
--------------

The script `icassp25/cifar100.acme.py` was written using
`example/train.py` from David Samuel's repository [(Adaptive) SAM
Optimizer](https://github.com/davda54/sam) as a model and starting
point.

Many thanks to him and his collaborators (or whoever wrote
that script in the first instance.)

ACME - Automatic Adaptation of the Step Size in Gradient Descent Training
=========================================================================

<img src="img/coyote.png" height="288" width="512"/>

This repository contains the PyTorch implementation that was used in the experimentation
of the paper *"Automatic Adaptation of the Step Size in Gradient Descent Training"*, submitted
to ICASSP 2025.

Installation
------------

Place the file `acme/acme.py` somewhere in your `PYTHONPATH` environment and import the
class `Acme`.

Construction of the optimizer
-----------------------------

Create the optimizer with the order `Acme(model, **args)`, where `model` may be either the
model to be optimized or its parameters (`model.params`). The reason of the duplicity is that
SAM/ASAM requires the field `params_with_name` of the model. If SAM/ASAM is not used, `Acme`
may replace other typical PyTorch optimizers, such as SGD or Adam, where `model.params` is
typically used.

### Other arguments of the constructor

| Argument | Description | Default value |
| -------- | ----------- | ------------- |
| `lr` | Initial Step Size | `1e-3` |
| `betas` | Exponential memory average coefficients (two are needed for ADAM) | `(0.9, 0.999)` |
| `epsilon` | Floor epsilon used in ADAM to prevent overflows (might have an impact in convergence) | `1e-8`|
| `weight_decay` | Weight decay used in L2 regularization (in ADAM, ADAMW is used) | `0` |
| `acme` | Use the ACME adaption of the step size | `False` |
| `adam` | Use ADAM algorithm | `False` |
| `asam` | Use ASAM algorithm | `False` |
| `rho` | Rho parameter of ASAM | `1.0` |
| `eta` | Eta parameter of ASAM | `0.0` |
| `nesterov` | Apply Nesterov accelerated gradient descent | `False` |
| `bias_correction` | Use bias correction to the Exponential moving average (decoupled from `adam`) | `False` |
| `differentiable` | Not used in `Acme`, but needed to be present for unknown to us reasons| `False` |

Use of the optimizer (`Acme.step()`)
------------------------------------

`Acme` performs a two step algorithm that requires the loss function and its gradient to be estimated
twice with exactly the same data. In order to do so, a `closure()` function is needed as its argument.

The `closure` must estimate the loss function and its gradient for every batch used in the training,
and return the loss.

For instance, the following code was used in the ICASSP 2025 submission:

```python
def closure():
    global correct

    with torch.enable_grad():
        # compute output
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        correct = (torch.argmax(outputs, 1) == targets).sum().div(len(targets))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()

        return loss
```

Then, the actual call to the optimizer becomes:

```python
    optimizer.step(closure)
```

Experimentation on CIFA100 submitted to ICASSP 2025
---------------------------------------------------

The script `cifar100.acme.py` in directory [icassp25](icassp25/README.md) provides an example
for using ACME with several options in the recognition of the CIFAR100 image database.
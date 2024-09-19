import sys

from typing import List, Optional
import numpy as np

import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer

__all__ = ['Acme']

limita = 2


def lrCubic(l0, l1, g0, g1, lr, beta, exp_avgs):
    l0, l1, g0, g1, lr = float(l0), float(l1), float(g0), float(g1), float(lr)

    mellora = l1 <= l0 # or l1 - l0 < l1 / 1.e8
    significativo = abs(l0 - l1) > max(l0, l1) / 1.e4

    a3 = -(g0 + g1) / lr ** 2 - 2 * (l1 - l0) / lr ** 3
    b3 = (2 * g0 + g1) / lr + 3 * (l1 - l0) / lr ** 2
    c3 = -g0
    d3 = l0

    a2act = -g1/lr - (l1 - l0)/lr**2
    b2act = g1 + 2 * (l1 - l0) / lr
    c2act = l0

    a2prev = g0 / lr + (l1 - l0) / lr**2
    b2prev = -g0
    c2prev = l0

    lr_opt = lr if mellora or not significativo else 0
    lr_opt = lr if mellora else 0
    lr_ = lr * (1 + beta) / beta if mellora else 0

    raiz = b3 ** 2 - 3 * a3 * c3

    if not significativo:
        lr_ = 2 * lr if mellora else lr
        cod = 0
    elif g0 > 0 and g1 < 0:
        lr_opt = (-b3 + raiz ** 0.5) / (3 * a3)
        lr_ = lr_opt
        cod = 1 if mellora else 2
    elif mellora and g1 < 0:
        lr_opt = -b2act / a2act / 2
        lr_ = lr_opt
        cod =  3
    elif not mellora and g0 > 0:
        lr_opt = -b2prev / a2prev / 2
        lr_ = lr_opt
        cod =  4
    elif mellora and g0 > 0 and g1 > 0:
        F_ = 2
        F = (F_ - beta) / (1 - beta)
        lr__ = lr * (1 + g1 / g0)
        lr_ = min(lr__, F * lr)
        cod = 5
    elif not mellora and g0 < 0 and g1 < 0:
        F_ = 1 / 2
        F = (F_ - beta) / (1 - beta)
        lr__ = -lr * g1 / g0
        lr_ = max(lr__, F * lr)
        cod = 6
    else:
        lr_ = lr if mellora else 0
        cod = 7

    return lr_, lr_opt, cod, a3, b3, c3, d3


def _use_grad_for_differentiable(func):
     def _use_grad(self, *args, **kwargs):
         prev_grad = torch.is_grad_enabled()
         try:
             torch.set_grad_enabled(self.defaults['differentiable'])
             ret = func(self, *args, **kwargs)
         finally:
             torch.set_grad_enabled(prev_grad)
         return ret
     return _use_grad


class Acme(Optimizer):
    def __init__(self, model, lr=1e-3, betas=(0.9, 0.999), epsilon=1e-8,
                 weight_decay=0, *,
                 acme=True, adam=True, asam=None, rho=1.0, eta=0, nesterov=False,
                 bias_correction=True,
                 differentiable: bool = False):

        if hasattr(model, 'parameters'):
            params = model.parameters()
        else:
            params = model

        self.model = model
        self.rho = rho
        self.eta = eta

        self.scheduled_lr = 1
        self.schedule_total = 50_000

        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= epsilon:
            raise ValueError("Invalid epsilon value: {}".format(epsilon))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, LR=lr, betas=betas, epsilon=epsilon, weight_decay=weight_decay,
                        acme=acme, adam=adam, asam=asam, rho=rho, eta=eta, nesterov=nesterov, step=1,
                        bias_correction=bias_correction, differentiable=differentiable)
        super().__init__(params, defaults)


    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault('acme', True)
            group.setdefault('adam', True)
            group.setdefault('asam', None)
            group.setdefault('nesterov', False)
            group.setdefault('bias_correction', True)
            group.setdefault('step', 1)
            group.setdefault('differentiable', False)
        state_values = list(self.state.values())


    @torch.no_grad()
    def asam_ascent_step(self):
        wgrads = []
        for n, p in self.model.named_parameters():
            if p.grad is None:
                continue
            t_w = self.state[p].get("eps")
            if t_w is None:
                t_w = torch.clone(p).detach()
                self.state[p]["eps"] = t_w
            if 'weight' in n:
                t_w[...] = p[...]
                t_w.abs_().add_(self.eta)
                p.grad.mul_(t_w)
            wgrads.append(torch.norm(p.grad, p=2))
        wgrad_norm = torch.norm(torch.stack(wgrads), p=2) + 1.e-16
        for n, p in self.model.named_parameters():
            if p.grad is None:
                continue
            t_w = self.state[p].get("eps")
            if 'weight' in n:
                p.grad.mul_(t_w)
            eps = t_w
            eps[...] = p.grad[...]
            eps.mul_(self.rho / wgrad_norm)
            p.add_(eps)


    @torch.no_grad()
    def sam_ascent_step(self):
        grads = []
        for n, p in self.model.named_parameters():
            if p.grad is None:
                continue
            grads.append(torch.norm(p.grad, p=2))
        grad_norm = torch.norm(torch.stack(grads), p=2) + 1.e-16
        for n, p in self.model.named_parameters():
            if p.grad is None:
                continue
            eps = self.state[p].get("eps")
            if eps is None:
                eps = torch.clone(p).detach()
                self.state[p]["eps"] = eps
            eps[...] = p.grad[...]
            eps.mul_(self.rho / grad_norm)
            p.add_(eps)


    ascent_step = asam_ascent_step


    def _init_group(
        self,
        group,
        params_with_grad,
        prev_params,
        grads,
        prev_grads,
        exp_avgs,
        exp_avg_sqs,
    ):
        if 'step' not in group:
            group['step'] = 1

        if 'vars' not in group:
            group['vars'] = []
            for cod in range(-1, 8):
                group[cod] = 0

        adam = group['adam']
        acme = group['acme']
        bias_correction = group['bias_correction']
        beta1, beta2 = group['betas']

        for p in group['params']:
            if p.grad is not None:
                params_with_grad.append(p)
                grads.append(p.grad)

                state = self.state[p]
                if len(state) == 0:
                    state['exp_avg'] = torch.clone(p.grad).detach().mul_((1 - beta1) if bias_correction else 1)

                    if adam:
                        var = torch.mean(torch.mul(p.grad, p.grad))
                        state['exp_avg_sq'] = torch.ones_like(p,
                            memory_format=torch.preserve_format).mul_(var * (1 if bias_correction else 1))
                        group['vars'].append(var)
                    else:
                        state['exp_avg_sq'] = torch.ones_like(p, memory_format=torch.preserve_format)
                        group['vars'].append(1)

                exp_avgs.append(state['exp_avg'])
                exp_avg_sqs.append(state['exp_avg_sq'])


    @_use_grad_for_differentiable
    def step(self, closure=None):
        """Performs a single optimization step.

        Args:
            closure (Callable, optional): A closure that reevaluates the model
                and returns the loss.
        """

        if closure is None:
            raise Exception('A "closure" function, performing estimation of the loss function, is required to use Acme')

        for index, group in enumerate(self.param_groups):
            group['index'] = index

        params_with_grad = {}
        prev_params = {}
        grads = {}
        prev_grads = {}
        exp_avgs = {}
        exp_avg_sqs = {}
        denoms = {}

        with torch.enable_grad():
            self.zero_grad()
            lossAnt = closure()

        for group in self.param_groups:
            index = group['index']

            params_with_grad[index] = []
            prev_params[index] = []
            grads[index] = []
            prev_grads[index] = []
            exp_avgs[index] = []
            exp_avg_sqs[index] = []

            beta1, beta2 = group['betas']
            lr = group['lr']
            acme = group['acme']
            adam = group['adam']
            asam = group['asam']
            nesterov = group['nesterov']
            bias_correction = group['bias_correction']
            step = group['step']

            self._init_group(
                group,
                params_with_grad[index],
                prev_params[index],
                grads[index],
                prev_grads[index],
                exp_avgs[index],
                exp_avg_sqs[index],
            )

            prev_params[index] = []
            prev_grads[index] = []
            grads[index] = []
            for p in group['params']:
                if p.grad is not None:
                    prev_params[index].append(torch.clone(p).detach())
                    prev_grads[index].append(torch.clone(p.grad).detach())
                    grads[index].append(p.grad)


            if adam and asam:
                if bias_correction:
                    bias_correction1 = 1 - beta1 ** step
                    bias_correction2 = 1 - beta2 ** step
                else:
                    bias_correction1 = bias_correction2 = 1

                denoms[index] = [torch.clone(tensor) for tensor in exp_avg_sqs[index]]
                torch._foreach_div_(denoms[index], bias_correction2)
                torch._foreach_sqrt_(denoms[index])
                torch._foreach_add_(denoms[index], group['epsilon'])
                torch._foreach_mul_(denoms[index], bias_correction1)

                torch._foreach_div_([p.grad for p in group['params'] if p.grad is not None], denoms[index])

            if asam:
                self.ascent_step()

                with torch.enable_grad():
                    self.zero_grad()
                    lossAsam = closure()

                asam_grads = {}
                for group in self.param_groups:
                    index = group['index']

                    asam_grads[index] = []

                    for p in group['params']:
                        if p.grad is not None:
                            asam_grads[index].append(p.grad)

                    with torch.no_grad():
                        for param, prev in zip(params_with_grad[index], prev_params[index]):
                            param[...] = prev[...]

            if True:
                acme_act_avgs(
                    grads[index] if not asam else asam_grads[index],
                    exp_avgs[index],
                    exp_avg_sqs[index],
                    beta1=beta1,
                    beta2=beta2,
                    epsilon=group['epsilon'],
                    step=step,
                    acme=acme,
                    adam=adam,
                    bias_correction=bias_correction,
                    limita=limita,
                )

            denoms[index] = acme_look(
                params_with_grad[index],
                grads[index] if not asam else asam_grads[index],
                exp_avgs[index],
                exp_avg_sqs[index],
                beta1=beta1,
                beta2=beta2,
                lr=lr,
                weight_decay=group['weight_decay'] if not acme else 0,
                epsilon=group['epsilon'],
                step=step,
                acme=acme,
                adam=adam,
                nesterov=nesterov,
                bias_correction=bias_correction,
            )

        if acme:
            with torch.enable_grad():
                self.zero_grad()
                lossAct = closure()

            for group in self.param_groups:
                index = group['index']

                for group in self.param_groups:
                    if group['index'] == index: break

                params_with_grad[index] = []
                grads[index] = []
                for p in group['params']:
                    if p.grad is not None:
                        params_with_grad[index].append(p)
                        grads[index].append(p.grad)

                lr, fact_act_avgs, step, ret = acme_leap(
                    params_with_grad[index],
                    prev_params[index],
                    grads[index],
                    prev_grads[index],
                    exp_avgs[index],
                    exp_avg_sqs[index],
                    beta1=beta1,
                    beta2=beta2,
                    lr=lr,
                    denoms=denoms[index],
                    weight_decay=group['weight_decay'],
                    epsilon=group['epsilon'],
                    step=step,
                    vars_=group['vars'],
                    bias_correction=bias_correction,
                    mellora=lossAct <= lossAnt,
                    l0=lossAnt,
                    l1=lossAct,
                    nesterov=nesterov,
                    asam=asam,
                    scheduled_lr=self.scheduled_lr,
                )

                scheculer_min = 1 / 100
                scheculer_max = 1
                self.scheduled_lr = scheculer_min + (scheculer_max - scheculer_min) * (1 + np.cos(np.pi * step / self.schedule_total)) / 2

                if lossAct < lossAnt and fact_act_avgs > 0:
                    acme_act_avgs(
                        grads[index],
                        exp_avgs[index],
                        exp_avg_sqs[index],
                        beta1=beta1,
                        beta2=beta2,
                        epsilon=group['epsilon'],
                        step=step,
                        adam=adam,
                        acme=acme,
                        bias_correction=bias_correction,
                        limita=limita,
                    )

                group['LR'] = lr
                group['lr'] = lr
                group['step'] = step + 1

                group['losses'] = f'{lossAnt:.4f}, {lossAct:.4f}'
                group['mellora'] = lossAct <= lossAnt
                group['ret'] = ret
                if ret[-1] is not None: group[ret[-1]] += 1
        else:
            group['LR'] = lr
            group['lr'] = lr
            group['step'] = step + 1

            group['losses'] = f'{lossAnt:.4f}'
            group['mellora'] = None
            group['ret'] = ''

        return lossAnt


def acme_act_avgs(
        grads: List[Tensor],
        exp_avgs: List[Tensor],
        exp_avg_sqs: List[Tensor],
        *,
        beta1: float,
        beta2: float,
        epsilon: float,
        step: float,
        acme: bool,
        adam: bool,
        bias_correction: bool,
        limita: bool,
    ):

    if adam and exp_avg_sqs is not None:
        if bias_correction:
            bias_correction1 = 1 - beta1 ** step
            bias_correction2 = 1 - beta2 ** step
        else:
            bias_correction1 = bias_correction2 = 1

        denoms = [torch.clone(tensor) for tensor in exp_avg_sqs]
        torch._foreach_div_(denoms, bias_correction2)
        torch._foreach_sqrt_(denoms)
        torch._foreach_add_(denoms, epsilon)
        torch._foreach_mul_(denoms, bias_correction1)
    else:
        denoms = [torch.tensor(1., device=_.device) for _ in exp_avgs]

    pg = pp = gg = 0
    for pgs, pps, ggs in zip(
        torch._foreach_div(torch._foreach_mul(exp_avgs, grads), denoms),
        torch._foreach_div(torch._foreach_mul(exp_avgs, exp_avgs), denoms),
        torch._foreach_div(torch._foreach_mul(grads, grads), denoms),
    ):
        pg += torch.sum(pgs, dtype=torch.float64)
        pp += torch.sum(pps, dtype=torch.float64)
        gg += torch.sum(ggs, dtype=torch.float64)


    if not torch.all(torch.isfinite(torch.tensor([pg, pp, gg]))):
        print('Vaya vaya la cigala')
        print()
        print('#' * 12, "s'acabó", '#' * 12)
        sys.exit()
        return False

    F = limita
    fact_act_avgs = 1
    raiz = (F*gg*pp - beta1**2*gg*pp + beta1**2*pg**2) ** 0.5
    k = (beta1*pg - raiz)/(gg*(beta1 - 1))
    if k < 1 and limita:
        fact_act_avgs = k

    grad_reds = torch._foreach_mul(grads, fact_act_avgs)

    torch._foreach_mul_(exp_avgs, beta1)
    torch._foreach_add_(exp_avgs, grad_reds, alpha=1 - beta1)

    if adam and exp_avg_sqs is not None:
        torch._foreach_mul_(exp_avg_sqs, beta2)
        torch._foreach_addcmul_(exp_avg_sqs, grad_reds, grad_reds, 1 - beta2)


def acme_look(
        params: List[Tensor],
        grads: List[Tensor],
        exp_avgs: List[Tensor],
        exp_avg_sqs: List[Tensor],
        *,
        beta1: float,
        beta2: float,
        lr: float,
        weight_decay: float,
        epsilon: float,
        step: float,
        acme: bool,
        adam: bool,
        nesterov: bool,
        bias_correction: bool,
    ):

    if len(params) == 0 or step == 0:
        return

    if bias_correction:
        bias_correction1 = 1 - beta1 ** step
        bias_correction2 = 1 - beta2 ** step
    else:
        bias_correction1 = bias_correction2 = 1

    if adam:
        denoms = [torch.clone(tensor) for tensor in exp_avg_sqs]
        torch._foreach_div_(denoms, bias_correction2)
        torch._foreach_sqrt_(denoms)
        torch._foreach_add_(denoms, epsilon)
        torch._foreach_mul_(denoms, bias_correction1)
    else:
        deno = 1. if not bias_correction else bias_correction1
        denoms = [torch.tensor(deno, device=params[0].device) for _ in params]

    torch._foreach_addcdiv_(params, exp_avgs, denoms, -lr * (beta1 if nesterov and not acme else 1))

    if nesterov and not acme:
        torch._foreach_addcdiv_(params, grads, denoms, -lr * (1 - beta1))

    if weight_decay != 0 and not acme:
        torch._foreach_mul_(params, 1 - lr * weight_decay)

    return denoms


def acme_leap(
        params: List[Tensor],
        prev_params: List[Tensor],
        grads: List[Tensor],
        prev_grads: List[Tensor],
        exp_avgs: List[Tensor],
        exp_avg_sqs: List[Tensor],
        *,
        beta1: float,
        beta2: float,
        lr: float,
        denoms: List[Tensor],
        weight_decay: float,
        epsilon: float,
        step: float,
        vars_: List[float],
        bias_correction: bool,
        mellora: bool,
        l0: float,
        l1: float,
        nesterov: bool,
        asam: bool,
        scheduled_lr: float,
    ):

    lr_ = lr

    if len(params) == 0:
        return float(lr), False, 0, 0, (None,)

    if denoms is None:
        return float(lr), 1, step + 1, (None,)

    pa = pg = aa = pp = gg = 0
    for pas, pgs, aas, pps, ggs in zip(
        torch._foreach_div(torch._foreach_mul(exp_avgs, prev_grads), denoms),
        torch._foreach_div(torch._foreach_mul(exp_avgs, grads), denoms),
        torch._foreach_div(torch._foreach_mul(prev_grads, prev_grads), denoms),
        torch._foreach_div(torch._foreach_mul(exp_avgs, exp_avgs), denoms),
        torch._foreach_div(torch._foreach_mul(grads, grads), denoms),
    ):
        pa += torch.sum(pas, dtype=torch.float64)
        pg += torch.sum(pgs, dtype=torch.float64)
        aa += torch.sum(aas, dtype=torch.float64)
        pp += torch.sum(pps, dtype=torch.float64)
        gg += torch.sum(ggs, dtype=torch.float64)

    g0 = pa
    g1 = pg

    lr_, lr_opt, cod, a3, b3, c3, d3 = lrCubic(l0, l1, g0, g1, lr, beta1, exp_avgs)

    str_pp = f'{l0=:_.6f}; {l1=:_.6f}; {g0=:_.2f}; {g1=:_.2f}; {pp=:_.2f}; {lr=:_.4e}; {lr_=:_.4e}; {lr_opt=:_.4e}; {cod=:d}'

    if not torch.all(torch.isfinite(torch.tensor([l0, l1, g0, g1, lr, pp, gg]))):
        print(f"RARO - NANO: {str_pp}\t{mellora}")

        torch._foreach_zero_(params)
        torch._foreach_add_(params, prev_params)

        torch._foreach_mul_(exp_avgs, 1 - beta1)

        lr *= beta1

        return float(lr), 0, 0, (str_pp, float(pa) > 0, float(pg) > 0, -1)

    gamma = 2 * (1 - beta1) / (1 + beta1) * (1 + 2 * beta1) / (2 - beta1) * scheduled_lr

    lr_act = (lr_opt if l1 > l0 else lr) * gamma
    for param, prev_param, exp_avg, denom, grad in zip(params, prev_params, exp_avgs, denoms, prev_grads):
        param.copy_(prev_param)
        param.addcdiv_(exp_avg, denom, value=-lr_act * (beta1 if nesterov else 1))

    if nesterov:
        torch._foreach_addcdiv_(params, prev_grads if l1 > l0 else grads, denoms, -lr_act * (1 - beta1))

    if weight_decay != 0:
        torch._foreach_mul_(params, 1 - lr_act * weight_decay)

    lr = beta1 * lr + (1 - beta1) * lr_

    return float(lr), 1, step, (str_pp, float(pa) > 0, float(pg) > 0, cod)

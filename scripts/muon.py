import torch
import torch.nn.functional as F

# Newton-Schulz orthogonalisation
def zeropower_via_newtonschulz5(G: torch.Tensor, steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
    assert G.ndim >= 2
    original_shape = G.shape
    if G.ndim > 2:
        G = G.reshape(G.shape[0], -1)

    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.to(torch.bfloat16) / (G.norm() + eps)

    transposed = G.shape[0] > G.shape[1]
    if transposed:
        X = X.T

    for _ in range(steps):
        A = X @ X.T
        X = a * X + b * (A @ X) + c * (A @ A @ X)

    if transposed:
        X = X.T

    return X.to(G.dtype).reshape(original_shape)


# Muon optimizer class
class Muon(torch.optim.Optimizer):
    def __init__(
        self,
        muon_params,
        lr: float = 0.02,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
        adamw_params=None,
        adamw_lr: float = 3e-4,
        adamw_betas=(0.95, 0.95),
        adamw_eps: float = 1e-8,
        adamw_wd: float = 0.0,
    ):
        defaults = dict(
            lr=lr,
            momentum=momentum,
            nesterov=nesterov,
            ns_steps=ns_steps,
            adamw_lr=adamw_lr,
            adamw_betas=adamw_betas,
            adamw_eps=adamw_eps,
            adamw_wd=adamw_wd,
        )
        muon_list = list(muon_params)
        adamw_list = list(adamw_params) if adamw_params is not None else []

        for p in muon_list:
            if p.ndim < 2:
                raise ValueError()

        param_groups = [
            {"params": muon_list,  "use_muon": True},
            {"params": adamw_list, "use_muon": False},
        ]
        super().__init__(param_groups, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if group["use_muon"]:
                self._muon_step(group)
            else:
                self._adamw_step(group)

        return loss

    def _muon_step(self, group):
        lr = group["lr"]
        momentum = group["momentum"]
        nesterov = group["nesterov"]
        ns_steps = group["ns_steps"]

        for p in group["params"]:
            if p.grad is None:
                continue

            g = p.grad
            state = self.state[p]

            if "momentum_buffer" not in state:
                state["momentum_buffer"] = torch.zeros_like(g)

            buf = state["momentum_buffer"]
            buf.mul_(momentum).add_(g)

            update = buf.lerp(g, 1 - momentum) if nesterov else buf

            g_orth = zeropower_via_newtonschulz5(update, steps=ns_steps)

            scale = max(1, update.size(0) / update.size(1)) ** 0.5
            p.add_(g_orth, alpha=-lr * scale)

    def _adamw_step(self, group):
        lr = group["adamw_lr"]
        beta1, beta2 = group["adamw_betas"]
        eps = group["adamw_eps"]
        wd = group["adamw_wd"]

        for p in group["params"]:
            if p.grad is None:
                continue

            g = p.grad
            state = self.state[p]

            if "step" not in state:
                state["step"] = 0
                state["exp_avg"]    = torch.zeros_like(p)
                state["exp_avg_sq"] = torch.zeros_like(p)

            state["step"] += 1
            t = state["step"]
            m = state["exp_avg"]
            v = state["exp_avg_sq"]

            m.mul_(beta1).add_(g, alpha=1 - beta1)
            v.mul_(beta2).addcmul_(g, g, value=1 - beta2)

            bc1 = 1 - beta1 ** t
            bc2 = 1 - beta2 ** t
            step_size = lr * (bc2 ** 0.5) / bc1

            p.mul_(1 - lr * wd)
            p.addcdiv_(m, v.sqrt().add_(eps), value=-step_size)

        return None

import torch
import math
import matplotlib.pyplot as plt

# 自定义带学习率衰减的 SGD
class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr}
        super().__init__(params, defaults)

    def step(self, closure=None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                t = state.get("t", 0)
                grad = p.grad.data
                p.data -= lr / math.sqrt(t + 1) * grad
                state["t"] = t + 1
        return loss

# 对比不同学习率的 loss 曲线
learning_rates = [1e1, 1e2, 1e3]
loss_records = {}

for lr in learning_rates:
    torch.manual_seed(0)
    weights = torch.nn.Parameter(5 * torch.randn((10, 10)))
    opt = SGD([weights], lr=lr)
    losses = []
    for t in range(10):
        opt.zero_grad()
        loss = (weights**2).mean()
        losses.append(loss.item())
        loss.backward()
        opt.step()
    loss_records[lr] = losses

# 画图
plt.figure(figsize=(10, 6))
for lr, losses in loss_records.items():
    plt.plot(losses, label=f"lr={lr}")
plt.xlabel("Step")
plt.ylabel("Loss")
plt.title("Loss over Training Steps for Different Learning Rates")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

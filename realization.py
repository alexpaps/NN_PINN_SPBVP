import torch
import torch.nn as nn
import matplotlib.pyplot as plt

torch.manual_seed(0)
torch.set_default_dtype(torch.float64)

# ============================================================
# Problem
# ============================================================

def f(x):
    return torch.ones_like(x)

def b(x):
    return torch.ones_like(x)

def u0(x):
    return f(x) / b(x)

# Exact solution
def u_exact(x, eps):
    if not torch.is_tensor(eps):
        eps = torch.tensor(eps, dtype=x.dtype, device=x.device)

    A = 1 + torch.exp(-1.0 / eps)
    return 1 - torch.exp(-x / eps)/A - torch.exp(-(1 - x)/eps)/A

# ============================================================
# Asymptotic components
# ============================================================

def phi(x):
    return x  # valid only because b(x)=1

def EL(x, eps):
    return torch.exp(-phi(x)/eps)

def ER(x, eps):
    return torch.exp(-(1 - phi(x))/eps)

def BL(x, eps):
    q = torch.exp(-1.0/eps)
    return (EL(x, eps) - q * ER(x, eps)) / (1 - q**2)

def BR(x, eps):
    q = torch.exp(-1.0/eps)
    return (ER(x, eps) - q * EL(x, eps)) / (1 - q**2)

# ============================================================
# Network (ONLY remainder)
# ============================================================

class SmallNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 32),
            nn.Tanh(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.net(x)

class UnifiedPINN(nn.Module):
    def __init__(self):
        super().__init__()
        self.rem_net = SmallNN()

    def forward(self, x, eps):

        eps = torch.tensor(eps, dtype=x.dtype, device=x.device)

        # asymptotic core (enforces BC strongly)
        u_asym = (
            u0(x)
            - u0(torch.zeros_like(x)) * BL(x, eps)
            - u0(torch.ones_like(x))  * BR(x, eps)
        )

        # O(eps) remainder
        rem = eps * x * (1 - x) * self.rem_net(x)

        return u_asym + rem

# ============================================================
# Energy functional
# ============================================================

def energy(model, x, eps):
    x.requires_grad_(True)

    u = model(x, eps)

    du = torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0]

    integrand = 0.5*(eps**2 * du**2 + u**2) - f(x)*u
    return torch.mean(integrand)

# ============================================================
# Training (epsilon continuation)
# ============================================================

def train(model, eps_list, Nx, lr):

    for eps in eps_list:
        print(f"\nTraining eps = {eps}")

        optimizer = torch.optim.LBFGS(model.parameters(), lr=lr, max_iter=500)

        def closure():
            optimizer.zero_grad()

            x = torch.rand(Nx,1)

            loss = energy(model, x, eps)

            loss.backward()
            return loss

        optimizer.step(closure)

    return model

# ============================================================
# RUN
# ============================================================

eps_list = [1, 0.3, 0.1, 0.03, 0.01, 0.003, 0.001]
Nx = 400
lr = 1e-2
eps = 1

model = UnifiedPINN()
model = train(model, eps_list, Nx, lr)

# ============================================================
# Evaluation
# ============================================================

x = torch.linspace(0,1,800).view(-1,1)
x.requires_grad = True

u_pred = model(x, eps)
u_ex = u_exact(x, torch.tensor(eps, dtype=x.dtype))

# enforce BC (should already hold, but numerics...)



du_dx = torch.autograd.grad(u_pred, x, grad_outputs=torch.ones_like(u_pred), create_graph=True)[0]
d2u_dx2 = torch.autograd.grad(du_dx, x, grad_outputs=torch.ones_like(du_dx), create_graph=True)[0]

residual_eps = -eps**2 * d2u_dx2 + b(x) * u_pred - f(x)

error_eps = torch.abs(u_pred - u_ex)
L2_eps = torch.sqrt(torch.mean(error_eps**2))

print(f"\nL2 Error: {L2_eps.item():.2e}")

# ============================================================
# Plots
# ============================================================

# -----------------------------
# Plot solutions
# -----------------------------
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(x.detach(), u_ex.detach(), '--', label=f"Exact ε={eps}", linewidth=2)
plt.plot(x.detach(), u_pred.detach(), label=f"PINN ε={eps}", linewidth=2)
plt.xlabel("x")
plt.ylabel("u(x)")
plt.legend()
plt.grid()
plt.title("Solution Comparison")

plt.subplot(1, 2, 2)
plt.plot(x.detach(), error_eps.detach(), label="Pointwise Error", linewidth=2)
plt.xlabel("x")
plt.ylabel("Error")
plt.legend()
plt.grid()
plt.title("Pointwise Error")

plt.suptitle(
    f"-ε²u''(x) + u(x) = 1, u(0)=u(1)=0 | ε={eps} |  Samples: {Nx} | LR: {lr}",
    fontsize=11,
    fontweight='bold'
)

plt.tight_layout()
plt.subplots_adjust(top=0.88)
plt.show()
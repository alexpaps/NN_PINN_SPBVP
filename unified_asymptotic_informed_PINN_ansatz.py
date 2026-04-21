import torch
import torch.nn as nn
import matplotlib.pyplot as plt

torch.manual_seed(0)
torch.set_default_dtype(torch.float64)

# ============================================================
# Problem
# ============================================================


# def f(x):
#     return torch.exp(-x)

# def c(x):
#     return torch.ones_like(x)

# def u_exact(x, eps):
#     outer = torch.exp(-x)/(1 - eps**2)
#     A = -outer[0]
#     B = -outer[-1]
#     return outer + A*torch.exp(-x/eps) + B*torch.exp(-(1-x)/eps)




def f(x):
    return torch.ones_like(x)

def c(x):
    return torch.ones_like(x)

def u_exact(x, eps):
    if not torch.is_tensor(eps):
        eps = torch.tensor(eps, dtype=x.dtype, device=x.device)

    A = 1 + torch.exp(-1.0 / eps)
    return 1 - torch.exp(-x / eps)/A - torch.exp(-(1 - x)/eps)/A

def u0(x):
    return f(x)/c(x)  #torch.ones_like(x)  

def phi(x):
    return x  # valid only because c(x)=1

# def phi(x, Nx=5000):
#     """
#     Compute phi(x) = ∫_0^x sqrt(b(s)) ds numerically
#     Supports x=0 exactly.
#     """
#     x_flat = x.detach().flatten().cpu().numpy()
#     s = np.linspace(0, 1, Nx)
#     b_vals = b(torch.tensor(s, dtype=torch.float64)).numpy()
#     sqrt_b = np.sqrt(b_vals)
#     # cumulative integral using trapezoid rule
#     cum_int = np.zeros_like(s)
#     cum_int[1:] = np.cumsum((sqrt_b[:-1] + sqrt_b[1:])/2 * (s[1]-s[0]))
#     # interpolate
#     phi_vals = np.interp(x_flat, s, cum_int)
#     return torch.tensor(phi_vals, dtype=torch.float64, device=x.device).view(-1,1)



# ============================================================
# Asymptotics
# ============================================================

def EL(x, eps):
    return torch.exp(-phi(x)/eps)

def ER(x, eps):
    return torch.exp(-(1 - phi(x))/eps)

def BL(x, eps):
    q = torch.exp(-1.0/eps)
    return (EL(x, eps) - q * ER(x, eps)) / (1 - q**2 + 1e-12)

def BR(x, eps):
    q = torch.exp(-1.0/eps)
    return (ER(x, eps) - q * EL(x, eps)) / (1 - q**2 + 1e-12)

# ============================================================
# Networks
# ============================================================

class FCNN(nn.Module):
    def __init__(self, hidden=64, layers=3):
        super().__init__()
        net = [nn.Linear(1, hidden), nn.Tanh()]
        for _ in range(layers-1):
            net += [nn.Linear(hidden, hidden), nn.Tanh()]
        net += [nn.Linear(hidden,1)]
        self.net = nn.Sequential(*net)

    def forward(self, x):
        return self.net(x)

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

# ============================================================
# Model
# ============================================================

class UnifiedPINN(nn.Module):
    def __init__(self, n_hidden=20, n_layers=1):
        super().__init__()

        self.small_net = SmallNN()
        self.bulk = FCNN(2*n_hidden,n_layers)
        self.left = FCNN(n_hidden,n_layers)
        self.right = FCNN(n_hidden,n_layers)

        # blending parameter
        self.alpha = nn.Parameter(torch.tensor(0.0))

    def forward(self, x, eps):

        eps = torch.tensor(eps, dtype=x.dtype, device=x.device)

        # --- asymptotic part ---
        base = u0(x)

        asym = (
            base
            - u0(torch.zeros_like(x))*BL(x,eps)
            - u0(torch.ones_like(x))*BR(x,eps)
        )

        # rem = eps * x*(1-x)*self.small_net(x)
        rem = self.small_net(x)
        u_asym = asym + rem

        # --- multiscale correction ---
        xiL = x/eps
        xiR = (1-x)/eps

        wL = torch.exp(-x/eps)
        wR = torch.exp(-(1-x)/eps)

        u_ms = x*(1-x)*(
            self.bulk(x)
            + wL*self.left(xiL)
            + wR*self.right(xiR)
        )

        alpha = torch.sigmoid(self.alpha)

        return u_asym + alpha*u_ms

# ============================================================
# Training
# ============================================================

def train(model, eps, Nx, epochs, lr, alpha=0.0, beta=0.0):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):

        x = torch.linspace(0,1,Nx).view(-1,1).requires_grad_(True)

        u = model(x, eps)

        du_dx = torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0]
        d2u_dx2 = torch.autograd.grad(du_dx, x, torch.ones_like(du_dx), create_graph=True)[0]

        residual = -eps**2 * d2u_dx2 + c(x) * u - f(x)
        pde_loss = torch.mean(residual**2)

        x_bc = torch.tensor([[0.0], [1.0]], requires_grad=True)
        u_bc = model(x_bc, eps)

        bc_loss = (u_bc[0] - alpha)**2 + (u_bc[1] - beta)**2

        loss = pde_loss + bc_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 1000 == 0:
            print(f"ε={eps}, Epoch {epoch}, Loss={loss.item():.2e}")

    return model

# ============================================================
# RUN
# ============================================================

eps = 1e-2
Nx = 400
epochs = 2000
n_hidden=30
n_layers=2
lr = 1e-2

model = UnifiedPINN(n_hidden=30, n_layers=2)
model = train(model, eps=eps, Nx=Nx, epochs=epochs, lr=lr)

# ============================================================
# Evaluation
# ============================================================

x = torch.linspace(0,1,800).view(-1,1).requires_grad_(True)

u_pred = model(x, eps)
u_ex = u_exact(x, torch.tensor(eps, dtype=x.dtype))

du_dx = torch.autograd.grad(u_pred, x, torch.ones_like(u_pred), create_graph=True)[0]
d2u_dx2 = torch.autograd.grad(du_dx, x, torch.ones_like(du_dx), create_graph=True)[0]

residual_eps = -eps**2 * d2u_dx2 + c(x) * u_pred - f(x)

error_eps = torch.abs(u_pred - u_ex)
L2_eps = torch.sqrt(torch.mean(error_eps**2))

print(f"\nL2 Error: {L2_eps.item():.2e}")

# ============================================================
# Plots
# ============================================================

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(x.detach(), u_ex.detach(), '--', label="Exact", linewidth=2)
plt.plot(x.detach(), u_pred.detach(), label="PINN", linewidth=2)
plt.xlabel("x")
plt.ylabel("u(x)")
plt.legend()
plt.grid()
plt.title("Solution Comparison")

plt.subplot(1, 2, 2)
plt.plot(x.detach(), error_eps.detach(), label="Error", linewidth=2)
plt.xlabel("x")
plt.ylabel("Error")
plt.legend()
plt.grid()
plt.title("Pointwise Error")


plt.suptitle(
    f"-ε²u''(x) + u(x) = 1, u(0)=u(1)=0 | ε={eps} | Epochs: {epochs} | Nodes: {n_hidden} | Layers: {n_layers} | Samples: {Nx} | LR: {lr}",
    fontsize=11,
    fontweight='bold'
)

plt.tight_layout()
plt.subplots_adjust(top=0.88)
plt.show()
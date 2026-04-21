import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# -----------------------------
# Problem definition
# -----------------------------
def f(x):
    return torch.ones_like(x)

def c(x):
    return torch.ones_like(x)

# Exact solution
def u_exact(x, eps):
    if not torch.is_tensor(eps):
        eps = torch.tensor(eps, dtype=x.dtype, device=x.device)

    A = 1 + torch.exp(-1.0 / eps)
    B = 1 - torch.exp(-x / eps) / A - torch.exp(-(1 - x) / eps) / A
    return B

# -----------------------------
# Neural Network
# -----------------------------
class PINN(nn.Module):
    def __init__(self, n_hidden=20, n_layers=1):
        super().__init__()
        layers = [nn.Linear(1, n_hidden), nn.Tanh()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(n_hidden, n_hidden), nn.Tanh()]
        layers += [nn.Linear(n_hidden, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

# -----------------------------
# Training function (SOFT BCs)
# -----------------------------
def train(model, eps, Nx, epochs, lr, alpha=0.0, beta=0.0):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        # Interior points
        # x = torch.rand(Nx, 1, requires_grad=True)
        x = torch.linspace(0,1,Nx).view(-1,1)
        x.requires_grad = True
        u = model(x)

        du_dx = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        d2u_dx2 = torch.autograd.grad(du_dx, x, grad_outputs=torch.ones_like(du_dx), create_graph=True)[0]

        residual = -eps**2 * d2u_dx2 + c(x) * u - f(x)
        pde_loss = torch.mean(residual**2)

        # Boundary points
        x_bc = torch.tensor([[0.0], [1.0]], requires_grad=True)
        u_bc = model(x_bc)

        bc_loss = (u_bc[0] - alpha)**2 + (u_bc[1] - beta)**2

        # Total loss
        loss = pde_loss + bc_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 1000 == 0:
            print(f"ε={eps}, Epoch {epoch}, Loss={loss.item():.2e}, PDE={pde_loss.item():.2e}, BC={bc_loss.item():.2e}")

    return model

# -----------------------------
# Train for ε = 0.01
# -----------------------------
torch.manual_seed(0)

eps = 1e-2
n_hidden = 40
n_layers = 1
Nx = 400
epochs = 5000
lr = 1e-2

model_eps = PINN(n_hidden=n_hidden, n_layers=n_layers)
model_eps = train(model_eps, eps=eps, Nx=Nx, epochs=epochs, lr=lr)

# -----------------------------
# Evaluation and comparison
# -----------------------------
x_plot = torch.linspace(0, 1, 400).view(-1, 1)
x_plot.requires_grad = True

u_ex = u_exact(x_plot, eps)

# PINN solution (no hard constraint anymore)
u_eps = model_eps(x_plot)

du_dx = torch.autograd.grad(u_eps, x_plot, grad_outputs=torch.ones_like(u_eps), create_graph=True)[0]
d2u_dx2 = torch.autograd.grad(du_dx, x_plot, grad_outputs=torch.ones_like(du_dx), create_graph=True)[0]

residual_eps = -eps**2 * d2u_dx2 + c(x_plot) * u_eps - f(x_plot)

error_eps = torch.abs(u_eps - u_ex)
L2_eps = torch.sqrt(torch.mean(error_eps**2))

print(f"\nL2 Error: {L2_eps.item():.2e}")

# -----------------------------
# Plot solutions
# -----------------------------
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(x_plot.detach(), u_ex.detach(), '--', label=f"Exact ε={eps}", linewidth=2)
plt.plot(x_plot.detach(), u_eps.detach(), label=f"PINN ε={eps}", linewidth=2)
plt.xlabel("x")
plt.ylabel("u(x)")
plt.legend()
plt.grid()
plt.title("Solution Comparison")

plt.subplot(1, 2, 2)
plt.plot(x_plot.detach(), error_eps.detach(), label="Pointwise Error", linewidth=2)
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
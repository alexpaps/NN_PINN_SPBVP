import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# Neural Network
class PINN(nn.Module):
    def __init__(self, n_hidden=32, n_layers=2):
        super().__init__()
        layers = [nn.Linear(1, n_hidden), nn.Tanh()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(n_hidden, n_hidden), nn.Tanh()]
        layers += [nn.Linear(n_hidden, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

# Training function
def train(model, Nx, epochs, lr):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        x = torch.rand(Nx, 1, requires_grad=True)
        y = model(x)
        
        # dy/dx via autograd
        dy_dx = torch.autograd.grad(y, x, grad_outputs=torch.ones_like(y), create_graph=True)[0]
        
        # ODE residual: dy/dx - x = 0
        residual = dy_dx - x
        
        # Initial condition: y(0) = 0
        y0 = model(torch.tensor([[0.0]]))
        ic_loss = y0**2
        
        loss = torch.mean(residual**2) + 10 * ic_loss
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if epoch % 500 == 0:
            print(f"Epoch {epoch}, Loss={loss.item():.2e}")

    return model

# Train
torch.manual_seed(0)
n_hidden = 20
n_layers = 1
Nx = 300
epochs = 5000
lr = 0.01

model = PINN(n_hidden=n_hidden, n_layers=n_layers)
model = train(model, Nx=Nx, epochs=epochs, lr=lr)

# Evaluation
x_plot = torch.linspace(0, 1, 400).view(-1, 1)
x_plot.requires_grad = True

y_pred = model(x_plot)
y_exact = x_plot**2 / 2  # Exact solution: x²/2

# Compute residual
dy_dx = torch.autograd.grad(y_pred, x_plot, grad_outputs=torch.ones_like(y_pred), create_graph=True)[0]
residual = dy_dx - x_plot

error = torch.abs(y_pred - y_exact)
L2_error = torch.sqrt(torch.mean(error**2))
print(f"\nL2 Error: {L2_error.item():.2e}")

# Plots
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(x_plot.detach(), y_exact.detach(), '--', label="Exact: x²/2", linewidth=2)
plt.plot(x_plot.detach(), y_pred.detach(), label="PINN", linewidth=2)
plt.xlabel("x")
plt.ylabel("y(x)")
plt.legend()
plt.grid()
plt.title("Solution Comparison")

plt.subplot(1, 2, 2)
plt.plot(x_plot.detach(), error.detach(), label="Pointwise Error", linewidth=2)
plt.xlabel("x")
plt.ylabel("Error")
plt.legend()
plt.grid()
plt.title("Pointwise Error")

# Add hyperparameters title
plt.suptitle(f"y'(x) = x, y(0) = 0 | Epochs: {epochs} | Hidden Nodes: {n_hidden} | Layers: {n_layers} | Samples: {Nx} | LR: {lr}", 
             fontsize=11, fontweight='bold')

plt.tight_layout()
plt.subplots_adjust(top=0.88)
plt.show()
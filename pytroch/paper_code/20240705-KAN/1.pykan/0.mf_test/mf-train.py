import torch
from kan import *


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.set_default_dtype(torch.float64)
print(torch.__version__)

# create a KAN: 2D inputs, 1D output, and 5 hidden neurons. cubic spline (k=3), 5 grid intervals (grid=5).
model = KAN(width=[2, 5, 1], grid=5, k=3, seed=0)
print(model)

# create dataset f(x,y) = exp(sin(pi*x)+y^2)
f = lambda x: torch.exp(torch.sin(torch.pi*x[:,[0]]) + x[:,[1]]**2)
dataset = create_dataset(f, n_var=2)
print(dataset['train_input'].shape, dataset['train_label'].shape)

# plot KAN at initialization
model(dataset['train_input'])
plt = model.plot(beta=100)
plt.show()

# train the model
model.train(dataset, opt="LBFGS", steps=20, lamb=0.01, lamb_entropy=10.)
plt = model.plot()
plt.show()

model.prune()
plt = model.plot(mask=True)
plt.show()

model = model.prune()
model(dataset['train_input'])
plt = model.plot()
plt.show()

model.train(dataset, opt="LBFGS", steps=50)
plt = model.plot()
plt.show()

mode = "auto"  # "manual"

if mode == "manual":
    # manual mode
    model.fix_symbolic(0,0,0,'sin');
    model.fix_symbolic(0,1,0,'x^2');
    model.fix_symbolic(1,0,0,'exp');
elif mode == "auto":
    # automatic mode
    lib = ['x','x^2','x^3','x^4','exp','log','sqrt','tanh','sin','abs']
    model.auto_symbolic(lib=lib)

model.train(dataset, opt="LBFGS", steps=50);

print(model.symbolic_formula()[0][0])
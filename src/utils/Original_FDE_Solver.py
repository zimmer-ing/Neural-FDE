#This is the original code for the Fractional Differential Equation Solver as used in https://arxiv.org/abs/2403.02737v2
#Code is available at https://github.com/CeciliaCoelho/neuralFDE

from scipy import special
import torch
import sys
import math


torch.set_default_dtype(torch.float64)

def solve(a, f, y0, tspan):
    t0 = tspan[0]
    h = tspan[1] - t0
    N = len(tspan) 
    y = torch.zeros((1,1))
    device = y0.device

    for n in range(0, N):
        temp = torch.zeros(1, dtype=torch.float, requires_grad=True).to(device)

        for k in range(0, int(torch.ceil(a))):
            temp.data += y0 * (tspan[n])**k / math.factorial(k)

        y_p = predictor(f, y, a, n, h, y0, t0, tspan)
        y_new = temp + h**a / math.gamma(a+2) *(f(tspan[n], y_p) + right(f, y, a, n, h)).unsqueeze(1)
        if n == 0: 
            y = y_new
        else:
            y = torch.cat([y, y_new], dim=0)

    return y

def right(f, y, a, n, h):
    device = y.device
    temp = torch.zeros((1)).to(device) 
    for j in range(0, n+1):
        if j == n:
            temp += A(j, n, a) * f((j*h).to(device), torch.zeros((1)).to(device))
        else:
            temp += A(j, n, a) * f((j*h).to(device), y[j].to(device))

    return temp

def A(j, n, a):
    if j == 0:
        return n**(a + 1) - (n - a) * (n + 1)**a
    elif 1 <= j <= n:
        return (n - j + 2)**(a + 1) + (n - j)**(a + 1) - 2*(n - j + 1)**(a+1)
    elif j == n + 1:
        return 1

def predictor(f, y, a, n, h, y0, t0, tspan):
    device = y.device
    predict = torch.zeros((1)).to(device) 
    leftsum = 0.
    l = torch.ceil(a)
    for k in range(0, int(l)):
        leftsum += y0 * (tspan[n])**k / math.factorial(k)

    for j in range(0, n+1):
        if j == n: 
            predict += torch.mul(B(j, n, a), f(tspan[n], torch.zeros((1)).to(device)))
        else:
            predict += torch.mul(B(j, n, a), f(tspan[n], y[j].to(device)))

    return leftsum.add(torch.mul(h**a / a, predict))

def B(j, n, a):
    return ((n + 1 - j)**a - (n - j)**a)


if __name__ == "__main__":
    def fractionalDiffEq(t, x):
        return -x

    from mittag_leffler import ml as mittag_leffler
    import matplotlib.pyplot as plt
    import time
    import numpy as np


    dtypes = [torch.float32, torch.float64]
    results = {}
    steps=2000


    t = torch.linspace(0., 20., steps)[1:]
    real_values = [mittag_leffler(-i.item() ** 0.6, 0.6) for i in t]
    real_values = real_values
    y0 = torch.tensor([1.] )

    time_start = time.time()


    solver_values = solve(torch.tensor([0.6]),fractionalDiffEq,y0,t)
    time_end = time.time()

    print('Time taken by solver : ', time_end - time_start)
    plt.plot(t.squeeze().detach().numpy(), solver_values.detach().numpy(), label=f'Solver')
    plt.plot(t.detach().numpy(), real_values, label='Real values')
    plt.legend()
    plt.show()

    real_values = np.array(real_values)
    predictor_values = solver_values.detach().numpy()
    error = real_values.flatten() - predictor_values.flatten()
    print('Total error : ', np.sum(np.abs(error))/len(error))
    plt.plot(t.detach().numpy(), error, label=f'Error ')
    plt.legend()
    plt.show()




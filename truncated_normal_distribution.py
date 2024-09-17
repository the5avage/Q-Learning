import torch
from torch.distributions.normal import Normal
import math
# As described in The Truncated Normal Distribution
# John Burkardt
# Department of Scientific Computing
# Florida State University
# https://people.sc.fsu.edu/∼jburkardt/presentations/truncated normal.pdf

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

sqrt_2 = torch.sqrt(torch.tensor(2.0, device=device))
sqrt_2_pi = torch.sqrt(torch.tensor(2.0 * torch.pi, device=device))
sqrt_e = torch.sqrt(torch.tensor(torch.e, device=device))
sqrt_e_2_pi = sqrt_e * sqrt_2_pi
minus_6 = torch.tensor([-6.0], device=device)

def phi(u, s, x):
    coeff = (s * sqrt_2_pi)
    exponent = -0.5 * ((x - u) / s) ** 2
    return torch.exp(exponent) / coeff

def PHI(u, s, x):
    return 0.5 * (1.0 + torch.erf((x - u)/ s / sqrt_2))

def PHI_inverse(u, s, p):
    p[(p < torch.zeros(p.size(), device=p.device)) | (p > torch.ones(p.size(), device=p.device))] = float('nan')
    return u + s * sqrt_2 * torch.erfinv(2.0 * p - 1.0)

#def phi(u, v, x):
#    s = torch.sqrt(v)
#    return Normal(u, s).log_prob(x).exp()

#def PHI(u, v, x):
#    s = torch.sqrt(v)
#    return Normal(u, s).cdf(x)

#def PHI_inverse(u, v, p):
#    s = torch.sqrt(v)
#    return Normal(u, s).icdf(p)

# Define the function f(x)
def truncated_normal_density(u, s, a, b, x):
    result = phi(u, s, x) / (PHI(u, s, b) - PHI(u, s, a))
    a_comp = torch.full(x.size(), a.item(), device=x.device)
    b_comp = torch.full(x.size(), b.item(), device=x.device)
    result[(x < a_comp) | (x > b_comp)] = 0.0
    return result

def truncated_normal_cmd(u, s, a, b, x):
    result = (PHI(u, s, x) - PHI(u, s, a)) / (PHI(u, s, b) - PHI(u, s, a))
    result[x <= torch.full(x.size(), a.item(), device=x.device)] = 0.0
    result[x >= torch.full(x.size(), b.item(), device=x.device)] = 1.0
    return result

def truncated_normal_inverse_cmd(u, s, a, b, p):
    p[(p < torch.zeros(p.size(), device=p.device)) | (p > torch.ones(p.size(), device=p.device))] = float('nan')
    return PHI_inverse(u, s, PHI(u, s, a) + p * (PHI(u, s, b) - PHI(u, s, a)))

def truncated_normal_entropy(u, s, a, b):
    alpha = (a - u) / s
    beta = (b - u) / s
    z = PHI(u, s, beta) - PHI(u, s, alpha)
    result = torch.log(sqrt_e_2_pi * s * z) + 0.5 * (alpha * phi(u, s, alpha) - beta * phi(u, s, beta)) / z
    if result < -10.0:
        result = minus_6 #avoid -inf
    elif result >= -10.0:
        pass
    else:
        result = minus_6 #avoid nan
    return result

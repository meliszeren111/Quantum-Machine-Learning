from docplex.mp.model import Model
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit_optimization.translators import from_docplex_mp
from qiskit.primitives import Sampler
from qiskit_algorithms.utils import algorithm_globals
from qiskit_algorithms import QAOA
from qiskit_algorithms.optimizers import SPSA
# https://pypi.org/project/qiskit-optimization/

n = 4
edges = [(0, 1, 1.0), (0, 2, 1.0), (2, 3, 1.0)]  # (node_i, node_j, weight)
model = Model()
x = model.binary_var_list(n)
model.maximize(model.sum(w * x[i] * (1 - x[j]) + w * (1 - x[i]) * x[j] for i, j, w in edges))
model.add(x[0] == 1)
problem = from_docplex_mp(model)
seed = 1234
algorithm_globals.random_seed = seed
# Simultaneous Perturbation Stochastic Approximation (SPSA)
spsa = SPSA(maxiter=250)
sampler = Sampler()
# Quantum Approximate Optimization Algorithm (QAOA)
qaoa = QAOA(sampler=sampler, optimizer=spsa, reps=5)
algorithm = MinimumEigenOptimizer(qaoa)
result = algorithm.solve(problem)
print(result.prettyprint())

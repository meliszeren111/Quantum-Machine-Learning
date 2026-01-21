from matplotlib import pyplot as plt
from qiskit import QuantumCircuit
from qiskit.circuit.library import XGate, IGate
from qiskit.visualization import plot_histogram
from qiskit_aer.primitives import Sampler

# SOURCE: https://github.com/MonitSharma/Learn-Quantum-Computing-with-Qiskit/blob/main/Deutsch_Jozsa_Algorithm.ipynb

def hadamard(circuit, n):
    circuit.h(n)
    return circuit

def oracle(circuit, bit_string, n):
    for i in range(n):
        if bit_string[i] == "1":
            circuit.append(XGate(), [i])
        else:
            circuit.append(IGate(), [i])
    return circuit


def implement(f, n, num_shots):
    circuit = QuantumCircuit(n+1, n)
    bit_string = [i for item in f for i in item]
    initial_bit_string = bit_string[:-1]
    hadamard(circuit, len(initial_bit_string)-1)
    oracle(circuit, bit_string, n)
    hadamard(circuit, initial_bit_string)
    circuit.measure(range(n), range(n))
    sampler = Sampler()
    result = sampler.run(circuit, shots=num_shots).result()
    counts = result.quasi_dists[0]
    return counts, circuit

def conclude(f, n, num_shots):
    counts, circuit = implement(f, n, num_shots)
    result_number = max(counts, key=counts.get)
    binary_number = bin(result_number)[2:]
    all_zero = all(bit == "0" for bit in binary_number)
    if all_zero:
        return f"{f} FUNCTION IS CONSTANT"
    else:
        return f"{f} FUNCTION IS BALANCED"

def convert_number_to_binary(number: int, bit_length: int):
    if number >= 2**bit_length:
        raise ValueError(f"{number} cannot be represented with {bit_length} bits.")
    binary = format(number, f'0{bit_length}b')
    return binary

def visualize(f, n, num_shots):
    counts, circuit = implement(f, n, num_shots)
    circuit.draw('mpl')
    plot_histogram(counts)
    plt.show()
    return

f3 = [[1, 0], [0, 1]]
f0 = [[0, 0], [0, 0]]
num_shots = 1000
n = 2
result = conclude(f0, n, num_shots)
print(result)
visualize(f0, n, num_shots)
result = conclude(f3, n, num_shots)
print(result)
visualize(f3, n, num_shots)
# identity_matrix_4x4 =   [[1, 0, 0, 0],
#                          [0, 1, 0, 0],
#                          [0, 0, 1, 0],
#                          [0, 0, 0, 1]]
# n = 16
# result = conclude(identity_matrix_4x4, n, num_shots)
# print(result)

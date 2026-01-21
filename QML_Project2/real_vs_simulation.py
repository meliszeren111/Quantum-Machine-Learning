from collections import Counter
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2, SamplerOptions
import yaml

yaml_path = r"data.yaml"
with open(yaml_path, "r") as file:
    yaml_data = yaml.load(file, Loader=yaml.FullLoader)
ibm_quantum_api_token = yaml_data["ibm_quantum_api_token"]
service = QiskitRuntimeService(channel="ibm_quantum", token=ibm_quantum_api_token)

def create_theta_circuit(theta):
    qc = QuantumCircuit(1, 1)
    qc.ry(theta, 0)
    qc.measure(0, 0)
    return qc

def run_simulator(circuit, num_shots=1024):
    aer_simulator = AerSimulator()
    simulator = aer_simulator.run(circuit, shots=num_shots)
    result = simulator.result()
    counts = result.get_counts()
    return counts

def run_real_device(circuit, shots=1024):
    backend = service.least_busy(simulator=False, operational=True)
    transpiled_circuit = transpile(circuit, backend=backend)
    options = SamplerOptions(default_shots=shots)
    sampler = SamplerV2(mode=backend, options=options)
    job = sampler.run([transpiled_circuit])
    result = job.result()
    bit_array = result[0].data.c
    buffer = bit_array._array
    num_shots = bit_array.num_shots
    num_bits = bit_array.num_bits
    bit_data = np.unpackbits(np.frombuffer(buffer, dtype=np.uint8))
    bit_data = bit_data[:num_shots * num_bits].reshape((num_shots, num_bits))
    bit_strings = [''.join(map(str, row)) for row in bit_data]
    counts = dict(Counter(bit_strings))
    return counts

def run_both(theta):
    circuit = create_theta_circuit(theta)
    print("Running on simulator...")
    simulator_dict = run_simulator(circuit)
    print("Running on real quantum computer...")
    real_device_dict = run_real_device(circuit)
    print("Simulator Counts:", simulator_dict)
    print("Real Device counts:", real_device_dict)
    simulator_total = sum(simulator_dict.values())
    simulator_probs = [{key: value/simulator_total} for key, value in simulator_dict.items()]
    real_total = sum(real_device_dict.values())
    real_probs = [{key: value / real_total} for key, value in real_device_dict.items()]
    return simulator_probs, real_probs

def create_theta(p):
    theta = 2 * np.arccos(np.sqrt(p))
    return theta

p = 0.33
theta = create_theta(p)
simulator_probs, real_probs = run_both(theta)
print("Simulator Probabilities:", simulator_probs)
print("Real Probabilities:", real_probs)

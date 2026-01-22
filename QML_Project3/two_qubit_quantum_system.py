import numpy as np
from qiskit import transpile
from qiskit.circuit.library import RealAmplitudes
from qiskit.quantum_info import SparsePauliOp
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit_aer.primitives import EstimatorV2
from qiskit_aer.primitives import SamplerV2
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

ibm_quantum_api_token = "d80520305a8e1768536a57e3d7c3d729c762062af7065c74f26fc61153933fdaf06efa11775acdd28908682b805d2e18e5a58e9d2092247b36cfd4fea7f89f57"
provider = QiskitRuntimeService(channel='ibm_quantum', token=ibm_quantum_api_token)
aer_simulator = AerSimulator()
# 1 QUBITLI SİSTEMDE 2 COMPUTATİONAL BASİS VAR
quantum_circuit = QuantumCircuit(2)  # 2 kübitli bir kuantum devresi
gate1 = transpile(quantum_circuit, aer_simulator)
gate2 = transpile(quantum_circuit, aer_simulator)
gate3 = transpile(quantum_circuit, aer_simulator)
# 4 adet kompleks sayı
real_parts = np.random.randn(4)
imag_parts = np.random.randn(4)
complex_vector = real_parts + 1j * imag_parts
# Normalizasyon
normalization = np.linalg.norm(complex_vector)
normalized_vector = complex_vector / normalization
state_vector = Statevector(normalized_vector)
print(state_vector)

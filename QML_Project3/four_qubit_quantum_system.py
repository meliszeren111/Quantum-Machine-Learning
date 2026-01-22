from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Statevector
import numpy as np
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime import QiskitRuntimeService
import matplotlib.pyplot as plt

ibm_quantum_api_token = "d80520305a8e1768536a57e3d7c3d729c762062af7065c74f26fc61153933fdaf06efa11775acdd28908682b805d2e18e5a58e9d2092247b36cfd4fea7f89f57"
provider = QiskitRuntimeService(channel='ibm_quantum', token=ibm_quantum_api_token)
aer_simulator = AerSimulator()
# COGNITION WITH QUITRIT
quantum_circuit_cognition = QuantumCircuit(2) # 2 qubit'lik kuantum devresi
state00_cognition = np.array([1, 0, 0, 0])
state01_cognition = np.array([0, 1, 0, 0])
state10_cognition = np.array([0, 0, 1, 0])
normalization_cognition = 1 / np.sqrt(3)
superposed_state_cognition = normalization_cognition * (state00_cognition + state01_cognition + state10_cognition)
state_vector_cognition = Statevector(superposed_state_cognition)
quantum_circuit_cognition.initialize(state_vector_cognition, [0, 1])
# EMOTION WITH QUITRIT
quantum_circuit_emotion = QuantumCircuit(2) # 2 qubit'lik kuantum devresi
state00_emotion = np.array([1, 0, 0, 0])
state01_emotion = np.array([0, 1, 0, 0])
state10_emotion = np.array([0, 0, 1, 0])
normalization_emotion = 1 / np.sqrt(3)
superposed_state_emotion = normalization_emotion * (state00_emotion + state01_emotion + state10_emotion)
state_vector_emotion = Statevector(superposed_state_emotion)
quantum_circuit_emotion.initialize(state_vector_emotion, [0, 1])
# COGNITION AND EMOTION IN ENTANGLEMENT
quantum_circuit_cognition_emotion = QuantumCircuit(2)
normalization_cognition_emotion = 1 / np.sqrt(2)
superposed_state_cognition_emotion = superposed_state_cognition + superposed_state_emotion
normalized_superposed_state_cognition_emotion = superposed_state_cognition_emotion / np.linalg.norm(superposed_state_cognition_emotion)
quantum_circuit_cognition_emotion.initialize(normalized_superposed_state_cognition_emotion, [0, 1])
quantum_circuit_cognition_emotion.h(0)
quantum_circuit_cognition_emotion.cx(0, 1)
entangled_cognition_emotion_state = Statevector.from_instruction(quantum_circuit_cognition_emotion)
# WILL WITH QUITRIT
quantum_circuit_will = QuantumCircuit(2) # 2 qubit'lik kuantum devresi
state00_will = np.array([1, 0, 0, 0])
state01_will = np.array([0, 1, 0, 0])
state10_will = np.array([0, 0, 1, 0])
normalization_will = 1 / np.sqrt(3)
normalized_superposed_state_will = normalization_will * (state00_will + state01_will + state10_will)
state_vector_will = Statevector(normalized_superposed_state_will, dims=[2, 2])

# ENTANGLED EMOTION COGNITION'A HADAMARD + RZ GATE İLE FAZ FARKI VERECEĞİZ
quantum_circuit_entangled_cognition_emotion = QuantumCircuit(2)
quantum_circuit_entangled_cognition_emotion.initialize(entangled_cognition_emotion_state, [0, 1])
quantum_circuit_entangled_cognition_emotion.h(0)
quantum_circuit_entangled_cognition_emotion.rz(np.pi, 0)
quantum_circuit_entangled_cognition_emotion.h(0)
phased_entangled_cognition_emotion_state = Statevector.from_instruction(quantum_circuit_entangled_cognition_emotion)

# ENTANGLED EMOTION-COGNITION INTERFERES WITH WILL
full_system_state_vector = phased_entangled_cognition_emotion_state.tensor(state_vector_will)
full_system_quantum_circuit = QuantumCircuit(4,4)
full_system_quantum_circuit.initialize(full_system_state_vector, [0, 1, 2, 3])
print("full_system", full_system_state_vector)
transpiled_gate = transpile(full_system_quantum_circuit, aer_simulator)
transpiled_gate.measure([0, 1], [0, 1])
job = aer_simulator.run(transpiled_gate, shots=1000)
result = job.result()
counts = result.get_counts()
total = sum(counts.values())
probabilities = {state: count / total for state, count in counts.items()}
print("Counts:", counts)
print("Probabilities:", probabilities)

prob_dict = full_system_state_vector.probabilities_dict()
decision_A = {"0000", "0001", "0010"}
decision_B = {"0100", "0101", "0110"}
decision_C = set(prob_dict.keys()) - decision_A - decision_B

total_probability_of_decision_A = sum(prob_dict[decision] for decision in decision_A if decision in prob_dict)
total_probability_of_decision_B = sum(prob_dict[decision] for decision in decision_B if decision in prob_dict)
total_probability_of_decision_C = sum(prob_dict[decision] for decision in decision_C if decision in prob_dict)

print("Karar A olasılığı:", total_probability_of_decision_A)
print("Karar B olasılığı:", total_probability_of_decision_B)
print("Karar C olasılığı:", total_probability_of_decision_C)
# quantum_circuit_cognition_emotion.draw('mpl')
# plt.show()
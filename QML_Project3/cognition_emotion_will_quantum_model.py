from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Statevector
import numpy as np
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime import QiskitRuntimeService
import matplotlib.pyplot as plt

ibm_quantum_api_token = ""
provider = QiskitRuntimeService(channel='ibm_quantum', token=ibm_quantum_api_token)
aer_simulator = AerSimulator()

# COGNITION WITH QUTRIT (note: implemented using a 2-qubit subspace)
quantum_circuit_cognition = QuantumCircuit(2)  # 2-qubit quantum circuit
state00_cognition = np.array([1, 0, 0, 0])
state01_cognition = np.array([0, 1, 0, 0])
state10_cognition = np.array([0, 0, 1, 0])
normalization_cognition = 1 / np.sqrt(3)
superposed_state_cognition = normalization_cognition * (state00_cognition + state01_cognition + state10_cognition)
state_vector_cognition = Statevector(superposed_state_cognition)
quantum_circuit_cognition.initialize(state_vector_cognition, [0, 1])

# EMOTION WITH QUTRIT (note: implemented using a 2-qubit subspace)
quantum_circuit_emotion = QuantumCircuit(2)  # 2-qubit quantum circuit
state00_emotion = np.array([1, 0, 0, 0])
state01_emotion = np.array([0, 1, 0, 0])
state10_emotion = np.array([0, 0, 1, 0])
normalization_emotion = 1 / np.sqrt(3)
superposed_state_emotion = normalization_emotion * (state00_emotion + state01_emotion + state10_emotion)
state_vector_emotion = Statevector(superposed_state_emotion)
quantum_circuit_emotion.initialize(state_vector_emotion, [0, 1])

# ENTANGLEMENT OF COGNITION AND EMOTION
quantum_circuit_cognition_emotion = QuantumCircuit(2)
superposed_state_cognition_emotion = superposed_state_cognition + superposed_state_emotion
normalized_superposed_state_cognition_emotion = superposed_state_cognition_emotion / np.linalg.norm(superposed_state_cognition_emotion)
quantum_circuit_cognition_emotion.initialize(normalized_superposed_state_cognition_emotion, [0, 1])
quantum_circuit_cognition_emotion.h(0)
quantum_circuit_cognition_emotion.cx(0, 1)
entangled_cognition_emotion_state = Statevector.from_instruction(quantum_circuit_cognition_emotion)

# WILL WITH QUTRIT (note: implemented using a 2-qubit subspace)
quantum_circuit_will = QuantumCircuit(2)  # 2-qubit quantum circuit
state00_will = np.array([1, 0, 0, 0])
state01_will = np.array([0, 1, 0, 0])
state10_will = np.array([0, 0, 1, 0])
normalization_will = 1 / np.sqrt(3)
normalized_superposed_state_will = normalization_will * (state00_will + state01_will + state10_will)
state_vector_will = Statevector(normalized_superposed_state_will, dims=[2, 2])

# APPLY PHASE SHIFT TO ENTANGLED COGNITION-EMOTION STATE USING HADAMARD + RZ GATES
quantum_circuit_entangled_cognition_emotion = QuantumCircuit(2)
quantum_circuit_entangled_cognition_emotion.initialize(entangled_cognition_emotion_state, [0, 1])
quantum_circuit_entangled_cognition_emotion.h(0)
quantum_circuit_entangled_cognition_emotion.rz(np.pi, 0)
quantum_circuit_entangled_cognition_emotion.h(0)
phased_entangled_cognition_emotion_state = Statevector.from_instruction(quantum_circuit_entangled_cognition_emotion)

# INTERFERENCE BETWEEN PHASED ENTANGLED COGNITION-EMOTION AND WILL
full_system_state_vector = phased_entangled_cognition_emotion_state.tensor(state_vector_will)
full_system_quantum_circuit = QuantumCircuit(4, 4)
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

print("Probability of Decision A:", total_probability_of_decision_A)
print("Probability of Decision B:", total_probability_of_decision_B)
print("Probability of Decision C:", total_probability_of_decision_C)

# quantum_circuit_cognition_emotion.draw('mpl')
# plt.show()
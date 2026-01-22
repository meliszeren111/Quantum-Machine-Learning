from pprint import pprint
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Statevector
import numpy as np
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit.visualization import plot_histogram, plot_bloch_multivector
from qiskit.circuit.library import HGate, XGate, RZGate, CXGate, RXGate, RYGate, YGate, ZGate, CZGate
from qiskit.visualization import plot_state_city
import matplotlib.pyplot as plt

# ibm_quantum_api_token = "d80520305a8e1768536a57e3d7c3d729c762062af7065c74f26fc61153933fdaf06efa11775acdd28908682b805d2e18e5a58e9d2092247b36cfd4fea7f89f57"
ibm_quantum_api_token = "915463c814de69b735c3b1e183e0adc69c18600302c6e63b1baa8f7fe155feff8259804643df19d00319fb2e7b1610b96c3acd916b792501fa8c02d6f2f8a010"
provider = QiskitRuntimeService(channel='ibm_quantum', token=ibm_quantum_api_token)
aer_simulator = AerSimulator()
decision_map = {
    "Go Ahead": [
        "000",  # Encouraged + Positive + Initiating
        "001",  # Encouraged + Positive + Maintaining
        "010",  # Encouraged + Ambiguous + Initiating
        "011",  # Encouraged + Ambiguous + Maintaining
    ],

    "Wait and Reassess": [
        "111",  # Uncertain + Ambiguous + Maintaining
        "112",  # Uncertain + Ambiguous + Blocking
        "121",  # Uncertain + Negative + Maintaining
        "110",  # Uncertain + Ambiguous + Initiating
    ],
    "Block or Withdraw": [
        "222",  # Frustrated + Negative + Blocking
        "221",  # Frustrated + Negative + Maintaining
        "220",  # Frustrated + Negative + Initiating
        "122",  # Uncertain + Negative + Blocking
        "212",  # Frustrated + Ambiguous + Blocking
    ]
}

class QML:

    @staticmethod
    def create_one_qutrit_with_two_qubits(amplitudes: list):
        state_vector = np.zeros(4, dtype=complex)
        state_vector[0] = amplitudes[0]  # |00⟩
        state_vector[1] = amplitudes[1]  # |01⟩
        state_vector[2] = amplitudes[2]  # |10⟩
        state_vector = state_vector / np.linalg.norm(state_vector)
        state_vector = Statevector(state_vector)
        return state_vector

    @staticmethod
    def create_one_emotion_qutrit_with_two_qubits(state0_weight, state1_weight, state2_weight, polarity0=1, polarity1=1, polarity2=1):
        state_vector = np.zeros(4, dtype=complex)
        state_vector[0] = state0_weight * polarity0  # |00⟩
        state_vector[1] = state1_weight * polarity1  # |01⟩
        state_vector[2] = state2_weight * polarity2  # |10⟩
        state_vector = state_vector / np.linalg.norm(state_vector)
        return Statevector(state_vector)


    @staticmethod
    def rotate(gate, theta, state_vector, index_of_qubit):
        if gate == "RXGate":
            state_vector_rotated_about_x = state_vector.evolve(RXGate(theta), qargs=[index_of_qubit])
            return state_vector_rotated_about_x
        if gate == "RYGate":
            state_vector_rotated_about_y = state_vector.evolve(RYGate(theta), qargs=[index_of_qubit])
            return state_vector_rotated_about_y
        if gate == "RZGate":
            state_vector_rotated_about_z = state_vector.evolve(RZGate(theta), qargs=[index_of_qubit])
            return state_vector_rotated_about_z
        if gate == "HGate":
            state_vector_applied_h = state_vector.evolve(HGate(), qargs=[index_of_qubit])
            return state_vector_applied_h
        if gate == "XGate":
            state_vector_applied_x = state_vector.evolve(XGate(), qargs=[index_of_qubit])
            return state_vector_applied_x
        if gate == "YGate":
            state_vector_applied_x = state_vector.evolve(YGate(), qargs=[index_of_qubit])
            return state_vector_applied_x
        if gate == "ZGate":
            state_vector_applied_x = state_vector.evolve(ZGate(), qargs=[index_of_qubit])
            return state_vector_applied_x
        if gate == "CXGate":
            state_vector_applied_cx = state_vector.evolve(CXGate(), qargs=[index_of_qubit])
            return state_vector_applied_cx
        if gate == "CYGate":
            state_vector_applied_cx = state_vector.evolve(CXGate(), qargs=[index_of_qubit])
            return state_vector_applied_cx
        if gate == "CZGate":
            state_vector_applied_cx = state_vector.evolve(CZGate(), qargs=[index_of_qubit])
            return state_vector_applied_cx

    @staticmethod
    def initialize_qutrit_on_circuit(quantum_circuit, state_vector, qubit_indices_to_init: list):
        quantum_circuit.initialize(state_vector.data, qubit_indices_to_init)
        return quantum_circuit

    @staticmethod
    def create_entanglement(state_vectors, qubit_list_to_init, control_qubit, target_qubit):
        assert len(state_vectors) >= 2, "At least two states are required."
        assert len(qubit_list_to_init) == sum(len(state_vector.dims()) for state_vector in state_vectors), "qubit_list_to_init length must match total qubits in state vectors."
        combined_state = state_vectors[0]
        for sv in state_vectors[1:]:
            combined_state = combined_state.tensor(sv)
        normalized_state = combined_state / np.linalg.norm(combined_state)
        num_qubits = len(qubit_list_to_init)
        quantum_circuit = QuantumCircuit(num_qubits)
        quantum_circuit.initialize(normalized_state.data, qubit_list_to_init)
        quantum_circuit.h(control_qubit)
        quantum_circuit.cx(control_qubit, target_qubit)
        entangled_state = Statevector.from_instruction(quantum_circuit)
        return entangled_state

    @staticmethod
    def create_phase_difference(state_vector):
        num_qubits_to_init = state_vector.num_qubits
        quantum_circuit = QuantumCircuit(num_qubits_to_init)
        quantum_circuit.initialize(state_vector)
        quantum_circuit.h(0)
        quantum_circuit.rz(np.pi, 0)
        quantum_circuit.h(0)
        phased_state = Statevector.from_instruction(quantum_circuit)
        return phased_state


    @staticmethod
    def create_system_state_vector(state_vectors: list):
        assert len(state_vectors) >= 2, "At least two states are required to combine."
        state_vector = state_vectors[0]
        for state_vec in state_vectors[1:]:
            state_vector = state_vector.tensor(state_vec)
        return state_vector

    @staticmethod
    def simulate_with_aer(system_state_vector, num_shots=1024):
        assert isinstance(system_state_vector, Statevector)
        qubit_list_to_init = list(range(system_state_vector.num_qubits))
        num_qubits_to_init = len(qubit_list_to_init)
        num_qubit_registers = num_qubits_to_init
        num_cbit_registers = num_qubits_to_init
        qubit_list_to_measure = list(range(num_qubits_to_init))
        cbit_list_to_measure = list(range(num_qubits_to_init))
        quantum_circuit = QuantumCircuit(num_qubit_registers, num_cbit_registers)
        quantum_circuit.initialize(system_state_vector.data, qubit_list_to_init)
        quantum_circuit.measure(qubit_list_to_measure, cbit_list_to_measure)
        transpiled_gate = transpile(quantum_circuit, aer_simulator)
        job = aer_simulator.run(transpiled_gate, shots=num_shots)
        result = job.result()
        counts = result.get_counts()
        total = sum(counts.values())
        prob_dict = {state: count / total for state, count in counts.items()}
        return prob_dict, counts

    @staticmethod
    def make_decision(prob_dict: dict, decision_map: dict):
        decision_probs = {key: 0.0 for key in decision_map}
        decision_counts = {key: 0 for key in decision_map}
        def convert_6bit_to_qutrit_label(bitstring):
            label = ""
            for i in range(0, 6, 2):
                pair = bitstring[i:i + 2]
                if pair == '00':
                    label += '0'
                elif pair == '01':
                    label += '1'
                elif pair == '10':
                    label += '2'
                else:
                    return None
            return label
        for bitstring, prob in prob_dict.items():
            qutrit_label = convert_6bit_to_qutrit_label(bitstring)
            if qutrit_label is None:
                continue  # geçersiz qutrit durumu ('11')
            for decision, patterns in decision_map.items():
                if qutrit_label in patterns:
                    decision_probs[decision] += prob
                    decision_counts[decision] += 1
                    break
        avg_decision_probs = {
            key: round(decision_probs[key] / decision_counts[key], 4)
            if decision_counts[key] > 0 else 0.0
            for key in decision_map
        }
        return avg_decision_probs

    @staticmethod
    def create_system_circuit(system_state_vector, num_qubits_to_init, qubit_indices=None, measure=False):
        if qubit_indices is None:
            qubit_indices = list(range(num_qubits_to_init))
        quantum_circuit = QuantumCircuit(num_qubits_to_init, num_qubits_to_init if measure else 0)
        quantum_circuit.initialize(system_state_vector.data, qubit_indices)
        if measure:
            quantum_circuit.measure(qubit_indices, qubit_indices)
        return quantum_circuit

    @staticmethod
    def visualize_circuit(quantum_circuit, title):
        quantum_circuit.draw("mpl")
        plt.title(title)
        plt.show()
        return plt

    @staticmethod
    def visualize_decomposed_circuit(quantum_circuit, title):
        decomposed_circuit = quantum_circuit.decompose()
        decomposed_circuit.draw("mpl")
        plt.title(title)
        plt.show()
        return plt

    @staticmethod
    def visualize_state_vector(state_vector, title):
        plot_state_city(state_vector, title=title)
        plt.show()
        return plt

    @staticmethod
    def visualize_measurements(counts, title):
        plot_histogram(counts, title=title)
        plt.xticks(rotation=70)
        plt.tight_layout()
        plt.show()
        return plt

    @staticmethod
    def visualize_decision(decision_probs, bar_color_list, y_label, title="Decisions"):
        plt.bar(decision_probs.keys(), decision_probs.values(), color=bar_color_list)
        plt.title(title)
        plt.ylabel(y_label)
        plt.xticks(rotation=70)
        plt.ylim(0, 1)
        plt.show()
        return plt

    @staticmethod
    def run_system(cognition_amplitudes, will_amplitudes, state0_weight, state1_weight, state2_weight, polarity0, polarity1, polarity2):
        cognitive_state = QML.create_one_qutrit_with_two_qubits(cognition_amplitudes)
        will_state = QML.create_one_qutrit_with_two_qubits(will_amplitudes)
        emotional_state = QML.create_one_emotion_qutrit_with_two_qubits(state0_weight, state1_weight, state2_weight, polarity0, polarity1, polarity2)
        states_to_be_entangled = [emotional_state, cognitive_state]
        qubit_list_to_init_in_entanglement = list(range(2 * len(states_to_be_entangled)))
        entangled_state = QML.create_entanglement([emotional_state, cognitive_state], qubit_list_to_init_in_entanglement, control_qubit_in_entanglement, target_qubit_in_entanglement)
        phased_entangled_state = QML.create_phase_difference(entangled_state)
        system_state_vector = QML.create_system_state_vector([phased_entangled_state, will_state])
        num_qubits_to_init_in_state_circuit = int(np.log2(len(system_state_vector)))
        system_circuit = QML.create_system_circuit(system_state_vector, num_qubits_to_init_in_state_circuit, qubit_indices=None, measure=True)
        decision_prob_dict, measurement_counts = QML.simulate_with_aer(system_state_vector, num_shots)
        averaged_decision_probs = QML.make_decision(decision_prob_dict, decision_map)
        return system_state_vector, system_circuit, decision_prob_dict, averaged_decision_probs, measurement_counts

    @staticmethod
    def conclude(avg_decision_probs: dict) -> str:
        selected_decision = max(avg_decision_probs, key=avg_decision_probs.get)
        if selected_decision == "Go Ahead":
            return "I am happy to proceed."
        elif selected_decision == "Wait and Reassess":
            return  "I will wait and see."
        elif selected_decision == "Block or Withdraw":
            return "I am giving up."
        else:
            return "I could not decide."

    @staticmethod
    def visualize(system_circuit, system_circuit_title, measurement_counts, measurement_title, decision_probs, bar_color_list, bar_y_label, decision_title):
        QML.visualize_circuit(system_circuit, system_circuit_title)
        QML.visualize_decomposed_circuit(system_circuit, system_circuit_title)
        QML.visualize_measurements(measurement_counts, measurement_title)
        QML.visualize_decision(decision_probs, bar_color_list, bar_y_label, title=decision_title)
        plt.show()

    @staticmethod
    def change_state(gate, theta, state_vector, index_of_qubit):
        changed_state = QML.rotate(gate, theta, state_vector, index_of_qubit)
        return changed_state

    @staticmethod
    def change_state_of_mind(cognition_amplitudes, will_amplitudes, state0_weight, state1_weight, state2_weight, polarity0, polarity1, polarity2):
        cognitive_state = QML.create_one_qutrit_with_two_qubits(cognition_amplitudes)
        will_state = QML.create_one_qutrit_with_two_qubits(will_amplitudes)
        emotional_state = QML.create_one_emotion_qutrit_with_two_qubits(state0_weight, state1_weight, state2_weight, polarity0, polarity1, polarity2)
        changed_cognitive_state = QML.change_state("RXGate", np.pi, cognitive_state, 0)
        changed_cognitive_state = QML.change_state("RXGate", np.pi/2, changed_cognitive_state, 1)
        changed_will_state = QML.change_state("RZGate", np.pi/3, will_state, 0)
        changed_emotional_state = QML.change_state("RYGate", np.pi/9, emotional_state, 0)
        changed_emotional_state = QML.change_state("RXGate", np.pi, changed_emotional_state, 1)
        changed_emotional_state = QML.change_state("RXGate", np.pi, changed_emotional_state, 1)
        return changed_cognitive_state, changed_will_state, changed_emotional_state

    @staticmethod
    def run_changed_system(cognition_amplitudes, will_amplitudes, state0_weight, state1_weight, state2_weight, polarity0, polarity1, polarity2):
        changed_cognitive_state, changed_will_state, changed_emotional_state = QML.change_state_of_mind(cognition_amplitudes, will_amplitudes, state0_weight, state1_weight, state2_weight, polarity0, polarity1, polarity2)
        states_to_be_entangled = [changed_emotional_state, changed_cognitive_state]
        qubit_list_to_init_in_entanglement = list(range(2 * len(states_to_be_entangled)))
        entangled_state = QML.create_entanglement([changed_emotional_state, changed_cognitive_state], qubit_list_to_init_in_entanglement, control_qubit_in_entanglement, target_qubit_in_entanglement)
        phased_entangled_state = QML.create_phase_difference(entangled_state)
        system_state_vector = QML.create_system_state_vector([phased_entangled_state, changed_will_state])
        num_qubits_to_init_in_state_circuit = int(np.log2(len(system_state_vector)))
        system_circuit = QML.create_system_circuit(system_state_vector, num_qubits_to_init_in_state_circuit, qubit_indices=None, measure=True)
        decision_prob_dict, measurement_counts = QML.simulate_with_aer(system_state_vector, num_shots)
        averaged_decision_probs = QML.make_decision(decision_prob_dict, decision_map)
        return system_state_vector, system_circuit, decision_prob_dict, averaged_decision_probs, measurement_counts


qubit_list_to_init_in_entanglement = [0, 1]
qubit_list_to_init_in_interference = [0, 1]
qubit_list_to_init_in_system = [0, 1, 2, 3]
qubit_list_to_init_in_simulation = [0, 1, 2, 3]
num_qubit_registers = 4
num_cbit_registers = 4
control_qubit_in_entanglement = 0
target_qubit_in_entanglement = 1
qubit_list_to_measure = [0, 1, 2, 3]
cbit_list_to_measure = [0, 1, 2, 3]
num_shots = 1000
system_circuit_title = "System Circuit"
measurement_title = "Measurement"
bar_color_list = ["blue", "green", "red"]
bar_y_label = "Probability"
decision_title = "Decision"
system_state_vector_title = "System State"
state0_weight = 0.3
state1_weight = 0.2
state2_weight = 0.5
polarity0 = 1
polarity1 = 1
polarity2 = 1
cognition_amplitudes = [0.2, 0.3, 0.5]
will_amplitudes = [0.1, 0.8, 0.1]
num_labels_to_show = 10
system_state_vector1, system_circuit, decision_prob_dict, averaged_decision_probs, measurement_counts = QML.run_system(
                        cognition_amplitudes=cognition_amplitudes,
                        will_amplitudes=will_amplitudes,
                        state0_weight=state0_weight,
                        state1_weight=state1_weight,
                        state2_weight=state2_weight,
                        polarity0=polarity0,
                        polarity1=polarity1,
                        polarity2=polarity2,

)
print("decision_prob_dict: ")
pprint(decision_prob_dict)
print("averaged_decision_probs: ")
pprint(averaged_decision_probs)
QML.visualize(
            system_circuit = system_circuit,
            system_circuit_title = system_circuit_title,
            measurement_counts = measurement_counts,
            measurement_title = measurement_title,
            decision_probs = decision_prob_dict,
            bar_color_list = bar_color_list,
            bar_y_label = bar_y_label,
            decision_title = decision_title,
)
conclusion = QML.conclude(averaged_decision_probs)
print("conclusion with initial state of mind: ", conclusion)

system_state_vector2, system_circuit, decision_prob_dict, averaged_decision_probs, measurement_counts = QML.run_changed_system(
                        cognition_amplitudes=cognition_amplitudes,
                        will_amplitudes=will_amplitudes,
                        state0_weight=state0_weight,
                        state1_weight=state1_weight,
                        state2_weight=state2_weight,
                        polarity0=polarity0,
                        polarity1=polarity1,
                        polarity2=polarity2,

)
print("decision_prob_dict: ")
pprint(decision_prob_dict)
print("averaged_decision_probs: ")
pprint(averaged_decision_probs)
QML.visualize(
            system_circuit = system_circuit,
            system_circuit_title = system_circuit_title,
            measurement_counts = measurement_counts,
            measurement_title = measurement_title,
            decision_probs = decision_prob_dict,
            bar_color_list = bar_color_list,
            bar_y_label = bar_y_label,
            decision_title = decision_title,
)
conclusion = QML.conclude(averaged_decision_probs)
print("conclusion with changed state of mind: ", conclusion)

plot_bloch_multivector(system_state_vector1)
plt.title("Bloch Sphere for Initial Mind (Qubits in |00⟩)")
plt.show()

plot_bloch_multivector(system_state_vector2)
plt.title("Bloch Sphere for Changed Mind (Qubits in |00⟩)")
plt.show()

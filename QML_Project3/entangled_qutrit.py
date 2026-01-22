from qiskit.quantum_info import Statevector
from qiskit_ibm_runtime import QiskitRuntimeService
ibm_quantum_api_token = "d80520305a8e1768536a57e3d7c3d729c762062af7065c74f26fc61153933fdaf06efa11775acdd28908682b805d2e18e5a58e9d2092247b36cfd4fea7f89f57"
service = QiskitRuntimeService.save_account(token=ibm_quantum_api_token, channel="ibm_quantum", overwrite=True)
psi = Statevector.from_label('012')
probs = psi.probabilities([0, 1, 2])
print('probs: {}'.format(probs))
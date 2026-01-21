import yaml
from qiskit_ibm_runtime import QiskitRuntimeService

yaml_path = r"data.yaml"
with open(yaml_path, "r") as file:
    yaml_data = yaml.load(file, Loader=yaml.FullLoader)
ibm_quantum_api_token = yaml_data["ibm_quantum_api_token"]
service = QiskitRuntimeService(channel="ibm_quantum", token=ibm_quantum_api_token)
real_devices = service.backends(simulator=False, operational=True)
for backend in real_devices:
    print(f"{backend.name}: pending jobs = {backend.status().pending_jobs}")
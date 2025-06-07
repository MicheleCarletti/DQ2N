"""
@author Michele Carletti
Run a trained DQ2N model on IBM quantum machines
"""
import gymnasium as gym
import pennylane as qml
import torch
import matplotlib.pyplot as plt
import random
import os
import logging
import csv
from datetime import datetime
from vqc_cartpole import QuantumDQN as DQ2N
from qiskit import QuantumCircuit, transpile
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2

# Log-in IBM Quantum Platform
QiskitRuntimeService.save_account(
    channel="ibm_quantum",
    token="c620be87e046720516e35c033ff66518354b5f6944da1b4ae6a54ebecd692a8dc9381791410a34ea7204777dd829a00dbde6c84bcf818572d8bd552c30107806",
    overwrite=True
)
service = QiskitRuntimeService(channel="ibm_quantum")
instances = service.instances() # Get service instances
# Select the least busy backend
backend = service.least_busy(
    min_num_qubits=4,
    instance="ibm-q/open/main",
)
print("Selected backend:", backend.name)
sampler = SamplerV2(mode=backend)   # Initialize the sampler

num_episodes = 100
num_qubits = 4
num_layers = 3
n_shots = 1024    # Number of shots for each circuit execution
comp_mode = True   # Compare real qiskit and simulated pennylane results 
params_path = "cartpole/v1_3l/850ep_0.01.pth" if num_layers == 3 else "cartpole/v1_4l/900ep_0.01.pth"    # Trained parameters path
dev = qml.device("default.qubit", wires=num_qubits)

# Log file data
os.makedirs("logs", exist_ok=True)
log_filename = f"logs/run_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename, mode='w'),
        logging.StreamHandler()
    ]
)
# Suppress no needed logging info
for name in logging.root.manager.loggerDict:
    if name.startswith("qiskit"):
        logging.getLogger(name).setLevel(logging.CRITICAL + 1)

# Header
logging.info("==== RUN HEADER ====")
logging.info(f"Number of Qubits: {num_qubits}")
logging.info(f"VQC Layers: {num_layers}")
logging.info(f"Trained Parameters File: {params_path}")
logging.info(f"Quantum Backend: {backend.name}")
logging.info(f"Number of Shots per Execution: {n_shots}")
logging.info(f"Comparison mode: {comp_mode}")
logging.info("====================\n")

# Create a csv in comparison mode to track differences between real QC and Pennylane sim
if comp_mode:
    csv_file = f"logs/comp_log_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"
    with open(csv_file, mode='w', newline="") as file:
        writer = csv.writer(file, delimiter=';')
        writer.writerow(
            ['EPisode']+
            [f'diff_expZ_{i}' for i in range(num_qubits)]+
            [f'diff_Q_{i}' for i in range(2)]+
            ['action_diff']
        )

def embedding(features):
    """
    Quantum angle embedding
    """
    for i in range(num_qubits):
        qml.RX(features[i], wires=i)
    

def ansatz(params):
    """
    Ansatz: parametric rotations and cyclical entanglement
    """
    for i in range(num_qubits):
        qml.RY(params[i], wires=i)

    for i in range(num_qubits):
        qml.RZ(params[i + num_qubits], wires=i)

    for i in range(num_qubits):
        qml.CNOT(wires=[(num_qubits - 2 - i) % num_qubits, (num_qubits - i - 1) % num_qubits])

@qml.qnode(dev, interface="torch")
def quantum_model(params, x):
    """
    VQC: embedding + ansatz 
    """
    embedding(x)
    ansatz(params[:2*num_qubits])
    embedding(x)
    ansatz(params[2*num_qubits:4*num_qubits])
    embedding(x)
    ansatz(params[4*num_qubits:6*num_qubits])
    if num_layers == 4:
        embedding(x)
        ansatz(params[6*num_qubits:8*num_qubits])
    
    return [qml.expval(qml.PauliZ(i)) for i in range(num_qubits)]

def compute_expZ(counts, num_qubits):
    """
    Compute the Pauli-Z expectation value for each qubit.
    Given the counts dictionary, {'00': 512, '01': 120, ...}, returns a list with expZ values
    """
    shots = sum(counts.values())    # Compute the number of shots
    expZ = [0.0] * num_qubits   # Initialize an empty list

    # Scan every bitstring and its count
    for bitstr, c in counts.items():

        # For each qubit j, add or subtract according to the rule
        for j, bit in enumerate(reversed(bitstr)):
            # Qiskit uses a little-endian format: index n => qubit 0, index n-1 => qubit 1, ...
            value = 1 if bit == '0' else -1
            expZ[j] += value * c
    
    # Normalize
    return [z / shots for z in expZ]

def select_action(state, policy_net, qweights, episode_num):
    """
    Choose the action according to deterministic policy
    """
    action_dict = {0.0:"Push to the left", 1.0:"Push to the right"}
    action_diff_flag = 0    # Flag to check if different actions are taken
    state = torch.FloatTensor(state).unsqueeze(0)   # shape (1,8)
    with torch.no_grad():
        # Get the embedding 
        x_emb = policy_net.embedding(state).squeeze(0)  # shape (8)
        # Compile the VQC with the learned weights and the current state embedding
        quantum_model(qweights, x_emb)
        # Convert the VQC to QASM string
        qasm_str = quantum_model.qtape.to_openqasm()
        logging.info("Pennylane -> OpenQASM")
        # Convert QASM to Qiskit
        qc_qiskit = QuantumCircuit.from_qasm_str(qasm_str)
        logging.info("OpenQASM -> Qiskit")
        # With SamplerV2 have to manually convert to ISA (Instruction Set Architecture) circuit, compatible with selected backend
        qc_isa = transpile(qc_qiskit, backend=backend, optimization_level=3)
        logging.info(f"Submitting the job to {backend.name}")
        # Run as a single job
        job = sampler.run(
            pubs = [qc_isa],
            shots = n_shots
        )
        prim_res = job.result()  # Get PrimitiveResult
        logging.info("Job completed!")
        res = prim_res[0]   # Get first SamplerPubResult
        counts = res.join_data().get_counts()    # Count results 
        #print(f"Counts: {counts}")
        expZ_values = compute_expZ(counts, num_qubits)  # Compute the expZ values
        logging.info(f"Pauli-Z expectation: {expZ_values}")
        q_values = policy_net.fc_out(torch.Tensor(expZ_values))
        logging.info(f"Q-values: {q_values.tolist()}")
        action = torch.argmax(q_values).item()
        logging.info(f"Action: {action_dict[action]}")
        
        if comp_mode:
            logging.info("Pennylane simulation")
            xe = policy_net.embedding(state)    
            pen_expZ = policy_net.quantum(xe).squeeze()
            logging.info(f"Pennylane expZ: {pen_expZ.tolist()}")
            pen_q_values = policy_net.fc_out(pen_expZ).squeeze()
            logging.info(f"Pennylane Q-values: {pen_q_values.tolist()}")
            pen_action = torch.argmax(pen_q_values).item()
            logging.info(f"Pennylane action: {action_dict[pen_action]}")

            # Compute distances
            diff_expZ = [round(float(abs(p - s)), 3) for p,s in zip(pen_expZ, expZ_values)]
            diff_Q = [round(float(abs(p - s)), 3) for p,s in zip(pen_q_values, q_values.tolist())]
            action_diff_flag = int(pen_action != action)

            # Write in CSV
            with open(csv_file, mode='a', newline='') as file:
                writer = csv.writer(file, delimiter=';')
                writer.writerow([episode_num] + diff_expZ + diff_Q + [action_diff_flag])

    return action

if __name__ == "__main__":

    # Set-up the environment
    env = gym.make("CartPole-v1", render_mode="None")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    model = DQ2N(state_dim, action_dim)
    model.load_state_dict(torch.load(params_path))
    model.eval()

    # Get VQC parameters
    quantum_weights = model.quantum.weights.detach()

    logging.info("=== Start Run ===\n")


    for episode in range(num_episodes):
        logging.info(f"--- Episode {episode + 1}/{num_episodes} ---")

        state, _ = env.reset(seed=random.randint(0, 1000))
        done = False
        total_reward = 0

        while not done:
            action = select_action(state, model, quantum_weights, episode+1)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            state = next_state
            total_reward += reward
            print(total_reward)
        
        logging.info(f"Episode Reward: {total_reward}\n")

    logging.info("=== Ended ===")

    
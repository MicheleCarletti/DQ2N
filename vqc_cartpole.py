"""
@author Michele Carletti
Quantum DQN for CartPole-v1
v1.0 - 4 layers 
"""

import gymnasium as gym
import pennylane as qml
from pennylane import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import warnings
import random
import time
import csv
import os

from qiskit_ibm_runtime import QiskitRuntimeService
from Prioritized_experience_replay import PER

warnings.filterwarnings("ignore")
real_quantum_hw = False # Flag to run on real quantum computer

if real_quantum_hw:
    QiskitRuntimeService.save_account(channel="ibm_quantum", token="aafb84630733e5bb1e0a602eddbc7f9a88c1a54c9bd6d781ead9fb381fcdc189f7e7078077b85aefdecc49d3faa693e43482c1a600a23139305e61907a683af7",
                                  overwrite=True)

    service = QiskitRuntimeService(channel="ibm_quantum", instance="ibm-q/open/main")


# ------------------ Quantum part ------------------
num_qubits = 4

if real_quantum_hw:
    backend = service.least_busy(operational=True, simulator=False, min_num_qubits=num_qubits)
    dev = qml.device('qiskit.remote', wires=127, backend=backend)
else:
     dev = qml.device("default.qubit", wires=num_qubits)

def embedding(features):
    """
    Quantum angle embedding
    """
    for i in range(num_qubits):
        qml.RX(features[i], wires=i)
    
    if not real_quantum_hw:
        qml.Barrier(range(num_qubits))    #No barrier when running on qiskit

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
    
    if not real_quantum_hw:
        qml.Barrier(range(num_qubits)) #No barrier when running on qiskit

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
    embedding(x)
    ansatz(params[6*num_qubits:8*num_qubits])
    
    return [qml.expval(qml.PauliZ(i)) for i in range(num_qubits)]


def print_circuit():
    features = 0.01 * torch.randn(8 * num_qubits)
    x = torch.randn(8)
    print(features)
    fig, ax = qml.draw_mpl(quantum_model, show_all_wires=True, decimals=2)(features, x)
    plt.show()

class EncodingLayer(nn.Module):
    def __init__(self, state_dim):
        """ 
        Classical embedding layer. Out range [-pi, pi]
        """
        super().__init__()
        self.weights = nn.Parameter(torch.Tensor(state_dim))
        torch.nn.init.uniform_(self.weights, -1, 1) # Initialization
    
    def forward(self, x):
        if not isinstance(x, torch.Tensor):
            x = torch.Tensor(x)
        x = self.weights * x
        x = torch.tanh(x) * torch.pi
        return x

# ------------------ Quantum Layer ------------------
class QuantumLayer(nn.Module):
    def __init__(self, n_qubits, n_weights):
        super().__init__()
        self.n_qubits = n_qubits
        self.weights = nn.Parameter(0.1 * torch.randn(n_weights))
    
    def forward(self, x):
        batch_size = x.shape[0]
        outputs = []

        # Iterate over each state
        for i in range(batch_size):
            qc_out = quantum_model(self.weights, x[i])
            # If list convert to Tensor
            if isinstance(qc_out, list):
                qc_out = torch.stack(qc_out)
            outputs.append(qc_out)
        # Stack results and convert to float
        qc_out = torch.stack(outputs).float()

        return qc_out


# ------------------ Quantum Q-Network ------------------
class QuantumDQN(nn.Module):
    def __init__(self, state_dim,  n_actions):
        super(QuantumDQN, self).__init__()
        # Quantum parameters randomly initialized
        self.embedding = EncodingLayer(state_dim)
        self.quantum = QuantumLayer(num_qubits, 8 * num_qubits)
        self.fc_out = nn.Sequential(
            nn.ReLU(),
            nn.Linear(num_qubits, n_actions)
        )

    def forward(self, x):
        x_e = self.embedding(x)
        x_q = self.quantum(x_e)
        q_values = self.fc_out(x_q)   # shape (batch_size, n_actions)
        return q_values
       
# ------------------ Classical DQN ------------------
# Neural network for Q function
class DQN(nn.Module):
    def __init__(self, state_dim, action_dim, num_hidden=64):
        """ Define the model to learn the Q function"""
        super(DQN, self).__init__()
        self.h2 = num_hidden // 2
        self.fc1 = nn.Linear(state_dim, num_hidden)
        self.fc2 = nn.Linear(num_hidden, self.h2)
        self.fc3 = nn.Linear(self.h2, action_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

def select_action(state, policy_net, epsilon, action_dim):
    """ Choose the action via epsilon-greedy policy"""
    if np.random.rand() < epsilon:
        return random.randrange(action_dim)  # Explore: choose a random action
    state = torch.FloatTensor(state).unsqueeze(0).to(device)
    with torch.no_grad():
        q_values = policy_net(state)
    return torch.argmax(q_values).item()  # Exploit: choose the action with the maximum Q value

def train(memory, policy_net, target_net, optimizer, batch_size, gamma, beta, losses, lambda_reg = 0.5):
    """ Train the DQN model"""
    if len(memory.buffer) < batch_size:
        return 0.0, 0.0
    
    (states, actions, rewards, next_states, dones), indicies, weights = memory.sample(batch_size, beta)

    states = torch.FloatTensor(states).to(device)
    actions = torch.LongTensor(actions).unsqueeze(1).to(device)
    rewards = torch.FloatTensor(rewards).to(device)
    next_states = torch.FloatTensor(next_states).to(device)
    dones = torch.FloatTensor(dones).to(device)
    weights = torch.FloatTensor(weights).to(device)

    # Compute actual Q values
    q_values = policy_net(states).gather(1, actions)

    # Compute target Q values
    with torch.no_grad():
        next_q_values = target_net(next_states).max(1)[0]
        target_q_values = rewards + (1 - dones) * gamma * next_q_values

    # Compute the loss
    loss = (weights * (q_values.squeeze() - target_q_values)**2).mean()  # Weighted MSE
    
    if mixture_mode:
        # Add regularization term based on target q-function
        with torch.no_grad():
            q_fun_target = target_net(states)
        q_fun_policy = policy_net(states)
        reg_term = torch.nn.functional.mse_loss(q_fun_target, q_fun_policy)
        loss = loss + lambda_reg * reg_term

    optimizer.zero_grad()
    loss.backward()
    # Compute L2 norm gradients
    grad_norm = 0.0
    for param in policy_net.parameters():
        if param.grad is not None:
            grad_norm += param.grad.data.norm(2).item() ** 2
    grad_norm = grad_norm ** 0.5

    torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 3.0)  # gradient clipping
    optimizer.step()

    losses.append(loss.item())
    # Update PER priorities
    priorities = (q_values.squeeze() - target_q_values).abs().cpu().detach().numpy() + 1e-5
    memory.update_priorities(indicies, priorities)
    return loss.item(), grad_norm

def prepare_results(reward_history, mar, loss_history, hpc):
    """ Analize training results"""

    if not hpc:

        # Plotting rewards
        plt.figure(figsize=(10, 8))
        plt.plot(reward_history, label='Reward per episode')
        plt.plot(mar, label='Moving average (100 episodes)')
        plt.title(f"Total Rewards\n[{session_name}]")
        plt.xlabel("Episode")
        plt.ylabel("Total Reward")
        plt.legend()
        plt.grid()
        plt.show()

        # Plotting loss
        plt.figure(figsize=(10, 8))
        plt.plot(loss_history)
        plt.title(f"Loss per Episode\n[{session_name}]")
        plt.xlabel("Episode")
        plt.ylabel("Loss")
        plt.grid()
        plt.show()

def test_model():
    single_state = torch.randn(8).unsqueeze(0) # Add batch dimension
    batched_state = torch.randn(10, 8).to(device)
    model = QuantumDQN(8, 4).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    base = torch.randn(8,4).to(device)
    target = torch.sin(batched_state @ base).to(device)    # Target values depend on state: size(10, 4)
    for i in range(160):
        q_values = model(batched_state)
        for name, param in model.quantum.named_parameters():
            print(f"{name} - params: {param.data},\n grads: {param.grad}")
        #print(f"q_values: {q_values}\ntarget_values: {target}")
        loss = torch.nn.functional.mse_loss(q_values, target) 
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f"Step: {i+1}: loss {loss:.3f}")
    print(f"Actions: {torch.argmax(model(batched_state), dim=1)}\nTarget actions: {torch.argmax(target, dim=1)}")

if __name__ == "__main__":

    #### Model parameters ###
    gamma = 0.99    # Discount factor
    epsilon = 1.0   # Initial exploration probability
    epsilon_min = 0.01  
    epsilon_decay = 0.995
    learning_rate = 1e-2 #0.0005
    batch_size = 16
    max_memory_size = 100000
    n_episodes = 1000
    target_net_freq = 10    # Update frequency for target network
    alpha = 0.6
    beta = 0.4
    lambda_reg = 0.5    # Regularization in case of mixture quantum and classical
    beta_increment_per_episode = 0.001
    running_on_hpc = False
    mixture_mode = False    # Mixture_mode => Quantum policy net, classical target net already trained
    from_checkpoint = False
    #######################

    # Set-up the environment
    env = gym.make("CartPole-v1", render_mode="None")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    # Define policy and target networks
    device = "cpu"  #torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    #print_circuit()
    #test_model()
    policy_net = QuantumDQN(state_dim, action_dim).to(device)
    if from_checkpoint:
            checkpoint_name = "cartpole/v1_4l/350ep_0.17.pth"
            policy_net.load_state_dict(torch.load(checkpoint_name, map_location=torch.device(device)))
    if mixture_mode:
        target_net_name = "../../lunar_lander/RL/ReinforcementLearning/models/models_with_PER/DQN_256h_10000e_23-11-2024_PER_hpc.pth"
        target_path = os.path.abspath(target_net_name)
        target_net = DQN(state_dim, action_dim, 256).to(device)
        target_net.load_state_dict(torch.load(target_path, map_location=torch.device(device)))
    else:
        target_net = QuantumDQN(state_dim, action_dim).to(device)
        target_net.load_state_dict(policy_net.state_dict())

    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=learning_rate)
    memory = PER(max_memory_size, alpha)

    # Save rewards
    rewards_history = []
    loss_history = []
    model_saved = False

    # Rewards moving average
    moving_avg_period = 100
    moving_avg_rewards = []

    if running_on_hpc:
        session_name = f"DQ2N_CartPole1_v1_{n_episodes}e_hpc"
    else:
        session_name = f"DQ2N_CartPole1_v1_{n_episodes}e"

    with open(f"cartpole/v1_4l/{session_name}.csv", mode='w', newline="") as file:
        writer = csv.writer(file, delimiter=';')
        writer.writerow(["Episode", "Total Reward", "Loss", "Gradient L2", "Epsilon", "Elapsed time [s]"]) # Header

        print("Training started")
        # Start the training 
        for episode in range(n_episodes):
            start_time = time.time()
            state, _ = env.reset(seed=random.randint(0, 1000))
            done = False
            total_reward = 0
            episode_loss = []
            n_step = 0


            while not done:
                action = select_action(state, policy_net, epsilon, action_dim)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                
                memory.add(state, action, reward, next_state, done)
                state = next_state
                total_reward += reward


                loss, grad = train(memory, policy_net, target_net, optimizer, batch_size, gamma, beta, episode_loss, lambda_reg)
                n_step += 1
        
                
            rewards_history.append(total_reward)    # Track rewards

            # Compute moving average
            if len(rewards_history) >= moving_avg_period:
                moving_avg_rewards.append(np.mean(rewards_history[-moving_avg_period:]))
            else:
                moving_avg_rewards.append(np.mean(rewards_history))

            #episode_loss = np.array(episode_loss)
            #loss_history.append(episode_loss.sum() / len(episode_loss))
            loss_history.append(loss)

            # Save a checkpoint every 30 episodes
            if episode % 30 == 0 and episode != 0:
                torch.save(policy_net.state_dict(), f"cartpole/v1_4l/{episode}ep_{epsilon:.2f}.pth") # Save the model
            
            # Decrease the epsion value
            if epsilon > epsilon_min:
                epsilon *= epsilon_decay
        
            beta = min(1.0, beta + beta_increment_per_episode)  # Update beta value

            if not mixture_mode:
                # Update the target network
                if episode % target_net_freq == 0:
                    target_net.load_state_dict(policy_net.state_dict())
            
            end_time = time.time()
            elapsed_time = end_time - start_time

            print(f"Episode {episode + 1}/{n_episodes}, Total Reward: {total_reward:.2f}, Loss: {loss:.3f}, Gradient L2: {grad:.3f}, Epsilon: {epsilon:.2f}, ET: {elapsed_time:.2f} s")
            # Write episode data to CSV
            writer.writerow([episode+1, round(total_reward,3), round(loss,3), round(grad,3), round(epsilon,3), round(elapsed_time,2)])

    env.close()

    prepare_results(rewards_history, moving_avg_rewards, loss_history, running_on_hpc)
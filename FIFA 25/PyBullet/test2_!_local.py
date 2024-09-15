import pybullet as p
import time
import pybullet_data
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
import keyboard  # To capture keyboard input for stopping training

class SimulationEnv(gym.Env):
    def __init__(self):
        super(SimulationEnv, self).__init__()

        # Connect to PyBullet
        self.physics_client = p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        
        self.agent = Agent()

        # Define action and observation spaces
        self.action_space = spaces.Box(low=-1, high=1, shape=(self.agent.num_joints,), dtype=np.float32)
        observation_dim = self.agent.num_joints * 2
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(observation_dim,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.agent.reset()
        return self._get_observation().astype(np.float32), {}

    def step(self, action):
        self.agent.apply_action(action)
        p.stepSimulation()
        time.sleep(1./240.)  # Slow down the simulation for better visualization
        
        observation = self._get_observation().astype(np.float32)
        reward = self._calculate_reward()
        done = self._is_done()
        info = {}

        return observation, reward, done, info

    def _get_observation(self):
        joint_positions, joint_velocities = self.agent.get_joint_states()
        return np.concatenate([joint_positions, joint_velocities])

    def _calculate_reward(self):
        base_position, _ = p.getBasePositionAndOrientation(self.agent.humanoid)
        height_reward = base_position[2]
        return height_reward

    def _is_done(self):
        base_position, _ = p.getBasePositionAndOrientation(self.agent.humanoid)
        return base_position[2] < 0.5

    def render(self, mode='human'):
        pass

    def close(self):
        p.disconnect()

class Agent:
    def __init__(self):
        # Load the humanoid URDF model
        self.humanoid = p.loadURDF("URDFs/human.urdf", [0, 0, 1], p.getQuaternionFromEuler([0, 0, 0]))
        self.num_joints = p.getNumJoints(self.humanoid)

    def reset(self):
        # Reset all joints to initial state
        for joint_index in range(self.num_joints):
            p.resetJointState(self.humanoid, joint_index, targetValue=0, targetVelocity=0)

    def apply_action(self, action):
        # Apply action to each joint
        for i, joint_index in enumerate(range(self.num_joints)):
            p.setJointMotorControl2(self.humanoid, joint_index, p.POSITION_CONTROL, targetPosition=action[i])

    def get_joint_states(self):
        joint_positions = []
        joint_velocities = []
        for joint_index in range(self.num_joints):
            joint_state = p.getJointState(self.humanoid, joint_index)
            joint_positions.append(joint_state[0])
            joint_velocities.append(joint_state[1])
        return np.array(joint_positions), np.array(joint_velocities)

if __name__ == "__main__":
    env = SimulationEnv()
    check_env(env)  # Check if the environment is valid
    model = PPO("MlpPolicy", env, verbose=1)

    try:
        while True:
            # Train the model for a small number of steps at a time
            model.learn(total_timesteps=1000, reset_num_timesteps=False)
            
            # Check if the 's' key is pressed to stop training
            if keyboard.is_pressed('s'):
                print("Training interrupted. Saving the model...")
                model.save("ppo_humanoid")
                break

    except KeyboardInterrupt:
        print("Training interrupted by user. Saving the model...")
        model.save("ppo_humanoid")
    
    env.close()

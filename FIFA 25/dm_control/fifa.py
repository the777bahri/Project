import os
import numpy as np
from dm_control.locomotion import soccer as dm_soccer
from dm_control.locomotion.soccer import camera
from dm_control.locomotion.soccer import soccer_ball  # Import the soccer ball module
import imageio

# Configure MuJoCo for software rendering (No GPU required)
os.environ["MUJOCO_GL"] = "glfw"

class SoccerEnvironment:
    """Class to handle the soccer environment setup and simulation."""

    def __init__(self):
        # Initialize environment
        self.env = dm_soccer.load(
            team_size=2,
            time_limit=10.0,
            disable_walker_contacts=False,
            enable_field_box=True,
            terminate_on_goal=False,
            walker_type=dm_soccer.WalkerType.HUMANOID
        )
        self.action_specs = self.env.action_spec()
        self.timestep = self.env.reset()

        # Initialize the soccer ball
        self.soccer_ball = soccer_ball.regulation_soccer_ball()

        # Add soccer ball to the environment
        self.env.task.arena.add_free_entity(self.soccer_ball)

        # Create a random state for initialization
        random_state = np.random.RandomState(42)

        # Compile ball with environment physics
        self.soccer_ball.after_compile(self.env.physics, random_state)

        # Register players with the soccer ball
        self.register_players()

    def register_players(self):
        """Register players (walkers) with the soccer ball."""
        # Use `iter_entities()` to access all entities in the environment
        for entity in self.env.task.iter_entities():
            # Check if the entity is a player (walker) using its type or attributes
            if hasattr(entity, 'walker_id') or entity.__class__.__name__.lower().startswith("walker"):
                self.soccer_ball.register_player(entity)

    def step(self, actions):
        """Step through the environment with given actions."""
        self.timestep = self.env.step(actions)
        return self.timestep

    def get_positions(self):
        """Get positions of entities (players and ball) in the environment."""
        positions = [self.env.physics.data.xpos[body_id] for body_id in range(self.env.physics.model.nbody)]
        return positions

    def get_physics(self):
        """Get the physics object from the environment."""
        return self.env.physics


class MultiplayerCamera:
    """Class to handle the camera settings and behavior."""

    def __init__(self):
        self.camera = camera.MultiplayerTrackingCamera(
            min_distance=3.0,
            distance_factor=1.5,
            smoothing_update_speed=0.1,
            azimuth=90,
            elevation=-30,
            width=640,
            height=480
        )

    def initialize(self, physics):
        """Initialize the camera with the environment's physics."""
        self.camera.after_compile(physics)
        initial_positions = [physics.data.xpos[body_id] for body_id in range(physics.model.nbody)]
        self.camera.initialize_episode(initial_positions)

    def update(self, positions):
        """Update the camera after each step."""
        self.camera.after_step(positions)

    def render(self):
        """Render the current frame using the tracking camera."""
        return self.camera.render()


class SoccerTraining:
    """Class to handle the training process, rewards, and recording."""

    def __init__(self):
        self.env = SoccerEnvironment()
        self.camera = MultiplayerCamera()
        self.video_writer = imageio.get_writer("soccer_training5.mp4", fps=30)
        self.frame_count = 0

    def calculate_distance_to_ball(self, player_position, ball_position):
        """Calculate the Euclidean distance between the player and the ball."""
        return np.linalg.norm(np.array(player_position) - np.array(ball_position))

    def calculate_rewards(self):
        """Calculate rewards based on player behaviors."""
        rewards = []
        physics = self.env.get_physics()
        
        # Access the ball position using the correct name
        ball_geom_name = 'soccer_ball/geom'  # Correct soccer ball geometry name
        try:
            ball_geom_id = physics.model.name2id(ball_geom_name, 'geom')
            ball_position = physics.data.xpos[ball_geom_id]
        except Exception as e:
            print(f"Error retrieving ball position: {e}")
            return rewards

        # List of player root names based on the environment
        player_root_names = ['home0/root', 'home1/root', 'away0/root', 'away1/root']  # Updated to match exact names

        for player_name in player_root_names:
            try:
                # Access the root body's position
                player_position = physics.named.data.xpos[player_name]

                # Calculate the distance to the soccer ball
                distance_to_ball = self.calculate_distance_to_ball(player_position, ball_position)

                # Reward for getting closer to the ball
                distance_reward = max(0, 1.0 - distance_to_ball)

                # Reward for kicking the ball
                kick_reward = 2.0 if self.env.soccer_ball.hit else 0

                # Access the linear velocity of the root body
                velocity = np.linalg.norm(physics.named.data.qvel[player_name])

                # Reward for moving (learning to walk/run)
                movement_reward = min(velocity, 1.0)

                # Total reward
                total_reward = distance_reward + kick_reward + movement_reward
                rewards.append(total_reward)

            except KeyError as e:
                print(f"Key error while accessing data for player '{player_name}': {e}")
                continue

        return rewards

    def print_rewards(self, rewards):
        """Print the rewards for each player."""
        for i, reward in enumerate(rewards):
            print(f"Player {i} reward: {reward:.2f}")

    def run_training(self):
        """Run the soccer training session."""
        self.camera.initialize(self.env.get_physics())

        while not self.env.timestep.last():
            # Generate random actions for all players
            actions = [
                np.random.uniform(action_spec.minimum, action_spec.maximum, size=action_spec.shape)
                for action_spec in self.env.action_specs
            ]

            # Step through the environment
            self.env.step(actions)

            # Update camera and calculate rewards
            entity_positions = self.env.get_positions()
            self.camera.update(entity_positions)
            rewards = self.calculate_rewards()

            # Print rewards and observations
            self.print_rewards(rewards)
            for i in range(len(self.env.action_specs)):
                print(f"Player {i}: observations = {self.env.timestep.observation[i]}")

            # Record frames at regular intervals
            if self.frame_count % 5 == 0:
                frame = self.camera.render()
                self.video_writer.append_data(frame)

            self.frame_count += 1

        self.video_writer.close()
        print("Training completed and video saved as soccer_training5.mp4.")


if __name__ == "__main__":
    # Run the soccer training
    training = SoccerTraining()
    training.run_training()

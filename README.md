Reinforcement Learning Project
This project uses Python, PyBullet, and Stable-Baselines3 for training an agent to learn various tasks.

Installation
Prerequisites
Python 3.10 (recommended to use Anaconda)
Visual Studio Code or another IDE
Setting Up the Environment
Install Anaconda:

Download and install Anaconda from here.
Create a New Conda Environment:

Open Anaconda Prompt and create a new environment:

lua
Copy code
conda create -n rl_env python=3.10
Activate the environment:

Copy code
conda activate rl_env
Installing Dependencies
Upgrade Pip and Setuptools:

css
Copy code
pip install --upgrade pip setuptools
Install PyTorch:

For the CPU version:

perl
Copy code
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
Install Stable-Baselines3:

Copy code
pip install stable-baselines3
Install Additional Dependencies:

rust
Copy code
pip install pybullet numpy
Usage
To run your training script, use the following command:

Copy code
python your_script.py
Replace your_script.py with the name of your Python file.

Stopping and Saving the Model
To manually stop and save the model:

Press Ctrl + C in the terminal to stop the training.

Make sure your script handles saving the model:

python
Copy code
model.save("path_to_save_model/model_name")
Load the model later with:

python
Copy code
model = PPO.load("path_to_save_model/model_name")
Troubleshooting
Common Issues
ModuleNotFoundError: Ensure all dependencies are installed in your activated environment.
Dependency Conflicts: If you face issues, try creating a new virtual environment and reinstalling the dependencies.

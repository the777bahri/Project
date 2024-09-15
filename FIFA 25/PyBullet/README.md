Reinforcement Learning Project

Installation
Prerequisites
• Python 3.10 (recommended to use Anaconda)
• Visual Studio Code or another IDE
Setting Up the Environment
1. Install Anaconda:
   • Download and install Anaconda from https://www.anaconda.com/products/distribution.

2. Create a New Conda Environment:
   Open Anaconda Prompt and create a new environment:

   conda create -n rl_env python=3.10

   Activate the environment:

   conda activate rl_env
Installing Dependencies
1. Upgrade Pip and Setuptools:

   pip install --upgrade pip setuptools

2. Install PyTorch:

   For the CPU version:

   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

3. Install Stable-Baselines3:

   pip install stable-baselines3

4. Install Additional Dependencies:

   pip install pybullet numpy
Usage
To run your training script, use the following command:

   python your_script.py

Replace your_script.py with the name of your Python file.
Stopping and Saving the Model
To manually stop and save the model:

1. Press Ctrl + C in the terminal to stop the training.
2. Make sure your script handles saving the model:

   model.save("path_to_save_model/model_name")

Load the model later with:

   model = PPO.load("path_to_save_model/model_name")
Troubleshooting
Common Issues
• ModuleNotFoundError: Ensure all dependencies are installed in your activated environment.
• Dependency Conflicts: If you face issues, try creating a new virtual environment and reinstalling the dependencies.

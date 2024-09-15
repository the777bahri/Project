import pybullet as p
import time
import pybullet_data
import numpy as np


class Simulation:
    def __init__(self) -> None:
        self.physics_client = p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())  
        
        self.env = Environment()
        self.agent = Agent()
        
        self.time_step = 1. / 240.
        
        self.loop()
        
    def degrees_to_radians(self, degrees):
        return degrees * (np.pi / 180)

    def loop(self):
        # User can input angles in degrees here
        shoulder_angle_degrees = 45  # 45 degrees
        elbow_angle_degrees = 90     # 90 degrees
        wrist_angle_degrees = 0      # 0 degrees (neutral position)

        # Convert the degrees to radians
        shoulder_angle = self.degrees_to_radians(shoulder_angle_degrees)
        elbow_angle = self.degrees_to_radians(elbow_angle_degrees)
        wrist_angle = self.degrees_to_radians(wrist_angle_degrees)

        while True: 
            p.stepSimulation()
            self.agent.moveRightArm(shoulder_angle, elbow_angle, wrist_angle)
            time.sleep(self.time_step)


class Environment:
    def __init__(self) -> None:
        self.loadEnvironment()
        self.placeBox()

    
    def loadEnvironment(self):
        p.setGravity(0, 0, -9.81)
        room_size = 5
        backgroundColor = [0.6, 0.4, 0.2, 1.0]

        floor = p.loadURDF("plane.urdf")
        p.changeVisualShape(floor, -1, rgbaColor=backgroundColor)
        
        back_wall = p.loadURDF("plane.urdf", [-room_size, -room_size, 0], p.getQuaternionFromEuler([0, np.pi/2, 0]))
        p.changeVisualShape(back_wall, -1, rgbaColor=backgroundColor)
        
        left_wall = p.loadURDF("plane.urdf", [-room_size, -room_size, 0], p.getQuaternionFromEuler([-np.pi/2, 0, 0]))
        p.changeVisualShape(left_wall, -1, rgbaColor=backgroundColor)
                
        right_wall = p.loadURDF("plane.urdf", [-room_size, room_size, 0], p.getQuaternionFromEuler([np.pi/2, 0, 0]))

    def placeBox(self):
        box_position = [2, 0, 1]  # 5 units forward in the x direction, same height as the agent
        box_orientation = [0, 0, 0]

        p.loadURDF("URDFs/box.urdf", box_position, p.getQuaternionFromEuler(box_orientation))
    

class Agent:
    def __init__(self) -> None:
        self.loadAgent(urdf_file="URDFs/human.urdf", start_pos=[0, 0, 1], start_orientation=[0, 0, 0])
        self.print_joint_info()

    def print_joint_info(self):
        num_joints = p.getNumJoints(self.humanoid)
        for joint_index in range(num_joints):
            joint_info = p.getJointInfo(self.humanoid, joint_index)
            print(f"Joint {joint_index}: {joint_info[1].decode('utf-8')}")

    def loadAgent(self, urdf_file, start_pos, start_orientation):
        self.humanoid = p.loadURDF(urdf_file, start_pos, p.getQuaternionFromEuler(start_orientation))
        self.joint_indices = range(p.getNumJoints(self.humanoid))

    def moveJoint(self, joint_index):
        for joint_index in self.joint_indices:
            p.setJointMotorControl2(self.humanoid, joint_index, p.TORQUE_CONTROL, force=np.random.uniform(-10, 10))
        p.stepSimulation()
    
    def moveRightArm(self, shoulder_angle, elbow_angle, wrist_angle):
        p.setJointMotorControl2(self.humanoid, 15, p.POSITION_CONTROL, targetPosition=shoulder_angle)
        p.setJointMotorControl2(self.humanoid, 16, p.POSITION_CONTROL, targetPosition=0)  # optional to adjust
        p.setJointMotorControl2(self.humanoid, 17, p.POSITION_CONTROL, targetPosition=0)  # optional to adjust

        p.setJointMotorControl2(self.humanoid, 18, p.POSITION_CONTROL, targetPosition=elbow_angle)
        p.setJointMotorControl2(self.humanoid, 19, p.POSITION_CONTROL, targetPosition=0)  # optional to adjust

        p.setJointMotorControl2(self.humanoid, 20, p.POSITION_CONTROL, targetPosition=wrist_angle)
        p.setJointMotorControl2(self.humanoid, 21, p.POSITION_CONTROL, targetPosition=0)  # optional to adjust

        p.stepSimulation()

    
if __name__ == "__main__":
    sim = Simulation()

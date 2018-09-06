#!/usr/bin/env python
import math
import socket
import time

import numpy as np
import rospy
import tf
from cv_bridge.core import CvBridge
from epuck.ePuck import ePuck
from geometry_msgs.msg import Point, Quaternion, Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image, Imu, Range
from visualization_msgs.msg import Marker
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import Adam

TCP_IP = '147.46.67.123'
TCP_PORT = 6666
BUFFER_SIZE = 1024

## Camera parameters
IMAGE_FORMAT = 'RGB_365'
CAMERA_ZOOM = 8

## Epuck dimensions
# Wheel Radio (cm)
WHEEL_DIAMETER = 4
# Separation between wheels (cm)
WHEEL_SEPARATION = 5.3

# Distance between wheels in meters (axis length); it's the same value as "WHEEL_SEPARATION" but expressed in meters.
WHEEL_DISTANCE = 0.053
# Wheel circumference (meters).
WHEEL_CIRCUMFERENCE = ((WHEEL_DIAMETER*math.pi)/100.0)
# Distance for each motor step (meters); a complete turn is 1000 steps.
MOT_STEP_DIST = (WHEEL_CIRCUMFERENCE/1000.0)    # 0.000125 meters per step (m/steps)

# available sensors
sensors = ['accelerometer', 'proximity', 'motor_position', 'light',
           'floor', 'camera', 'selector', 'motor_speed', 'microphone']

# Sight Range Boundary Radius
boundaryRadius = 30

# Goal Position of each Robots
goalPos = [[-5, -10], [-75, -10], [-75, 60], [-5, 60]] 
 
# Interfering Robot Number
obsNumber = 0

# Number of Main Robots
mainRobotNumber = 4

# State Size
state_size = 2

# Action Size
action_size = 9

# Radius
obstacleRadius = 10
agentRadius = 10

# mode = 0: Const, mode = 1: Linear, mode = 2: Quad
mode = 2

# A2C(Advantage Actor-Critic) agent
class A2CAgent:
    def __init__(self, state_size, action_size):
        self.load_model1 = True
        # self.load_model2 = True
        
        # get size of state and action
        self.state_size = state_size
        self.action_size = action_size
        self.value_size = 1

        # These are hyper parameters for the Policy Gradient
        self.discount_factor = 0.99
        self.actor_lr = 0.00002
        self.critic_lr = 0.00005

        # create model for policy network
        self.actor = self.build_actor()
        self.critic = self.build_critic()

        if self.load_model1:
            self.actor.load_weights("/home/howoongjun/RLPractice/Practice/Practice004_DataSave/Backup/Actor_Rev_171222_3600.h5")
            self.critic.load_weights("/home/howoongjun/RLPractice/Practice/Practice004_DataSave/Backup/Critic_Rev_171222_3600.h5")
            # self.actor.load_weights("/home/howoongjun/catkin_ws/src/simple_create/src/DataSave/backup/Actor_Rev_180112.h5")
            # self.critic.load_weights("/home/howoongjun/catkin_ws/src/simple_create/src/DataSave/backup/Critic_Rev_180112.h5")
            
    # approximate policy and value using Neural Network
    # actor: state is input and probability of each action is output of model
    def build_actor(self):
        actor = Sequential()
        actor.add(Dense(128, input_dim=self.state_size, activation='relu', kernel_initializer='glorot_normal'))
        actor.add(Dense(self.action_size, activation='softmax', kernel_initializer='glorot_normal'))
        actor.summary()
        # See note regarding crossentropy in cartpole_reinforce.py
        actor.compile(loss='categorical_crossentropy', optimizer=Adam(lr=self.actor_lr))
        return actor

    # critic: state is input and value of state is output of model
    def build_critic(self):
        critic = Sequential()
        critic.add(Dense(128, input_dim=self.state_size, activation='relu', kernel_initializer='glorot_normal'))
        critic.add(Dense(self.value_size, activation='linear', kernel_initializer='glorot_normal'))
        critic.summary()
        critic.compile(loss="mse", optimizer=Adam(lr=self.critic_lr))
        return critic

    # using the output of policy network, pick action stochastically
    def get_action(self, state):
        policy = self.actor.predict(state, batch_size=1).flatten()
        return policy

    # update policy network every episode
    def train_model(self, state, action, reward, next_state, done):
        target = np.zeros((1, self.value_size))
        advantages = np.zeros((1, self.action_size))

        value = self.critic.predict(state)[0]
        next_value = self.critic.predict(next_state)[0]

        if done:
            advantages[0][action] = reward - value
            target[0][0] = reward
        else:
            advantages[0][action] = reward + self.discount_factor * (next_value) - value
            target[0][0] = reward + self.discount_factor * next_value

        self.actor.fit(state, advantages, epochs=1, verbose=0)
        self.critic.fit(state, target, epochs=1, verbose=0)

def stateGenerator(obsPosition, agtPosition, idx):
    returnSum = []
    if idx != -1:
        returnSum = returnSum + [float(agtPosition[0]) - float(obsPosition[idx][0]), float(agtPosition[1]) - float(obsPosition[idx][1])]
    else:
        returnSum = returnSum + [float(agtPosition[0]) - float(obsPosition[0]), float(agtPosition[1]) - float(obsPosition[1])]
    returnSum = np.reshape(returnSum, [1, state_size])
    return returnSum

def rangeFinder(allObsPos, rangeCenter):
    countObs = 0
    rangeObstacle = [[0,0] for _ in range(obsNumber + mainRobotNumber - 1)]
    allObsAgtDistance = [0 for _ in range(obsNumber + mainRobotNumber - 1)]
    for i in range(0, obsNumber + mainRobotNumber - 1):
        allObsAgtDistance[i] = math.sqrt((float(allObsPos[i][0]) - float(rangeCenter[0]))**2 + (float(allObsPos[i][1]) - float(rangeCenter[1]))**2)
        if math.sqrt((float(rangeCenter[0]) - float(allObsPos[i][0]))**2 + (float(rangeCenter[1]) - float(allObsPos[i][1]))**2) < boundaryRadius:
            rangeObstacle[countObs] = [float(allObsPos[i][0]), float(allObsPos[i][1])]
            countObs += 1

    index = np.argmin(allObsAgtDistance)
    return [countObs, rangeObstacle, index]

def goalFinder(idx, agtPos):
    goalAngle = 0
    if goalPos[idx][0] == float(agtPos[0]):
        if goalPos[idx][1] > float(agtPos[1]):
            goalAngle = 90 * math.pi / 180
        else:
            goalAngle = -90 * math.pi / 180
    else:
        goalAngle = math.atan(1.0*(goalPos[idx][1] - float(agtPos[1]))/(goalPos[idx][0] - float(agtPos[0])))
    if goalPos[idx][0] < float(agtPos[0]):
        goalAngle += math.pi
        
    tmpGoal = [0,0]
    tmpGoal[0] = float(agtPos[0]) + boundaryRadius * math.cos(goalAngle)
    tmpGoal[1] = float(agtPos[1]) + boundaryRadius * math.sin(goalAngle)
    return tmpGoal

def action2degree(action):
    degree = 0
    if action == 0:
        degree = 0
    elif action == 1:
        degree = math.pi / 4
    elif action == 2:
        degree = math.pi / 2
    elif action == 3:
        degree = math.pi * 3 / 4
    elif action == 4:
        degree = math.pi
    elif action == 5:
        degree  = -3 * math.pi / 4
    elif action == 6:
        degree = -1 * math.pi / 2
    elif action == 7:
        degree = -1 * math.pi / 4

    return degree

def takeAction(desiredHeading, robotYaw):
    linearX = 0
    angularZ = 0
    angularVelocityCalibration = 1.0
    maxSpeed = 3.0

    # Just for correcting my mistakes done by learning process
    if desiredHeading == 2:
        desiredHeading = 7
    elif desiredHeading == 7:
        desiredHeading = 2
    
    # Match Coordinates
    if robotYaw > math.pi:
        robotYaw = robotYaw - math.pi * 2

    # Convert Action into Desired Degree
    desiredDegree = action2degree(desiredHeading)

    # If desired degree matches with the current robot orientation, go max speed
    if desiredDegree == robotYaw:
        linearX = maxSpeed
        angularZ = 0
    else:
        angularDiff = robotYaw - desiredDegree
        if angularDiff > math.pi:
            angularDiff = angularDiff - math.pi * 2
        elif angularDiff < -math.pi:
            angularDiff = angularDiff + math.pi * 2
        delimeter = 2
        if mode == 1:
            delimeter = 2
            if angularDiff >= 0:
                linearX = -2 * maxSpeed / math.pi * angularDiff + maxSpeed
            elif angularDiff < 0:
                linearX = 2 * maxSpeed / math.pi * angularDiff + maxSpeed
        elif mode == 2:
            delimeter = math.sqrt(2)
            linearX = -2 * maxSpeed / (math.pi * math.pi) * (angularDiff * angularDiff) + maxSpeed
        
        if abs(angularDiff) == math.pi:
            if mode == 0:
                linearX = -maxSpeed
            angularZ = 0
        elif abs(angularDiff) <= math.pi / delimeter:
            if mode == 0:
                linearX = maxSpeed
            angularZ = angularDiff * angularVelocityCalibration
        elif angularDiff > math.pi / delimeter:
            if mode == 0:
                linearX = -maxSpeed
            angularZ = (angularDiff - math.pi) * angularVelocityCalibration
        elif angularDiff < -1 * math.pi / delimeter:
            if mode == 0:
                linearX = -maxSpeed
            angularZ = (angularDiff + math.pi) * angularVelocityCalibration
        if desiredHeading == 8:
            linearX = 0
            angularZ = 0
    return [linearX, angularZ]



class EPuckDriver(object):
    """
    :param epuck_name:
    :param epuck_address:
    """
    
    def __init__(self, epuck_name, epuck_address, init_xpos, init_ypos, init_theta):
        self._bridge = ePuck(epuck_address, False)
        self._name = epuck_name
        
        self.enabled_sensors = {s: None for s in sensors}

        self.prox_publisher = []
        self.prox_msg = []

        self.theta = init_theta
        self.x_pos = init_xpos
        self.y_pos = init_ypos
        self.leftStepsPrev = 0
        self.rightStepsPrev = 0
        self.leftStepsDiff = 0
        self.rightStepsDiff = 0
        self.deltaSteps = 0
        self.deltaTheta = 0
        self.startTime = time.time()
        self.endTime = time.time()
        self.br = tf.TransformBroadcaster()

    def greeting(self):
        """
        Hello by robot.
        """
        self._bridge.set_body_led(1)
        self._bridge.set_front_led(1)
        rospy.sleep(0.5)
        self._bridge.set_body_led(0)
        self._bridge.set_front_led(0)

    def disconnect(self):
        """
        Close bluetooth connection
        """
        self._bridge.close()

    def setup_sensors(self):
        """
        Enable epuck sensors based on the parameters.
        By default, all sensors are false.

        """
        # get parameters to enable sensors
        for sensor in sensors:
            self.enabled_sensors[sensor] = rospy.get_param('~' + sensor, False)
            #print rospy.get_param('~' + sensor, False)

        # Only enabled sensors
        enable = [s for s, en in self.enabled_sensors.items() if en]

        # Enable the right sensors
        self._bridge.enable(*enable)

        if self.enabled_sensors['camera']:
            self._bridge.set_camera_parameters(IMAGE_FORMAT, 40, 40, CAMERA_ZOOM)

        #if self.enabled_sensors['proximity']:
            #self._bridge.calibrate_proximity_sensors()

    def run(self):
        # Connect to the ePuck
        self._bridge.connect()

        # Setup the necessary sensors.
        self.setup_sensors()

        # Disconnect when rospy is going to down
        rospy.on_shutdown(self.disconnect)
        
        self.greeting()

        self._bridge.step()

        # Actor Critic Agent
        agent = A2CAgent(state_size, action_size)

        # Subscribe to Commando Velocity Topic
        rospy.Subscriber("mobile_base/cmd_vel", Twist, self.handler_velocity)

        # Sensor Publishers
        # rospy.Publisher("/%s/mobile_base/" % self._name, )

        if self.enabled_sensors['camera']:
            self.image_publisher = rospy.Publisher("camera", Image)

        if self.enabled_sensors['proximity']:
            for i in range(0,8):
                self.prox_publisher.append(rospy.Publisher("proximity"+str(i), Range))
                self.prox_msg.append(Range())
                self.prox_msg[i].radiation_type = Range.INFRARED
                self.prox_msg[i].header.frame_id =  self._name+"/base_prox" + str(i)
                self.prox_msg[i].field_of_view = 0.26 	# About 15 degrees...to be checked!
                self.prox_msg[i].min_range = 0.005	# 0.5 cm
                self.prox_msg[i].max_range = 0.05		# 5 cm

        if self.enabled_sensors['motor_position']:
            self.odom_publisher = rospy.Publisher('odom', Odometry)

        if self.enabled_sensors['accelerometer']:
            self.accel_publisher = rospy.Publisher('accel', Imu)    # Only "linear_acceleration" vector filled.

        if self.enabled_sensors['selector']:
            self.selector_publisher = rospy.Publisher('selector', Marker)

        if self.enabled_sensors['light']:
            self.light_publisher = rospy.Publisher('light', Marker)

        if self.enabled_sensors['motor_speed']:
            self.motor_speed_publisher = rospy.Publisher('motor_speed', Marker)

        if self.enabled_sensors['microphone']:
            self.microphone_publisher = rospy.Publisher('microphone', Marker)

        if self.enabled_sensors['floor']:
            self.floor_publisher = rospy.Publisher('floor', Marker)

        posMainRobot_pub = []
        posMainRobot_msg = []

        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect((TCP_IP, TCP_PORT))
        
        print "Connected!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"

        for i in range(0, mainRobotNumber):
            posMainRobot_pub = posMainRobot_pub + [rospy.Publisher('/epuck_robot_' + str(i) + '/mobile_base/cmd_vel', Twist, queue_size = 10)]
            posMainRobot_msg.append(Twist())

        # Initialize goalReached flag
        goalReached = []
        for i in range(0, mainRobotNumber):
            goalReached = goalReached + [False]

        # Spin almost forever
        #rate = rospy.Rate(7)   # 7 Hz. If you experience "timeout" problems with multiple robots try to reduce this value.
        self.startTime = time.time()
        while not rospy.is_shutdown():

            # Socket Communication
            data = s.recv(BUFFER_SIZE)
            s.send("OK")

            dataList = data.split(" ")
            # print data
            mainRobotCoordinates = []
            # print dataList
            for i in range(0, mainRobotNumber):
                mainRobotCoordinates = mainRobotCoordinates + [[dataList[4 * i], dataList[4 * i + 1], dataList[4 * i + 2]]]
                
            self._bridge.step()
            self.update_sensors()


            # For Test
            # for curRobNo in range(0, mainRobotNumber):
            #     posMainRobot_msg[curRobNo].linear.x = 0.0
            #     posMainRobot_msg[curRobNo].angular.z = 0.0
            #     posMainRobot_pub[curRobNo].publish(posMainRobot_msg[curRobNo])
                
            # Obstacle Robot Coordinates. In the point of current robot, other main robots are regarded as obstacles
            obst_coordinates = []
            interfRobot_coordinates = []

            for curRobNo in range(0, mainRobotNumber):
                tmpAction = []

                obst_coordinates = interfRobot_coordinates

                for i in range(0, mainRobotNumber):
                    if i != curRobNo:
                        obst_coordinates = obst_coordinates + [mainRobotCoordinates[i]]
                
                [rangeObsNumber, rangeObsPos, _] = rangeFinder(obst_coordinates, mainRobotCoordinates[curRobNo])

                for i in range(0, rangeObsNumber):
                    state = stateGenerator(rangeObsPos, mainRobotCoordinates[curRobNo], i)
                    policyArr = agent.get_action(state)
                    if i == 0:
                        tmpAction = (1 - policyArr)
                    else:
                        tmpAction = tmpAction * (1 - policyArr)

                if tmpAction != []:
                    for j in range(0,action_size):
                        if tmpAction[j] > 0.999:
                            tmpAction[j] = 1
                        elif tmpAction[j] > 0.995:
                            tmpAction[j] = 0.1
                        else:
                            tmpAction[j] = 0
                    tmpArgMax = np.argmax(tmpAction)
                
                if rangeObsNumber == 0:
                    tmpAction = [1.0/action_size for _ in range(0,action_size)]
                tmpGoalPos = goalFinder(curRobNo, mainRobotCoordinates[curRobNo])
                state = stateGenerator(tmpGoalPos, mainRobotCoordinates[curRobNo], -1)
                policyArr = agent.get_action(state)
                if np.mean(tmpAction) == 0:
                    rospy.logwarn("No Action Selected! Random Action")
                    tmpAction[tmpArgMax] = 1

                tmpAction = tmpAction * np.asarray(policyArr)

                tmpAction = tmpAction / np.sum(tmpAction)
                action = np.random.choice(action_size, 1, p = tmpAction)[0]

                linearX = 0
                angularZ = 0
                [linearX, angularZ] = takeAction(action, float(mainRobotCoordinates[curRobNo][2]) * math.pi / 180)

                if(math.sqrt((float(mainRobotCoordinates[curRobNo][0]) - goalPos[curRobNo][0])**2 + (float(mainRobotCoordinates[curRobNo][1]) - goalPos[curRobNo][1])**2) <= agentRadius):
                    if goalReached[curRobNo] == False:
                        print str(curRobNo) + "Robot has reached the goal !"
                    goalReached[curRobNo] = True
                # print "1th: " + str(mainRobotCoordinates[1])
                # print str(mainRobotCoordinates)
                # print "2nd: " + str(mainRobotCoordinates[2])

                posMainRobot_msg[curRobNo].linear.x = linearX
                posMainRobot_msg[curRobNo].angular.z = -angularZ
                
                if dataList[curRobNo * 4 + 3] < 80:
                    posMainRobot_msg[curRobNo].linear.x = 0
                    posMainRobot_msg[curRobNo].angular.z = 0
                    print "Not reliable data"

                if goalReached[curRobNo]:
                    posMainRobot_msg[curRobNo].linear.x = 0
                    posMainRobot_msg[curRobNo].angular.z = 0
                
                # posMainRobot_msg[3].linear.x = 0
                # posMainRobot_msg[3].angular.z = 0
                # posMainRobot_msg[1].linear.x = 0
                # posMainRobot_msg[1].angular.z = 0
                posMainRobot_pub[curRobNo].publish(posMainRobot_msg[curRobNo])
            #     print str(curRobNo) + ": " + str(posMainRobot_msg[curRobNo])
            #     print str(goalPos[curRobNo]) + "," + str(rangeObsNumber)
            #     print str(mainRobotCoordinates[curRobNo])
                # print "ACTION: " + str(action)
                # print str(curRobNo) + ": " + str(rangeObsNumber)
            # print "========================================================================="
            tmpCount = 1
            for i in range(0, mainRobotNumber):
                tmpCount = tmpCount * goalReached[i]
            
            if tmpCount == 1:
                tmpGoal = goalPos[1]
                goalPos[1] = goalPos[3]
                goalPos[3] = tmpGoal
                tmpGoal = goalPos[2]
                goalPos[2] = goalPos[0]
                goalPos[0] = tmpGoal
                print "Goal Point Changed to " + str(goalPos)
                for i in range(0, mainRobotNumber):
                    goalReached[i] = False
        s.close()

            #rate.sleep()	# Do not call "sleep" otherwise the bluetooth communication will hang.
                            # We communicate as fast as possible, this shouldn't be a problem...

    def update_sensors(self):
        # print "accelerometer:", self._bridge.get_accelerometer()
        # print "proximity:", self._bridge.get_proximity()
        # print "light:", self._bridge.get_light_sensor()
        # print "motor_position:", self._bridge.get_motor_position()
        # print "floor:", self._bridge.get_floor_sensors()
        # print "image:", self._bridge.get_image()

        ## If image from camera
        if self.enabled_sensors['camera']:
            # Get Image
            image = self._bridge.get_image()
            #print image
            if image is not None:
                nimage = np.asarray(image)
                image_msg = CvBridge().cv2_to_imgmsg(nimage, "rgb8")
                self.image_publisher.publish(image_msg)

        if self.enabled_sensors['proximity']:
            prox_sensors = self._bridge.get_proximity()
            for i in range(0,8):
                if prox_sensors[i] > 0:
                    self.prox_msg[i].range = 0.5/math.sqrt(prox_sensors[i])	# Transform the analog value to a distance value in meters (given from field tests).
                else:
                    self.prox_msg[i].range = self.prox_msg[i].max_range

                if self.prox_msg[i].range > self.prox_msg[i].max_range:
                    self.prox_msg[i].range = self.prox_msg[i].max_range
                if self.prox_msg[i].range < self.prox_msg[i].min_range:
                    self.prox_msg[i].range = self.prox_msg[i].min_range
                self.prox_msg[i].header.stamp = rospy.Time.now()
                self.prox_publisher[i].publish(self.prox_msg[i])


            # e-puck proximity positions (cm), x pointing forward, y pointing left
            #           P7(3.5, 1.0)   P0(3.5, -1.0)
            #       P6(2.5, 2.5)           P1(2.5, -2.5)
            #   P5(0.0, 3.0)                   P2(0.0, -3.0)
            #       P4(-3.5, 2.0)          P3(-3.5, -2.0)
            #
            # e-puck proximity orentations (degrees)
            #           P7(10)   P0(350)
            #       P6(40)           P1(320)
            #   P5(90)                   P2(270)
            #       P4(160)          P3(200)
            self.br.sendTransform((0.035, -0.010, 0.034), tf.transformations.quaternion_from_euler(0, 0, 6.11), rospy.Time.now(), self._name+"/base_prox0", self._name+"/base_link")
            self.br.sendTransform((0.025, -0.025, 0.034), tf.transformations.quaternion_from_euler(0, 0, 5.59), rospy.Time.now(), self._name+"/base_prox1", self._name+"/base_link")
            self.br.sendTransform((0.000, -0.030, 0.034), tf.transformations.quaternion_from_euler(0, 0, 4.71), rospy.Time.now(), self._name+"/base_prox2", self._name+"/base_link")
            self.br.sendTransform((-0.035, -0.020, 0.034), tf.transformations.quaternion_from_euler(0, 0, 3.49), rospy.Time.now(), self._name+"/base_prox3", self._name+"/base_link")
            self.br.sendTransform((-0.035, 0.020, 0.034), tf.transformations.quaternion_from_euler(0, 0, 2.8), rospy.Time.now(), self._name+"/base_prox4", self._name+"/base_link")
            self.br.sendTransform((0.000, 0.030, 0.034), tf.transformations.quaternion_from_euler(0, 0, 1.57), rospy.Time.now(), self._name+"/base_prox5", self._name+"/base_link")
            self.br.sendTransform((0.025, 0.025, 0.034), tf.transformations.quaternion_from_euler(0, 0, 0.70), rospy.Time.now(), self._name+"/base_prox6", self._name+"/base_link")
            self.br.sendTransform((0.035, 0.010, 0.034), tf.transformations.quaternion_from_euler(0, 0, 0.17), rospy.Time.now(), self._name+"/base_prox7", self._name+"/base_link")


        if self.enabled_sensors['accelerometer']:
            accel = self._bridge.get_accelerometer()
            accel_msg = Imu()
            accel_msg.header.stamp = rospy.Time.now()
            accel_msg.header.frame_id = self._name+"/base_link"
            accel_msg.linear_acceleration.x = (accel[1]-2048.0)/800.0*9.81 # 1 g = about 800, then transforms in m/s^2.
            accel_msg.linear_acceleration.y = (accel[0]-2048.0)/800.0*9.81
            accel_msg.linear_acceleration.z = (accel[2]-2048.0)/800.0*9.81
            accel_msg.linear_acceleration_covariance = [0.01,0.0,0.0, 0.0,0.01,0.0, 0.0,0.0,0.01]
            #print "accel raw: " + str(accel[0]) + ", " + str(accel[1]) + ", " + str(accel[2])
            #print "accel (m/s2): " + str((accel[0]-2048.0)/800.0*9.81) + ", " + str((accel[1]-2048.0)/800.0*9.81) + ", " + str((accel[2]-2048.0)/800.0*9.81)
            accel_msg.angular_velocity.x = 0
            accel_msg.angular_velocity.y = 0
            accel_msg.angular_velocity.z = 0
            accel_msg.angular_velocity_covariance = [0.01,0.0,0.0, 0.0,0.01,0.0, 0.0,0.0,0.01]
            q = tf.transformations.quaternion_from_euler(0, 0, 0)
            accel_msg.orientation = Quaternion(*q)
            accel_msg.orientation_covariance = [0.01,0.0,0.0, 0.0,0.01,0.0, 0.0,0.0,0.01]
            self.accel_publisher.publish(accel_msg)

        if self.enabled_sensors['motor_position']:
            motor_pos = list(self._bridge.get_motor_position()) # Get a list since tuple elements returned by the function are immutable.
            # Convert to 16 bits signed integer.
            if(motor_pos[0] & 0x8000):
                motor_pos[0] = -0x10000 + motor_pos[0]
            if(motor_pos[1] & 0x8000):
                motor_pos[1] = -0x10000 + motor_pos[1]
            #print "motor_pos: " + str(motor_pos[0]) + ", " + str(motor_pos[1])

            self.leftStepsDiff = motor_pos[0]*MOT_STEP_DIST - self.leftStepsPrev    # Expressed in meters.
            self.rightStepsDiff = motor_pos[1]*MOT_STEP_DIST - self.rightStepsPrev  # Expressed in meters.
            #print "left, right steps diff: " + str(self.leftStepsDiff) + ", " + str(self.rightStepsDiff)

            self.deltaTheta = (self.rightStepsDiff - self.leftStepsDiff)/WHEEL_DISTANCE # Expressed in radiant.
            self.deltaSteps = (self.rightStepsDiff + self.leftStepsDiff)/2  # Expressed in meters.
            #print "delta theta, steps: " + str(self.deltaTheta) + ", " + str(self.deltaSteps)

            self.x_pos += self.deltaSteps*math.cos(self.theta + self.deltaTheta/2)  # Expressed in meters.
            self.y_pos += self.deltaSteps*math.sin(self.theta + self.deltaTheta/2)  # Expressed in meters.
            self.theta += self.deltaTheta   # Expressed in radiant.
            #print "x, y, theta: " + str(self.x_pos) + ", " + str(self.y_pos) + ", " + str(self.theta)

            self.leftStepsPrev = motor_pos[0]*MOT_STEP_DIST  # Expressed in meters.
            self.rightStepsPrev = motor_pos[1]*MOT_STEP_DIST    # Expressed in meters.

            odom_msg = Odometry()
            odom_msg.header.stamp = rospy.Time.now()
            odom_msg.header.frame_id = "odom"
            odom_msg.child_frame_id = self._name+"/base_link"
            odom_msg.pose.pose.position = Point(self.x_pos, self.y_pos, 0)
            q = tf.transformations.quaternion_from_euler(0, 0, self.theta)
            odom_msg.pose.pose.orientation = Quaternion(*q)
            self.endTime = time.time()
            odom_msg.twist.twist.linear.x = self.deltaSteps / (self.endTime-self.startTime) # self.deltaSteps is the linear distance covered in meters from the last update (delta distance);
                                                                                            # the time from the last update is measured in seconds thus to get m/s we multiply them.
            odom_msg.twist.twist.angular.z = self.deltaTheta / (self.endTime-self.startTime)    # self.deltaTheta is the angular distance covered in radiant from the last update (delta angle);
                                                                                                # the time from the last update is measured in seconds thus to get rad/s we multiply them.
            #print "time elapsed = " + str(self.endTime-self.startTime) + " seconds"
            self.startTime = self.endTime

            self.odom_publisher.publish(odom_msg)
            pos = (odom_msg.pose.pose.position.x, odom_msg.pose.pose.position.y, odom_msg.pose.pose.position.z)
            ori = (odom_msg.pose.pose.orientation.x, odom_msg.pose.pose.orientation.y, odom_msg.pose.pose.orientation.z, odom_msg.pose.pose.orientation.w)
            self.br.sendTransform(pos, ori, odom_msg.header.stamp, odom_msg.child_frame_id, odom_msg.header.frame_id)

        if self.enabled_sensors['light']:
            light_sensors = self._bridge.get_light_sensor()
            light_sensors_marker_msg = Marker()
            light_sensors_marker_msg.header.frame_id = self._name+"/base_link"
            light_sensors_marker_msg.header.stamp = rospy.Time.now()
            light_sensors_marker_msg.type = Marker.TEXT_VIEW_FACING
            light_sensors_marker_msg.pose.position.x = 0.15
            light_sensors_marker_msg.pose.position.y = 0
            light_sensors_marker_msg.pose.position.z = 0.15
            q = tf.transformations.quaternion_from_euler(0, 0, 0)
            light_sensors_marker_msg.pose.orientation = Quaternion(*q)
            light_sensors_marker_msg.scale.z = 0.01
            light_sensors_marker_msg.color.a = 1.0
            light_sensors_marker_msg.color.r = 1.0
            light_sensors_marker_msg.color.g = 1.0
            light_sensors_marker_msg.color.b = 1.0
            light_sensors_marker_msg.text = "light: " + str(light_sensors)
            self.light_publisher.publish(light_sensors_marker_msg)

        if self.enabled_sensors['floor']:
            floor_sensors = self._bridge.get_floor_sensors()
            floor_marker_msg = Marker()
            floor_marker_msg.header.frame_id = self._name+"/base_link"
            floor_marker_msg.header.stamp = rospy.Time.now()
            floor_marker_msg.type = Marker.TEXT_VIEW_FACING
            floor_marker_msg.pose.position.x = 0.15
            floor_marker_msg.pose.position.y = 0
            floor_marker_msg.pose.position.z = 0.13
            q = tf.transformations.quaternion_from_euler(0, 0, 0)
            floor_marker_msg.pose.orientation = Quaternion(*q)
            floor_marker_msg.scale.z = 0.01
            floor_marker_msg.color.a = 1.0
            floor_marker_msg.color.r = 1.0
            floor_marker_msg.color.g = 1.0
            floor_marker_msg.color.b = 1.0
            floor_marker_msg.text = "floor: " + str(floor_sensors)
            self.floor_publisher.publish(floor_marker_msg)

        if self.enabled_sensors['selector']:
            curr_sel = self._bridge.get_selector()
            selector_marker_msg = Marker()
            selector_marker_msg.header.frame_id = self._name+"/base_link"
            selector_marker_msg.header.stamp = rospy.Time.now()
            selector_marker_msg.type = Marker.TEXT_VIEW_FACING
            selector_marker_msg.pose.position.x = 0.15
            selector_marker_msg.pose.position.y = 0
            selector_marker_msg.pose.position.z = 0.11
            q = tf.transformations.quaternion_from_euler(0, 0, 0)
            selector_marker_msg.pose.orientation = Quaternion(*q)
            selector_marker_msg.scale.z = 0.01
            selector_marker_msg.color.a = 1.0
            selector_marker_msg.color.r = 1.0
            selector_marker_msg.color.g = 1.0
            selector_marker_msg.color.b = 1.0
            selector_marker_msg.text = "selector: " + str(curr_sel)
            self.selector_publisher.publish(selector_marker_msg)

        if self.enabled_sensors['motor_speed']:
            motor_speed = list(self._bridge.get_motor_speed()) # Get a list since tuple elements returned by the function are immutable.
            # Convert to 16 bits signed integer.
            if(motor_speed[0] & 0x8000):
                motor_speed[0] = -0x10000 + motor_speed[0]
            if(motor_speed[1] & 0x8000):
                motor_speed[1] = -0x10000 + motor_speed[1]
            motor_speed_marker_msg = Marker()
            motor_speed_marker_msg.header.frame_id = self._name+"/base_link"
            motor_speed_marker_msg.header.stamp = rospy.Time.now()
            motor_speed_marker_msg.type = Marker.TEXT_VIEW_FACING
            motor_speed_marker_msg.pose.position.x = 0.15
            motor_speed_marker_msg.pose.position.y = 0
            motor_speed_marker_msg.pose.position.z = 0.09
            q = tf.transformations.quaternion_from_euler(0, 0, 0)
            motor_speed_marker_msg.pose.orientation = Quaternion(*q)
            motor_speed_marker_msg.scale.z = 0.01
            motor_speed_marker_msg.color.a = 1.0
            motor_speed_marker_msg.color.r = 1.0
            motor_speed_marker_msg.color.g = 1.0
            motor_speed_marker_msg.color.b = 1.0
            motor_speed_marker_msg.text = "speed: " + str(motor_speed)
            self.motor_speed_publisher.publish(motor_speed_marker_msg)

        if self.enabled_sensors['microphone']:
            mic = self._bridge.get_microphone()
            microphone_marker_msg = Marker()
            microphone_marker_msg.header.frame_id = self._name+"/base_link"
            microphone_marker_msg.header.stamp = rospy.Time.now()
            microphone_marker_msg.type = Marker.TEXT_VIEW_FACING
            microphone_marker_msg.pose.position.x = 0.15
            microphone_marker_msg.pose.position.y = 0
            microphone_marker_msg.pose.position.z = 0.07
            q = tf.transformations.quaternion_from_euler(0, 0, 0)
            microphone_marker_msg.pose.orientation = Quaternion(*q)
            microphone_marker_msg.scale.z = 0.01
            microphone_marker_msg.color.a = 1.0
            microphone_marker_msg.color.r = 1.0
            microphone_marker_msg.color.g = 1.0
            microphone_marker_msg.color.b = 1.0
            microphone_marker_msg.text = "microphone: " + str(mic)
            self.microphone_publisher.publish(microphone_marker_msg)


    def handler_velocity(self, data):
        """
        Controls the velocity of each wheel based on linear and angular velocities.
        :param data:
        """
        linear = data.linear.x
        angular = data.angular.z

        # Kinematic model for differential robot.
        wl = (linear - (WHEEL_SEPARATION / 2.) * angular) / WHEEL_DIAMETER
        wr = (linear + (WHEEL_SEPARATION / 2.) * angular) / WHEEL_DIAMETER

        # At input 1000, angular velocity is 1 cycle / s or  2*pi/s.
        left_vel = wl * 1000.
        right_vel = wr * 1000.
        self._bridge.set_motors_speed(left_vel, right_vel)


def run():
    rospy.init_node("epuck_drive", anonymous=True)
    epuck_address = rospy.get_param("~epuck_address")
    epuck_name = rospy.get_param("~epuck_name", "epuck")
    init_xpos = rospy.get_param("~xpos", 0.0)
    init_ypos = rospy.get_param("~ypos", 0.0)
    init_theta = rospy.get_param("~theta", 0.0)
    #print "init x, y, th: " + str(init_xpos) + ", " + str(init_ypos) + ", " + str(init_theta)

    EPuckDriver(epuck_name, epuck_address, init_xpos, init_ypos, init_theta).run()


if __name__ == "__main__":
    run()

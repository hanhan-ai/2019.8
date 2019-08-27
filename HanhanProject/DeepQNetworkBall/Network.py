"""
2019.8.25 上传
赵士陆

"""
#   My first try on DeepQNetwork
#   God bless you my little brain.
#   2019.8.22 - 2019.8.25


# Import libs
import numpy as np
import tensorflow as tf
import cv2
import random
# import pygame
from collections import *
import BikeGame.ai_action as ai_act
from Interaction.start import game_convertion as gc
# import matplotlib.pyplot as plt
import matplotlib.pyplot as plt


"""
# Define basic parameters of the simple test game
GAME = 'My DQN ball catch'
BLACK = (0, 0, 0)               # colors
WHITE = (255, 255, 255)
SCREEN_SIZE = [320, 400]        # screen
BAR_SIZE = [50, 5]              # what we control
BALL_SIZE = [15, 15]            # what we should catch

# Define the output of the little brain (which is how to control the movement of the bar)
MOVE_STAY = [1, 0, 0]
MOVE_LEFT = [0, 1, 0]
MOVE_RIGHT = [0, 0, 1]

# /----------------------------------GAME PART--------------------------------------------/
# this is a simple game to test the DQN

class Game (object):                                                # Create a game environment
    def __init__(self):
        pygame.init()
        self.clock = pygame.time.Clock()                            # clock to refresh the screen
        self.screen = pygame.display.set_mode(SCREEN_SIZE)          # create a screen

        pygame.display.set_caption('Simple Game')                   # set a caption

        self.ball_pos_x = SCREEN_SIZE[0] // 2 - BALL_SIZE[0] / 2    # define the position of the ball
        self.ball_pos_y = SCREEN_SIZE[1] // 2 - BALL_SIZE[1] / 2
        self.ball_dir_x = -1  # -1 = left 1 = right                 # define the move direction of the ball
        self.ball_dir_y = -1  # -1 = up   1 = down
        self.ball_pos = pygame.Rect(self.ball_pos_x, self.ball_pos_y, BALL_SIZE[0], BALL_SIZE[1])
        self.bar_pos_x = SCREEN_SIZE[0] // 2 - BAR_SIZE[0] // 2     # ?????
        self.bar_pos = pygame.Rect(self.bar_pos_x, SCREEN_SIZE[1] - BAR_SIZE[1], BAR_SIZE[0], BAR_SIZE[1])

    # actions are MOVE_STAY、MOVE_LEFT、MOVE_RIGHT
    # My little brain controls the movement of the bar
    # This function will return the screen pixel and the reward of the game
    # Based on the screen pixel we judge if the bar catched the ball or not
    # And give the relevant reward

    def step(self, action):
        if all(action == MOVE_LEFT):
            self.bar_pos_x = self.bar_pos_x - 2                     # bar move
        elif all(action == MOVE_RIGHT):
            self.bar_pos_x = self.bar_pos_x + 2
        else:
            pass

        # the bar can not move when it touch the edge of the screen
        if self.bar_pos_x < 0:
            self.bar_pos_x = 0
        if self.bar_pos_x > SCREEN_SIZE[0] - BAR_SIZE[0]:
            self.bar_pos_x = SCREEN_SIZE[0] - BAR_SIZE[0]

        self.screen.fill(BLACK)                                     # the color of the screen, BLACK is fine.

        # Draw the bar
        self.bar_pos.left = self.bar_pos_x
        pygame.draw.rect(self.screen, WHITE, self.bar_pos)
        # the ball move
        self.ball_pos.left += self.ball_dir_x * 2
        self.ball_pos.bottom += self.ball_dir_y * 3
        # draw the ball
        pygame.draw.rect(self.screen, WHITE, self.ball_pos)

        # if the ball touch the edge of the screen, it rebound
        if self.ball_pos.top <= 0 or self.ball_pos.bottom >= (SCREEN_SIZE[1] - BAR_SIZE[1] + 1):
            self.ball_dir_y = self.ball_dir_y * -1
        if self.ball_pos.left <= 0 or self.ball_pos.right >= (SCREEN_SIZE[0]):
            self.ball_dir_x = self.ball_dir_x * -1
        # And it's a normal move, no reward or punishment will be given to my little brain.
        reward = 0

        # !!-----------REWARD GIVEN------------!!
        if self.bar_pos.top <= self.ball_pos.bottom and (
                self.bar_pos.left < self.ball_pos.right and self.bar_pos.right > self.ball_pos.left):

            reward = 1  # REWARD

        elif self.bar_pos.top <= self.ball_pos.bottom and (
                self.bar_pos.left > self.ball_pos.right or self.bar_pos.right < self.ball_pos.left):

            reward = -1  # PUNISHMENT

        # screen shoot
        screen_image = pygame.surfarray.array3d(pygame.display.get_surface())
        pygame.display.update()

        return reward, screen_image
"""

# Define the basic parameters of the DeepQNetwork
ACTIONS = 4                 # NUMBER OF VALID ACTIONS
STAYACTION = 3              # NUMBER OF STAY ACTION IN ACTION VECTOR
GAMMA = 0.95                # DECAY RATE OF PAST OBSERVATIONS
OBSERVE = 100000            # TIME OF STEPS TO OBSERVE BEFORE TRAINING
EXPLORE = 2000000           # FRAMES OVER WHICH TO ANNEAL EPSILON
FINAL_EPSILON = 0.0001      # FINAL RATE OF EXPLORE
INITIAL_EPSILON = 0.5       # INITIAL RATE OF EXPLORE
REPLAY_MEMORY = 50000       # NUMBER OF PREVIOUS TRANSITIONS TO REMEMBER
BATCH = 32                  # SIZE OF MINIBATCH
FRAME_PER_ACTION = 1        # ONE ACTION, ONE FRAME FORWARD

# /----------------------------------NETWORK PART--------------------------------------------/


def weight_variable(shape):                             # random initial weight
    initial = tf.truncated_normal(shape, stddev=0.01)
    return tf.Variable(initial)


def bias_variable(shape):                               # constant initial bias
    initial = tf.constant(0.01, shape=shape)
    return tf.Variable(initial)


def conv2d(x, w, stride):                               # set the convolution kernel(stride:滑动步长，padding：填充方式)
    return tf.nn.conv2d(x, w, strides=[1, stride, stride, 1], padding="SAME")


def max_pool_2x2(x):                                    # set the pooling method of the CNN
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding = "SAME")


def createNetwork():                                    # ------FINALLY WE CREATE A NETWORK--------
    # network weights
    w_conv1 = weight_variable([8, 8, 4, 32])
    b_conv1 = bias_variable([32])

    w_conv2 = weight_variable([4, 4, 32, 64])
    b_conv2 = bias_variable([64])

    w_conv3 = weight_variable([3, 3, 64, 64])
    b_conv3 = bias_variable([64])

    # calculate how many featuremaps the results include by yourself(h*w*c)
    # [a,b], a=(h*w*c), b=the dimension of next vector
    w_fc1 = weight_variable([1600, 512])
    b_fc1 = bias_variable([512])

    w_fc2 = weight_variable([512, ACTIONS])
    b_fc2 = bias_variable([ACTIONS])

    # input layer
    s = tf.placeholder("float", [None, 80, 80, 4])

    # hidden layers
    h_conv1 = tf.nn.relu(conv2d(s, w_conv1, 4) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    h_conv2 = tf.nn.relu(conv2d(h_pool1, w_conv2, 2) + b_conv2)
    #h_pool2 = max_pool_2x2(h_conv2)

    h_conv3 = tf.nn.relu(conv2d(h_conv2, w_conv3, 1) + b_conv3)

    h_conv3_flat = tf.reshape(h_conv3, [-1, 1600])

    h_fc1 = tf.nn.relu(tf.matmul(h_conv3_flat, w_fc1) + b_fc1)

    # network_result layer
    network_result = tf.matmul(h_fc1, w_fc2) + b_fc2

    return s, network_result, h_fc1


def trainNetwork(s, net_result, h_fc1, sess):       # ------------TRAIN MY LITTLE DQN-------------
    # define the cost function
    a = tf.placeholder("float", [None, ACTIONS])    # action
    y = tf.placeholder("float", [None])             # the target value calculated based on the data in minibatch
    readout_action = tf.reduce_sum(tf.multiply(net_result, a), reduction_indices=1)     # predict action
    cost = tf.reduce_mean(tf.square(y - readout_action))                                # calculate of LOSS function
    train_step = tf.train.AdamOptimizer(1e-6).minimize(cost)                            # process of train circulation

    # import the game invironment
    # game = Game()

    # Game never over
    terminal = 0

    # store the previous observations in replay memory
    D = deque()

    # printing
    # a_file = open("logs_bike/readout.txt", 'w')
    # h_file = open("logs_bike/hidden.txt", 'w')

    stay = np.zeros([ACTIONS])
    stay[3] = 1
    # get the first state by doing nothing and preprocess the image to 80x80x4
    reward_t, frame_t = gc(stay)                             # do nothing
    # reward_t, frame_t = game.step(stay)
    # frame_t:input one frame; r_0:reward of first state; terminal:judge game stop or not

    frame_t = cv2.resize(frame_t, (80, 80))
    # ret, frame_t = cv2.threshold(frame_t, 1, 255, cv2.THRESH_BINARY)            # ret means nothing
    state_t = np.stack((frame_t, frame_t, frame_t, frame_t), axis=2)            # one whole input batch, 4 frames.
    # x_image_array = np.array(frame_t)

    # saving and loading networks
    saver = tf.train.Saver()
    sess.run(tf.initialize_all_variables())
    checkpoint = tf.train.get_checkpoint_state("saved_bike_networks")
    if checkpoint and checkpoint.model_checkpoint_path:
        saver.restore(sess, checkpoint.model_checkpoint_path)
        print("Successfully loaded:", checkpoint.model_checkpoint_path)
    else:
        print("Could not find old network weights")

    # start training
    epsilon = INITIAL_EPSILON
    t = 0       # times of train
    while True:
        # choose an action epsilon greedily
        result_t = net_result.eval(feed_dict={s: [state_t]})[0]         # the predict output of this time
        # action_t = np.zeros([ACTIONS])                                # the action vector
        action_t = np.zeros([ACTIONS])
        action_index = 0
        if t % FRAME_PER_ACTION == 0:                                   # decide what action to do
            if random.random() <= epsilon:
                print("----------Random Action----------")
                action_index = random.randrange(ACTIONS)
                action_t[random.randrange(ACTIONS)] = 1
            else:                                                   # --------------------WARNING-----------------------
                action_index = np.argmax(result_t)                  # ----------------NEEDS TO MODIFY!!!!!!-------------
                action_t[action_index] = 1                          # -----------------UPDATE:MODIFIED------------------
        else:
            action_t[STAYACTION] = 1          # do nothing

        # scale down epsilon
        if epsilon > FINAL_EPSILON and t > OBSERVE:
            epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE

        # run the selected action and observe next state and reward
        reward_t, frame_t1_origin = gc(action_t)    # ----------THE ACTIONS EXECUTED!--------
        next_frame_t = cv2.resize(frame_t1_origin, (80, 80))
        # ret, next_frame_t = cv2.threshold(next_frame_t, 1, 255, cv2.THRESH_BINARY)

        count = 1
        plt.figimage(next_frame_t)
        plt.savefig("x_t_image" + str(count) + ".png")                # to save a convoluted image to debug
        count += 1

        next_frame_t = np.reshape(next_frame_t, (80, 80, 1))            # unnecessary operation ??? NO,NECESSARY!
        next_state_t = np.append(next_frame_t, state_t[:, :, :3], axis=2)

        # count = 1                                                     # to save a reshaped image to debug//failed
        # plt.imshow(next_frame_t)
        # plt.show()
        # plt.savefig("x_t_image" + str(count) + ".png")

        # store the transition in D
        D.append((state_t, action_t, reward_t, next_state_t, terminal))
        if len(D) > REPLAY_MEMORY:
            D.popleft()

        # only train if done observing
        if t > OBSERVE:
            # sample a minibatch to train on
            minibatch = random.sample(D, BATCH)

            # get the batch variables
            state_batch = [d[0] for d in minibatch]
            action_batch = [d[1] for d in minibatch]
            reward_batch = [d[2] for d in minibatch]
            next_state_batch = [d[3] for d in minibatch]

            Qvalue_batch = []
            readout_j1_batch = net_result.eval(feed_dict = {s : next_state_batch})
            for i in range(0, len(minibatch)):
                terminal = minibatch[i][4]
                # if terminal, only equals reward
                if terminal:
                    Qvalue_batch.append(reward_batch[i])
                else:
                    Qvalue_batch.append(reward_batch[i] + GAMMA * np.max(readout_j1_batch[i]))

            # perform gradient step
            train_step.run(feed_dict = {
                y: Qvalue_batch,
                a: action_batch,
                s: state_batch}
            )

        # update the old values
        state_t = next_state_t
        t += 1

        # save progress every 10000 iterations
        if t % 10000 == 0:
            saver.save(sess, 'saved_bike_networks/' + 'bike' + '-dqn', global_step=t)

        # print info
        state = ""
        if t <= OBSERVE:
            state = "observe"
        elif t > OBSERVE and t <= OBSERVE + EXPLORE:
            state = "explore"
        else:
            state = "train"

        print("TIMESTEP", t, "/ STATE", state,
            "/ EPSILON", epsilon, "/ ACTION", action_index, "/ REWARD", end='')
        if reward_t==-1:
            print('\033[1;31m -1\033[0m'
, end='')
        else:
            print(  reward_t,end='')
        print("/ Q_MAX %e" % np.max(result_t))
        # write info to files

        # if t % 10000 <= 100:
        #     a_file.write(",".join([str(x) for x in result_t]) + '\n')
        #     h_file.write(",".join([str(x) for x in h_fc1.eval(feed_dict={s:[state_t]})[0]]) + '\n')
        #     cv2.imwrite("logs_tetris/frame" + str(t) + ".png", next_frame_t)


def startTrain():
    sess = tf.InteractiveSession()
    s, readout, h_fc1 = createNetwork()
    trainNetwork(s, readout, h_fc1, sess)


def startNetwork():         # START
    startTrain()

















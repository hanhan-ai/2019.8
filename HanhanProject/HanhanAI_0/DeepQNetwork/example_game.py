"""
开始时间2019.8.22
最后一次修改时间2019.8.30
赵士陆

一个测试神经网络的小游戏
"""
from DeepQNetwork.network import *
import pygame

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
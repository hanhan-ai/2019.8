import numpy as np
import cv2

action_number = 4
stay_action = 1     # 其实不是
def PacManPictureHandle(observation):
    cp_observation_origin = cv2.cvtColor(observation, cv2.COLOR_BGR2GRAY)
    # cp_observation_origin = np.reshape(cp_observation_origin, (-1, -1))
    # cp_observation = cv2.resize(cp_observation_origin, (inputImageSize[1], inputImageSize[0]))
    return cp_observation_origin


def PacManActionHandle(act):
    act_return = [0, 0, 0, 0]
    if all(act) == [1, 0, 0, 0]:
        act_return = [0, 1]
    elif all(act) == [0, 1, 0, 0]:
        act_return = [0, -1]
    elif all(act) == [0, 0, 1, 0]:
        act_return = [1, 0]
    elif all(act) == [0, 0, 0, 1]:
        act_return = [-1, 0]
    return act_return

def PacManTerminalHandle(is_win,is_gameover):
    terminal = False
    if is_gameover == True:
        terminal = True
    elif is_win == True:
        terminal = True

    return terminal
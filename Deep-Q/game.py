from selenium import webdriver
from selenium.webdriver import ChromeOptions
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.keys import Keys
import pyautogui

import numpy as np
import pandas as pd
import time
import os
from PIL import Image
from IPython.display import clear_output
from io import BytesIO
import base64
import json
import pickle
import random
import matplotlib.pyplot as plt

import cv2
from keras.models import model_from_json, load_model
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.optimizers import SGD , Adam
from keras.callbacks import TensorBoard
from collections import deque

#Global
PATH = "C:\\Users\\kanis\\Softwares\\chromedriver.exe"
game_url = "https://chromedino.com"
# game_url = "http://www.trex-game.skipser.com/"
init_script = "document.getElementsByClassName('runner-canvas')[0].id = 'runner-canvas'"
getbase64Script = "canvasRunner = document.getElementById('runner-canvas'); \
return canvasRunner.toDataURL().substring(22)"

generation_score = []
actions_df = pd.DataFrame(columns = ['actions'])
scores_df = pd.DataFrame(columns = ['scores'])
loss_df = pd.DataFrame(columns = ['loss'])
q_table = pd.DataFrame(columns = ['Q-values'])

def save_obj(obj, name):
    with open('./objs/' + name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name):
    with open('./objs/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)

def _init_():
    save_obj(INITIAL_EPSILON, 'epsilon')
    t = 0
    save_obj(t, 'time')
    
    experience_tuple = deque()
    save_obj(experience_tuple, 'experience_tuple')
    

class Game:
    def __init__(self):
        options = Options()
        options.add_experimental_option(
            "excludeSwitches", ["enable-automation"])
        options.add_experimental_option("useAutomationExtension", False)
        options.add_argument('--mute-audio')
        options.add_argument('--disable-notifications')
        options.add_argument('--disable-popup-blocking')
        # options.add_extension(extension_url)
        self.driver = webdriver.Chrome(PATH, options=options)
        self.driver.set_window_position(x=-10, y=0)
        self.driver.get(game_url)
        self.driver.execute_script('Runner.config.ACCELERATION=0')
        self.driver.execute_script(init_script)
        self.driver.implicitly_wait(30)
        self.driver.maximize_window()
        self.is_down = False

    def get_crashed(self):
        return self.driver.execute_script("return Runner.instance_.crashed")

    def get_playing(self):
        return self.driver.execute_script("return Runner.instance_.playing")

    def restart(self):
        if self.is_down:
            pyautogui.keyUp('down')
        self.driver.execute_script("Runner.instance_.restart()")
        print('restarted')

    def press_up(self):
        if self.is_down:
            pyautogui.keyUp('down')
        self.driver.find_element_by_tag_name('body').send_keys(Keys.ARROW_UP)
        print('Jumped')

    def press_down(self):
        # self.driver.find_element_by_tag_name('body').send_keys(Keys.ARROW_DOWN)
        pyautogui.keyDown('down')
        self.is_down = True
        print('Ducked')

    def get_score(self):
        score_arr = self.driver.execute_script(
            'return Runner.instance_.distanceMeter.digits')
        score = ''.join(score_arr)
        return int(score)

    def get_highscore(self):
        score_arr = self.driver.execute_script(
            'return Runner.instance_.distanceMeter.highscore')
        for i in range(len(score_arr)):
            if score_arr[i] == '':
                break
        score_arr = score_arr[i:]
        score = ''.join(score_arr)
        return int(score)

    def pause(self):
        # if self.is_down:
        #     pyautogui.keyUp('down')
        return self.driver.execute_script('return Runner.instance_.stop()')

    def resume(self):
        return self.driver.execute_script("return Runner.instance_.play()")

    def end(self):
        self.driver.close()

class DinoAgent:
    def __init__(self, game):
        self._game_ = game
        self.jump()

    def is_running(self):
        return self._game_.get_playing()

    def is_crashed(self):
        return self._game_.get_crashed()

    def jump(self):
        self._game_.press_up()

    def duck(self):
        self._game_.press_down()

class Gamestate:
    def __init__(self,agent,game):
        self._agent = agent
        self._game = game
        self._display = show_img() #display the processed image on screen using openCV, implemented using python coroutine 
        self._display.__next__() # initiliaze the display coroutine 

    def get_state(self,actions):
        actions_df.loc[len(actions_df)] = actions[1] # storing actions in a dataframe
        score = self._game.get_score() 
        reward = 0.1
        game_over = False #game over

        if actions[1] == 1:
            self._agent.jump()

        elif actions[2] == 1:
            self._agent.duck()
            

        image = screenshot(self._game.driver)
        self._display.send(image)

        if self._agent.is_crashed():
            scores_df.loc[len(loss_df)] = score # log the score when game is over
            self._game.restart()
            reward = -1
            game_over = True

        return image, reward, game_over #return the Experience tuple

def process_img(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = image[:500, : 600]
    image = cv2.resize(image, (84, 84)) # is the resizing done because of the input size image sent to neural net?
    image[image > 0] = 255
    image = np.reshape(image, (84, 84, 1))
    # plt.imshow(image)
    # plt.show()
    return image

def screenshot(driver):
    image_b64 = driver.execute_script(getbase64Script)
    screen = np.array(Image.open(BytesIO(base64.b64decode(image_b64))))
    image = process_img(screen)
    return image

def image_to_tensor(image):
    image = np.transpose(image, (2, 0, 1)) #Image.shape = (1, 84, 84) from (84, 84, 1)
    image_tensor = image.astype(np.float32)
    image_tensor = torch.from_numpy(image_tensor)
    # if torch.cuda.is_available(): #I can add this but will see properly if i will use this
    #     image_tensor = image_tensor.cuda()
    return image_tensor


def show_img(graphs = False):
    """
    Show images in new window
    """
    while True:
        screen = (yield)
        window_title = "logs" if graphs else "game_play"
        cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)        
        imS = cv2.resize(screen, (800, 400)) 
        cv2.imshow(window_title, screen)
        if (cv2.waitKey(1) & 0xFF == ord('q')):
            cv2.destroyAllWindows()
            break

#game parameters: Global
ACTIONS = 3 # possible actions: jump, duck, do nothing
GAMMA = 0.99 # Penalization rate
OBSERVATION = 1000 # timesteps to observe before training
EXPLORE = 1000  # frames over which to anneal epsilon
FINAL_EPSILON = 0.0001 # final value of epsilon
INITIAL_EPSILON = 0.1 # starting value of epsilon
REPLAY_MEMORY = 50000 # number of previous transitions to remember
BATCH = 16 # size of minibatch
FRAME_PER_ACTION = 1
LEARNING_RATE = 1e-4
img_rows , img_cols = 84, 84
img_channels = 4 #We stack 4 frames

def build_model():
    model = Sequential()
    model.add(Conv2D(32, (8, 8), padding='same',strides=(4, 4),input_shape=(img_cols,img_rows,img_channels)))  #80*80*4
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (4, 4),strides=(2, 2),  padding='same'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3),strides=(1, 1),  padding='same'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Activation('relu'))
    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dense(ACTIONS))
    adam = Adam(lr=LEARNING_RATE)
    model.compile(loss='mse',optimizer=adam)
        
    #create model file if not present
    model.save('./objs/model.h5')
    return model

def train_model(model, game_state, observe = False):
    last_time = time.time()
    
    experience_tuple = load_obj('experience_tuple') #Load experience tuple from memory
    
    do_nothing = np.zeros(ACTIONS)
    do_nothing[0] = 1
    
    image, reward, game_over = game_state.get_state(do_nothing)
    
    image_stack = np.stack((image, image, image, image), axis = 2)
    image_stack = image_stack.reshape(1, image_stack.shape[0], image_stack.shape[1], image_stack.shape[2])
    
    init_state = image_stack
    
    if observe:
        OBSERVE = 9999999 
        epsilon = FINAL_EPSILON
        model.load_weights('model.h5')
        adam = Adam(lr = LEARNING_RATE)
        model.compile(loss = 'mse', optimizer = adam)
    
    else:
        OBSERVE = OBSERVATION
        epsilon = load_obj('epsilon')
        model.load_weights('model.h5')
        adam = Adam(lr = LEARNING_RATE)
        model.compile(loss = 'mse', optimizer = adam)
        
    t = load_obj('time')
    
    while True:
        # Initialize loss, Q-value that is calculated on State-Action pairs, reward and actions and their indices 
        loss = 0
        Q_sa = 0
        action_index = 0
        reward_t = 0
        action_t = np.zeros([ACTIONS])
        
        #Exploration vs Exploitation:
        #Greedy-epsilon:
        if random.random() <= epsilon:
            print('---------Random Action----------')
            action_index = random.randrange(ACTIONS)
            action_t[action_index] = 1
        
        #Epsilon gradual decrease:
        if epsilon > FINAL_EPSILON and t > OBSERVE:
            epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE
            
        expected_next_image_t, expected_reward, terminal = game_state.get_state(action_t)
        
#         print('fps: {0}'.format(1 / (time.time()-last_time))) # helpful for measuring frame rate
#         last_time = time.time()
        expected_next_image_t = np.array(expected_next_image_t)
        # print(f"Shape expected state    {expected_next_image_t.shape}")
        # print(f"Shape image stack state    {image_stack.shape}")


        e_shape = expected_next_image_t.shape
        expected_next_image_t = expected_next_image_t.reshape(1, e_shape[0],  e_shape[1], 1) 
        next_image = np.append(expected_next_image_t, image_stack[:, :, :, : 3], axis = 3)
        
        
        
        experience_tuple.append((image_stack, action_index, reward_t, next_image, terminal)) #curr state, which action, cur reward, next state, gameOver?
        
        if len(experience_tuple) > REPLAY_MEMORY:
            experience_tuple.popleft()
        
        #Train if done observing:
        
        if t > OBSERVE:
            
            minibatch = random.sample(experience_tuple, BATCH)
            inputs = np.zeros((BATCH, image_stack.shape[1], image_stack.shape[2], image_stack.shape[3])) #16, 20, 20, 4
            targets = np.zeros((inputs.shape[0], ACTIONS))
            
            #Experience Replay:
            for i in range(len(minibatch)):
                image_stack_mini = minibatch[i][0] 
                action_t_mini = minibatch[i][1]
                reward_t_mini = minibatch[i][2]
                next_state_mini = minibatch[i][3] # Next state is nothing but the next image state
                game_over = minibatch[i][4]
                
                inputs[i : i + 1] = image_stack_mini

                # print(f"Mini batch image shape :   {image_stack_mini.shape}")
                
                targets[i] = model.predict(image_stack_mini)
                
                Q_sa = model.predict(next_state_mini)
                if game_over:
                    targets[i, action_t_mini] = reward_t_mini
                else:
                    targets[i, action_t_mini] = reward_t_mini + GAMMA * np.max(Q_sa)
 
            loss += model.train_on_batch(inputs, targets)
            loss_df.loc[len(loss_df)] = loss
            q_table.loc[len(q_table)] = np.max(Q_sa)
            
        image_stack = init_state if game_over else next_image
        t += 1
        print(f"Time steps : {t}")
        
        #Save Progress for every 1000 iterations
        if t % 100 == 0:
            print('Saving model')
            game_state._game.pause()
            model.save_weights('model.h5', overwrite = True)
            save_obj(experience_tuple, 'experience_tuple')
            save_obj(t, 'time')
            save_obj(epsilon, 'epsilon')
            save_obj(q_table, 'q_table')
            
            loss_df.to_csv('./objs/loss_df.csv', index = False)
            scores_df.to_csv('./objs/scores_df.csv', index = False)
            actions_df.to_csv("./objs/actions_df.csv", index=False)
            q_table.to_csv('./objs/q_table.csv', index=False)
            
            with open('model.json', 'w') as outfile:
                json.dump(model.to_json(), outfile)
            clear_output()
            
            game_state._game.resume()
            
            state = ''
            if t < OBSERVE:
                state = 'observe'
            elif t > OBSERVE and t <= OBSERVE + EXPLORE:
                state = 'explore'
            else:
                state = 'train'
            
            print("TIMESTEP", t, "/ STATE", state,             "/ EPSILON", epsilon, "/ ACTION", action_index, "/ REWARD", reward_t,             "/ Q_MAX " , np.max(Q_sa), "/ Loss ", loss)

            # if t == 1000:
            #     break
    print("Episode finished!")
    print("************************")

if __name__ == "__main__":
    _init_()
    game = Game()
    dino = DinoAgent(game)
    game_state = Gamestate(dino, game)
    # model = build_model()
    model = load_model('./objs/model.h5')
    try:
        observe = True
        train_model(model, game_state, observe)
    except StopIteration:
        game.end()

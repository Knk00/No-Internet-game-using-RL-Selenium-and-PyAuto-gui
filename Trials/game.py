from selenium import webdriver
from selenium.webdriver import ChromeOptions
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.keys import Keys
import time
import os

# import numpy as np
# import pandas as pd
# from PIL import Image
# from io import BytesIO
# import base64
# import json
# import pickle
# import random

# import cv2

PATH = "C:\\Users\\kanis\\Softwares\\chromedriver.exe"
game_url = "https://chromedino.com"
init_script = "document.getElementsByClassName('runner-canvas')[0].id = 'runner-canvas'"
getbase64Script = "canvasRunner = document.getElementById('runner-canvas'); \
return canvasRunner.toDataURL().substring(22)"

generation_score = []

class Game:
    def __init__(self):
        options = Options()
        options.add_experimental_option("excludeSwitches", ["enable-automation"])  
        options.add_experimental_option("useAutomationExtension", False)  
        options.add_argument('--mute-audio')
        self.driver = webdriver.Chrome(PATH, options= options)
        self.driver.set_window_position(x = -10, y = 0)
        self.driver.get(game_url)
        # self.driver.execute_script('Runner.config.ACCELERATION=0')
        self.driver.execute_script(init_script)
        self.driver.implicitly_wait(30)
        self.driver.maximize_window()
        
    def get_crashed(self):
        return self.driver.execute_script("return Runner.instance_.crashed")
    
    def get_playing(self):
        return self.driver.execute_script("return Runner.instance_.playing")
    
    def restart(self):
        self.driver.execute_script("Runner.instance_.restart()")
    
    def press_up(self):
        self.driver.find_element_by_tag_name('body').send_keys(Keys.ARROW_UP)
    
    def press_down(self):
        self.driver.find_element_by_tag_name('body').send_keys(Keys.ARROW_DOWN)
        print('pressed')
    
    def get_score(self):
        score_arr = self.driver.execute_script('return Runner.instance_.distanceMeter.digits')
        score = ''.join(score_arr)
        return int(score)

    def get_highscore(self):
        score_arr = self.driver.execute_script('return Runner.instance_.distanceMeter.highscore')
        for i in range(len(score_arr)):
            if score_arr[i] == '':
                break
        score_arr = score_arr[i:]
        score = ''.join(score_arr)
        return int(score)

    def pause(self):
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

class GameState:
    def __init__(self, dino, game):
        self._dino_ = dino
        self._game_ = game
        self._display_ = show_img()
        self._display_.__next__() # init the display routine

    def get_next_state(self, actions):
        score = self._game_.get_score()
        high_score = self._game_.get_highscore()

        reward = 0.1
        is_over = False
        if actions[0] == 1:
            self._dino_.jump()
        elif actions[1] == 1:
            self._dino_.duck()
        
        image = screenshot(self._game_.driver)

def screenshot(driver):
    image_b64 = driver.execute_script(getbase64Script)
    screen = np.array(Image.open(BytesIO(base64.b64decode(image_b64))))
    image = process_img(screen)
    return image

def process_img(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = image[:500, : 600]
    image = cv2.resize(image, (84, 84))
    image[image > 0] = 255
    image = np.reshape(image, (84, 84, 1))
    return image


g = Game()
time.sleep(2)
g.press_up()
# time.sleep(2)
g.press_down()
time.sleep(5)
print(g.get_score())



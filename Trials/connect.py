# from selenium import webdriver
# from selenium.webdriver.common.keys import Keys
# from selenium.webdriver.chrome.options import Options



# options = Options()
# # options.add_argument('--profile-directory=Default')
# # options.add_argument("user-data-dir=C:\\Program Files (x86)\\Google\\Chrome\\Application\\chrome")
# # options.add_experimental_option("excludeSwitches", ["enable-automation"])  
# # options.add_experimental_option("useAutomationExtension", False)  
# # options.add_argument('--mute-audio')

# driver = webdriver.Chrome(PATH, options=options)
# driver.set_window_position(-10, 0)
# driver.get('')
# driver.execute_script('Runner.config.ACCELERATION=0')
# driver.find_element_by_tag_name("body").send_keys(Keys.ARROW_UP)
# score_array = driver.execute_script("return Runner.instance_.distanceMeter.digits")
# score = ''.join(score_array)
# driver.maximize_window()
# print(score)

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.keys import Keys
import time

# game_url = "chrome://dino"
# # chromebrowser_path = "..\\Desktop\\chromedriver_win32\\chromedriver.exe"

# options = Options()
# options.add_argument('--profile-directory=Default')
# # options.add_argument("user-data-dir=C:\\ProgramData\\Microsoft\\Windows\\Start Menu")
# options.add_argument('disable-infobars')

# browser = webdriver.Chrome(executable_path = PATH, options=options)
# browser.get('chrome://dino/')
# time.sleep(10)
# browser.maximize_window()

class interface:
    def __init__(self, url, path):
        self.PATH = path
        self.URL = url

    def open_webpage(self):
        options = Options()
        options.add_argument('--profile-directory=Default')
        # options.add_argument("user-data-dir=C:\\ProgramData\\Microsoft\\Windows\\Start Menu")
        options.add_experimental_option("excludeSwitches", ["enable-automation"])  
        options.add_experimental_option("useAutomationExtension", False)  
        # options.add_argument('disable-infobars')
        driver = webdriver.Chrome(executable_path = self.PATH, options=options)
        driver.get(self.URL)
        driver.find_element_by_tag_name("body").send_keys(Keys.SPACE)
        # time.sleep(2)
        # txt = driver.find_element_by_xpath("//*[@id='error-information-popup-content']/div[2]")
        # print(txt)
        # driver.execute_script(<div class="error-code" jscontent="errorCode" jstcache="18">ERR_INTERNET_DISCONNECTED</div>)
        # driver.maximize_window()        

if __name__ == "__main__":
    inter = interface('chrome://dino', "C:\\Users\\kanis\\Softwares\\chromedriver.exe")
    inter.open_webpage()
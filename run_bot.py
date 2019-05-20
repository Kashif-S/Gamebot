import retro
import keyboard
import time
import cv2
import numpy as np
from pygame import sndarray, mixer
from threading import Thread
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import normalize


env = None

audio_rate = 0
memory_size = 10

frequency = 60
period = 1.0/frequency

def setEnv(envName, state=None):
    global env, audio_rate
    env = retro.make(envName, state, use_restricted_actions=retro.Actions.ALL)
    audio_rate = int(env.em.get_audio_rate())
    mixer.init(frequency=audio_rate, buffer=512)

def playSound(audio):
    sound = sndarray.make_sound(audio)
    sound.play()

def getInputArray():
    a = int(keyboard.is_pressed('z'))
    b = int(keyboard.is_pressed('x'))

    select = int(keyboard.is_pressed('shift'))
    start = int(keyboard.is_pressed('enter'))

    up = int(keyboard.is_pressed('up'))
    down = int(keyboard.is_pressed('down'))
    left = int(keyboard.is_pressed('left'))
    right = int(keyboard.is_pressed('right'))

    #x = int(keyboard.is_pressed('s'))
    #y = int(keyboard.is_pressed('a'))

    l = int(keyboard.is_pressed('a'))
    r = int(keyboard.is_pressed('s'))

    return [b, 0, select, start, up, down, left, right, a, 0, l, r]

def run():

    setEnv('MegamanZero2-Gba', 'Zero.state')
    obs = env.reset()

    model = load_model('model.h5')
    last_obs = np.zeros((memory_size, 80, 120, 1), dtype=np.int8)

    while not keyboard.is_pressed('escape'):
        time_before = time.time()
        thread = Thread(target=playSound, args=(env.em.get_audio(), ))
        thread.start()

        env.render()

        obs = cv2.resize(obs, (0, 0), fx=0.5, fy=0.5)
        obs = cv2.cvtColor(obs, cv2.COLOR_BGR2GRAY)
        obs = np.reshape(obs, (80, 120, 1))
        obs = normalize(obs)

        last_obs = np.append(last_obs[1:], [obs], axis=0)

        if keyboard.is_pressed('control'):
            action = getInputArray()
        else:
            action = model.predict(np.reshape(last_obs, (1, memory_size, 80, 120, 1)))
            print(action)
            action = np.array(np.around(action[0]))
            print(action)

        obs, _, _, _ = env.step(action)

        thread.join()

        while(time.time() - time_before) < period and not keyboard.is_pressed('space'):
            time.sleep(0.001)

run()




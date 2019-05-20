import retro
import keyboard
import time
import cv2
import numpy as np
from pygame import sndarray, mixer
from threading import Thread

env = None

audio_rate = 0

frequency = 60
period = 1.0/frequency

def setEnv(envName, state=None):
    global env, audio_rate
    env = retro.make(envName, 'Zero.state', use_restricted_actions=retro.Actions.ALL)
    audio_rate = int(env.em.get_audio_rate())
    mixer.init(frequency=audio_rate, buffer=512)

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

def playSound(audio):
    sound = sndarray.make_sound(audio)
    sound.play()

def run():

    setEnv('MegamanZero2-Gba', 'Zero.state')
    obs = env.reset()

    data = []

    while not keyboard.is_pressed('escape'):
        time_before = time.time()
        thread = Thread(target=playSound, args=(env.em.get_audio(), ))
        thread.start()

        env.render()

        obs = cv2.resize(obs, (0, 0), fx=0.5, fy=0.5)
        obs = cv2.cvtColor(obs, cv2.COLOR_BGR2GRAY)
        obs = np.reshape(obs, (80, 120, 1))

        action = getInputArray()

        if (not keyboard.is_pressed('control')):
            data.append([obs, action])

        obs, _, _, _ = env.step(action)

        thread.join()

        while(time.time() - time_before) < period and not keyboard.is_pressed('space'):
            time.sleep(0.001)

    training_data = np.array(data)
    np.savez_compressed('training_data.npz', training_data=training_data)

run()




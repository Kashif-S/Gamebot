# Gamebot
A Robot that learns to play games through machine learning.

This is a python project which allows users to record gameplay of Gameboy Advance games and train neural networks to mimic the user's gameplay.

This project uses the OpenAI Retro library to directly access the frame data. It records user input and frames in a dataset which is then used to train a LSTM convolutional neural network. The neural network then plays the game through the OpenAI Retro interface.

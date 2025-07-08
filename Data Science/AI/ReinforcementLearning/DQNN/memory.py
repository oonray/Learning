import numpy as np
import tensorflow as tf
import pandas as pd

memory_size = 500
batch = 32
memory_counter = 0

state_len = 6
actions = 3

memory = np.zeros((memory_size, state_len+2+state_len))

def reset_memory():
    memory = np.zeros((memory_size, state_len + 2 + state_len))
    mem_full = False

def store_transition(s, a, r, s_):
    global mem_full
    index = memory_counter % memory_size
    memory[index,:] = [i for i in s] + [a,r] + [i for i in s_]
    if memory_counter % memory_size == 0:
        return True
    else: return False

import gym
import inspect
import numpy as np
import tensorflow as tf
import pandas as pd

state_len = 6
actions = 3

from memory import *
from nn import *

env = gym.make("Acrobot-v1")

print(env.action_space)

train_step = 0
step_counter = 0
replace = 300

mem_full = False

Runs = tf.Variable(0)

with tf.variable_scope("logs"):
    tf.summary.scalar("cost", loss)
    cost_scalar = tf.summary.merge_all()


    with tf.Session() as sess:
        saver = tf.train.Saver()
        try:
            saver.restore(sess,"./model/")
        except:
            print("No previous runs found.")
        Runs += 1

        try:
            train_logger = tf.summary.FileWriter("./logs/train")
            sess.run(tf.global_variables_initializer())

            tf.summary.FileWriter("logs/", sess.graph)



            while True:
                observation = env.reset()
                env.render()

                while True:
                    """Taking an action"""
                    observation = observation[np.newaxis, :][0]

                    if np.random.uniform() < e and step_counter > 1000:
                        actions_value = sess.run([Q], feed_dict={s: [observation]})
                        action_t = np.argmax(actions_value[0])
                    else:
                        action_t = env.action_space.sample()

                    observation_, reward, done, options = env.step(action_t)
                    mem_full = store_transition(observation,action_t,reward, observation_)
                    memory_counter += 1

                    """Training The model"""


                    if done or mem_full:
                        if mem_full:
                                if step_counter % replace == 0:
                                    sess.run(target_replace_op)

                                if memory_counter >= memory_size:
                                    sample_index = np.random.choice(memory_size, size=batch)
                                else:
                                    sample_index = np.random.choice(memory_counter, size=batch)

                                batch_memory = memory[sample_index, :]
                                _, cost = sess.run(
                                    [optimizer, cost_scalar],
                                    feed_dict={
                                        s: batch_memory[:, :state_len],
                                        action: batch_memory[:, state_len + 1],
                                        r: batch_memory[:, state_len + 2],
                                        s_: batch_memory[:, -state_len:],
                                    })

                                observation = observation_
                                train_logger.add_summary(cost, train_step)
                                train_step += 1
                                reset_memory()
                        break
                step_counter += 1
        except KeyboardInterrupt as e:
            saver.save(sess,"./model/")
            raise e

        except Exception as e:
            raise e






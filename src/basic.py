import tensorflow as tf
import numpy as np

def bernoulli_sample_generation():
    import numpy as np
    rand = tf.random_uniform(shape=(10000, 1), seed=100)

    p_success = 0.5

    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        uniform_values = session.run(rand)

        print('probability of success: ')
        print(p_success)

        print('uniform values:')
        print(uniform_values.flatten())

        # convert probability into 0 or 1
        bernoulli_values = []
        for item in uniform_values:

            if item > p_success:
                bernoulli_values.append(1)
            else:
                bernoulli_values.append(0)
        print('bernoulli values:')
        print(bernoulli_values)

        print('mean = ' + str(np.mean(bernoulli_values)))
        print('variance = ' + str(np.var(bernoulli_values)))


def random_uniform():
    # generate a matrix MxN, where each row is a normal random distribution
    rand = tf.random_uniform(shape=(10000, 1000), seed=100)

    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        uniform_values = session.run(rand)

        print('mean = ' + str(np.mean(uniform_values[2, :])))
        print('variance = ' + str(np.var(uniform_values[2, :])))
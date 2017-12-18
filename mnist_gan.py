import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


from tensorflow.examples.tutorials.mnist import input_data
#training data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

#training parameters
training_steps = 1000
batch_size = 128
generator_input_size = 100


def sample_random(m, n):
    return np.random.uniform(-1, 1, size=[m, n])



#generator neural network architecture
generator_input = tf.placeholder(tf.float32, shape=[None, 100])
generator_w1 = tf.Variable(tf.random_uniform(shape=[100, 1500], minval= -0.01, maxval = 0.01))
generator_b1 = tf.Variable(tf.zeros(shape=[1500]))
generator_w2 = tf.Variable(tf.random_uniform(shape=[1500, 784], minval= -0.01, maxval = 0.01))
generator_b2 = tf.Variable(tf.zeros(shape=[784]))
generator_parameters = [generator_w1, generator_b1, generator_w2, generator_b2]

def generator(generator_input):
    hidden_output = tf.nn.relu(tf.matmul(generator_input, generator_w1) + generator_b1)
    generator_output = tf.nn.sigmoid(tf.matmul(hidden_output, generator_w2) + generator_b2)
    return generator_output


#discriminator neural network architecture
discriminator_input = tf.placeholder(tf.float32, shape=[None, 784])
discriminator_w1 = tf.Variable(tf.random_uniform(shape=[784, 150], minval = -0.01, maxval = 0.01))
discriminator_b1 = tf.Variable(tf.zeros(shape=[150]))
discriminator_w2 = tf.Variable(tf.random_uniform(shape=[150, 1], minval = -0.01, maxval = 0.01))
discriminator_b2 = tf.Variable(tf.zeros(shape=[1]))

discriminator_parameters = [discriminator_w1, discriminator_b1, discriminator_w2, discriminator_b2]


def discriminator(discriminator_input):
    hidden_output = tf.nn.relu(tf.matmul(discriminator_input, discriminator_w1) + discriminator_b1)
    discriminator_output = tf.nn.sigmoid(tf.matmul(hidden_output, discriminator_w2) + discriminator_b2)
    return discriminator_output




#objective functions
generator_output = generator(generator_input)
discriminator_real = discriminator(discriminator_input)
discriminator_fake = discriminator(generator_output)

discriminator_objective = -tf.reduce_mean(tf.log(discriminator_real) + tf.log(1.0 - discriminator_fake))
generator_objective = -tf.reduce_mean(tf.log(discriminator_fake))


discriminator_optimizer = tf.train.AdamOptimizer().minimize(discriminator_objective, var_list=discriminator_parameters)
generator_optimizer = tf.train.AdamOptimizer().minimize(generator_objective, var_list=discriminator_parameters)



def main():
    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()


    # Train
    for i in range(training_steps):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        sess.run(generator_output, feed_dict={generator_input: sample_random(16, generator_input_size)})
        _, current_discriminator_loss = sess.run([discriminator_optimizer, discriminator_objective], feed_dict={discriminator_input: batch_xs, generator_input: sample_random(batch_size, generator_input_size)})
        _, current_generator_loss = sess.run([generator_optimizer, generator_objective], feed_dict={generator_input: sample_random(batch_size, generator_input_size)})
        if i % 100 == 0:
            print(current_discriminator_loss, current_generator_loss)

if __name__ == '__main__':
    main()

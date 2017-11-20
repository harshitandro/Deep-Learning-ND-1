from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

# MNIST dataset loaded & one-hot encoded by tensorflow
mnist = input_data.read_data_sets("../dataset/mnist",one_hot=True,reshape=False)

# Setting learning parameters
learning_rate = 0.01
iterations = 30
batch_size = 128
display_step = 1
n_input = 784               # MNIST data input is 28*28 shape images
n_labels = 10                  # MNIST image categories

# Hidden Layer Parameters
n_hidden_nodes  = 256       # ie size of hidden layer

# Weights & Biases
weights = {
    'hidden_layer': tf.Variable(tf.truncated_normal([n_input,n_hidden_nodes])),
    'output_layer': tf.Variable(tf.truncated_normal([n_hidden_nodes,n_labels]))
}

##Done : Can be replaced with random init values
biases = {
    'hidden_layer': tf.Variable(tf.truncated_normal([n_hidden_nodes])),
    'output_layer': tf.Variable(tf.truncated_normal([n_labels]))
}


# tf Graph input
input = tf.placeholder("float",[None,28,28,1])
output = tf.placeholder("float",[None,n_labels])

flat_input = tf.reshape(input,[-1,n_input]) # why -1??

# Hidden layer with RELU activation
layer_1 = tf.add(tf.matmul(flat_input,weights['hidden_layer']),biases['hidden_layer'])
layer_1 = tf.nn.relu(layer_1)

# Output layer with linear activation
## Linear act function so as to make it logistic classifier
layer_output = tf.matmul(layer_1,weights['output_layer']) + biases['output_layer']

error = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=layer_output,labels=output))      # Softmax activation
print(error)
optimizer = tf.train.AdagradOptimizer(learning_rate=learning_rate).minimize(error)              # ADAGRAD as optimiser

# Initializing the variables
init = tf.global_variables_initializer()

# Run the Network
with tf.Session() as sess:
    sess.run(init)

    # Training cycle
    for iteration in range(iterations):
        total_batches = int(mnist.train.num_examples/batch_size)

        # Loop over all batches
        for i in range(total_batches):
            batch_input ,batch_output = mnist.train.next_batch(batch_size)
            sess.run(optimizer,feed_dict={input : batch_input , output : batch_output})

        # Display logs per epoch step
        if iteration % display_step == 0:
            c = sess.run(error, feed_dict={input: batch_input, output: batch_output})
            print("Epoch:", '%04d' % (iteration+1), "cost=", \
                "{:.9f}".format(c))

    print("Optimization Finished!")

    # Test model
    correct_prediction = tf.equal(tf.argmax(layer_output, 1), tf.argmax(output, 1))
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    # Decrease test_size if you don't have enough memory
    test_size = 256
    print("Accuracy:", accuracy.eval({input: mnist.test.images[:test_size], output: mnist.test.labels[:test_size]}))
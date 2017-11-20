from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("../dataset/mnist", one_hot=True, reshape=False)

import tensorflow as tf

# Parameters
learning_rate = 0.0002
epochs = 8
batch_size = 128

# Number of samples to calculate validation and accuracy
# Decrease this if you're running out of memory to calculate accuracy
test_valid_size = 256

# Network Parameters
n_classes = 10  # MNIST total classes (0-9 digits)
dropout = 0.75  # Dropout, probability to keep units

# Weights & Biases for 2 conv & 1 dropout layer
weights = {
    'wc1' : tf.Variable(tf.truncated_normal([5,5,1,32])),
    'wc2' : tf.Variable(tf.truncated_normal([5,5,32,64])),
    'wd1' : tf.Variable(tf.truncated_normal([7*7*64,1024])),
    'out' : tf.Variable(tf.truncated_normal([1024,n_classes])),
}

biases = {
    'bc1' : tf.Variable(tf.truncated_normal([32])),
    'bc2' : tf.Variable(tf.truncated_normal([64])),
    'bd1' : tf.Variable(tf.truncated_normal([1024])),
    'out' : tf.Variable(tf.truncated_normal([n_classes]))
}

def conv2d(x,W,b,strides=1):
    x = tf.nn.conv2d(x,W,strides=[1,strides,strides,1],padding='SAME')
    x = tf.nn.bias_add(x,b)
    return tf.nn.relu(x)

def maxpool2d(x,k=2):
    return tf.nn.max_pool(x,ksize=[1,k,k,1],strides=[1,k,k,1],padding='SAME')

def conv_net(x,weights,biases,dropout_rate):
    # Layer 1 : 28*28*1 to 14*14*32
    layer_conv1 = conv2d(x,weights['wc1'],biases['bc1'])
    layer_conv1 = maxpool2d(layer_conv1)

    # Layer 2 : 14*14*32 to 7*7*64
    layer_conv2 = conv2d(layer_conv1, weights['wc2'], biases['bc2'])
    layer_conv2 = maxpool2d(layer_conv2)

    # Fully connected layer : 7*7*64 to 1024
    layer_fully_connected = tf.reshape(layer_conv2,[-1,weights['wd1'].get_shape().as_list()[0]])
    print(layer_fully_connected.get_shape())
    layer_fully_connected = tf.add(tf.matmul(layer_fully_connected,weights['wd1']),biases['bd1'])
    layer_fully_connected = tf.nn.relu(layer_fully_connected)
    layer_fully_connected = tf.nn.dropout(layer_fully_connected,dropout_rate)

    # Output Layer : 1024 to 10
    layer_output = tf.add(tf.matmul(layer_fully_connected,weights['out']),biases['out'])
    return layer_output


## Session Implementation

# tf graph input
input = tf.placeholder(tf.float32,[None,28,28,1])
target = tf.placeholder(tf.float32,[None,n_classes])
keep_prob = tf.placeholder(tf.float32)

# Model
output = conv_net(input,weights,biases,keep_prob)

# Loss & Optimiser
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output,labels=target))
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

# Accuracy
correct_target = tf.equal(tf.argmax(output,1),tf.argmax(target,1))
accuracy = tf.reduce_mean(tf.cast(correct_target,tf.float32))

# Initialize global variables
init = tf.global_variables_initializer()


with tf.Session() as sess :
    sess.run(init)

    for epoch in range(epochs):
        for batch in range(mnist.train.num_examples//batch_size):
            batch_x , batch_y = mnist.train.next_batch(batch_size)

            # Run Optimiser
            sess.run(optimizer,feed_dict={
                input : batch_x,
                target : batch_y,
                keep_prob : dropout
            })

            # Run Loss & accuracy calc
            loss = sess.run(cost, feed_dict={
                input: batch_x,
                target: batch_y,
                keep_prob: 1.})

            valid_acc = sess.run(accuracy,feed_dict={
                input : mnist.validation.images[:test_valid_size],
                target : mnist.validation.labels[:test_valid_size],
                keep_prob : 1.0
            })

            print('Epoch {:>2}, Batch {:>3} -'
                  'Loss: {:>10.4f} Validation Accuracy: {:.6f}'.format(
                epoch + 1,
                batch + 1,
                loss,
                valid_acc))

    # Run Final Test Accuracy calc
    test_acc = sess.run(accuracy, feed_dict={
            input : mnist.test.images[:test_valid_size],
            target : mnist.test.labels[:test_valid_size],
            keep_prob: 1.0 })
    print('Testing Accuracy: {}'.format(test_acc))
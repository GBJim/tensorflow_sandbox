import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data



def get_weight(shape, stddev=0.1):
	return tf.Variable(tf.truncated_normal(shape, stddev))

def get_bias(shape):
		return tf.Variable(tf.constant(0.1, shape))


def conv_layer(x, W):
	return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')
def pooling_layer(x, ksize = [1,2,2,1]):
	return tf.nn.max_pool(x, ksize, strides=[1,2,2,1], padding='SAME')


def test():
	#print(weight_variable([784,10]))
	#print(bias_variable([10]))


if __name__ == "__main__":
	pass



mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
sess = tf.InteractiveSession()

x = tf.placeholder(tf.float32, shape = [None, 784])
y_ = tf.placeholder(tf.float32, shape = [None, 10])


W_1 = get_weight([5,5, 1 ,32])
b_1 = get_bias([32])
x_image = tf.reshape(x, [-1, 28,28,1])

activation_1 = tf.nn.relu(conv_layer(x, W_1) + b_1)
pool_1  = pooling_layer(activation_1)



W_2 = get_weight([5,5, 32 ,64])
b_2 = get_bias([64])

activation_2 = tf.nn.relu(conv_layer(pool_1, W_2) + b_2)
pool_2  = pooling_layer(activation_2)
pool_2_flat = tf.reshape(pool_2, 7*7*64)

W_dense = get_weight([7*8*64, 1024])
b_dense = get_bias([1024])

activation_dense = tf.nn.relu(tf.matmul(pool_2_flat, W_dense) + b_dense)
#softmax = tf.nn.softmax(activation_dense)

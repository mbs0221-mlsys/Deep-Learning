#  GRAPH CODE
# ============

# Import Tensorflow
import tensorflow as tf

# ======================
# Define the Graph
# ======================

# Define the Placeholders
X = tf.placeholder("float", [10, 10], name="X")
Y1 = tf.placeholder("float", [10, 1], name="Y1")
Y2 = tf.placeholder("float", [10, 1], name="Y2")

# Define the weights for the layers
shared_layer_weights = tf.Variable([10, 20], name="share_W")
Y1_layer_weights = tf.Variable([20, 1], name="share_Y1")
Y2_layer_weights = tf.Variable([20, 1], name="share_Y2")

# Construct the Layers with RELU Activations
shared_layer = tf.nn.relu(tf.matmul(X, shared_layer_weights))
Y1_layer = tf.nn.relu(tf.matmul(shared_layer, Y1_layer_weights))
Y2_layer = tf.nn.relu(tf.matmul(shared_layer, Y2_layer_weights))

# Calculate Loss
Y1_Loss = tf.nn.l2_loss(Y1, Y1_layer)
Y2_Loss = tf.nn.l2_loss(Y2, Y2_layer)

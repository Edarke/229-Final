#Test file for tensorflow operations

import tensorflow as tf

#sets up the graph
node1 = tf.constant(3.0, dtype=tf.float32)
node2 = tf.constant(4.0) # also tf.float32 implicitly
print(node1, node2)

#make a tensor flow session to run the graph

sess = tf.Session()

print (sess.run([node1, node2]))




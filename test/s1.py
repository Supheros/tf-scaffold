import tensorflow as tf
import os
import sys
sys.path.append(os.path.join(os.path.dirname("__file__"),os.path.pardir))
from serialize import saver as saver

v1 = tf.Variable([[1, 1, 1], [2, 2, 2]], dtype=tf.float32, name='v1')
v2 = tf.Variable([[3, 3, 3], [4, 4, 4]], dtype=tf.float32, name='v2')
output = tf.add(1 * v1, 1 * v2, name='add')

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    output_val = sess.run(output)
    print(output_val)
    print(v1.name)
    print(type(v1))
    print(type(v1.op))
    print(type(tf.get_default_graph().get_tensor_by_name("v1:0")))
    print(type(tf.get_default_graph().get_operation_by_name("v1")))

    print(tf.get_default_graph().get_tensor_by_name("v1:0").op is v1.op)
    print(tf.get_default_graph().get_operation_by_name("v1") is v1.op)

    for v in tf.trainable_variables():
        print(v, type(v), v.name, v.op.name)
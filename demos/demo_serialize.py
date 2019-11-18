import tensorflow as tf
import os
import sys

sys.path.append(os.path.join(os.path.dirname("__file__"), os.path.pardir))
from serialize import saver as saver


def build_graph(alpha, beta):
    v1 = tf.Variable([[1, 1, 1], [2, 2, 2]], dtype=tf.float32, name='v1')
    v2 = tf.Variable([[3, 3, 3], [4, 4, 4]], dtype=tf.float32, name='v2')
    output = tf.add(alpha * v1, beta * v2, name='add')
    return output, v1, v2


def test_save_all():
    build_graph(3, 5)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver.save(sess, './save_all', 0)


def test_save_part():
    output, v1, v2 = build_graph(3, 5)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        # saver.save(sess, './save_part', 0, var_list=[v1])
        saver.save(sess, './save_part', 0, tensor_name_list=['v1:0'])


def test_load_all():
    output, v1, v2 = build_graph(1, 1)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver.load_data(sess, './save_all')
        print(sess.run(output))


def test_load_part():
    v1 = tf.Variable([[10, 10, 10], [20, 20, 20]], dtype=tf.float32, name='v1')  # change value but restored
    v2 = tf.Variable([[30, 30, 30], [40, 40, 40]], dtype=tf.float32, name='v2')
    output = tf.add(1 * v1, 1 * v2, name='add')
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        # saver.load_data(sess, './save_part-0', var_list=[v1])
        saver.load_data(sess, './save_part-0', tensor_name_list=['v1:0'])
        print(sess.run(output))


def main():
    # test_save_all()
    # tf.reset_default_graph()
    # test_load_all()
    test_save_part()
    tf.reset_default_graph()
    test_load_part()


if __name__ == '__main__':
    main()

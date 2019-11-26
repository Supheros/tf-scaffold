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
    build_graph(1, 1)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver.save(sess, './save', 0)


def test_save_part():
    output, v1, v2 = build_graph(3, 5)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        # saver.save(sess, './save_part', 0, var_list=[v1])
        saver.save(sess, './save', 0, tensor_name_list=['v1:0'])

def test_save_tensor_name_exclude(tensor_name_exclude):
    build_graph(3, 5)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver.save(sess, './save', 0, tensor_name_exclude=tensor_name_exclude)

def test_save_tensor_name_matcher(tensor_name_matcher):
    build_graph(3, 5)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver.save(sess, './save', 0, tensor_name_matcher=tensor_name_matcher)

def test_load_all():
    output, v1, v2 = build_graph(1, 1)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        # 1. method 1: Give the model name and the step.
        # saver.load_data(sess, './save-0')
        saver.load_data(sess, ckpt_path='./')

        # 2. method 2: Give the path that contains the 'checkpoint'.
        # latest_model = tf.train.latest_checkpoint('./')
        # saver.load_data(sess, latest_model)
        print(sess.run(output))


def test_load_part():
    v1 = tf.Variable([[10, 10, 10], [20, 20, 20]], dtype=tf.float32, name='v1')
    v2 = tf.Variable([[30, 30, 30], [40, 40, 40]], dtype=tf.float32, name='v2')
    output = tf.add(1 * v1, 1 * v2, name='add')
    with tf.Session() as sess:
        # 1. Do not need to run initializer if load all tensors.
        sess.run(tf.global_variables_initializer())
        saver.load_data(sess, './save-0', var_list=[v2])

        # 2. Initialize all tensors and load part.
        # sess.run(tf.global_variables_initializer())
        # saver.load_data(sess, './save_all-0', var_list=[v1])

        # 3. Initialize all tensors and load part.
        # sess.run(tf.global_variables_initializer())
        # saver.load_data(sess, './save_all-0', tensor_name_list=['v2:0'])
        print(sess.run(output))

def test_load_tensor_name_exclude(tensor_name_exclude):
    v1 = tf.Variable([[10, 10, 10], [20, 20, 20]], dtype=tf.float32, name='v1')
    v2 = tf.Variable([[30, 30, 30], [40, 40, 40]], dtype=tf.float32, name='v2')
    output = tf.add(1 * v1, 1 * v2, name='add')
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver.load_data(sess, './save-0', tensor_name_exclude=tensor_name_exclude)
        print(sess.run(output))

def test_load_tensor_name_matcher(tensor_name_matcher):
    v1 = tf.Variable([[10, 10, 10], [20, 20, 20]], dtype=tf.float32, name='v1')
    v2 = tf.Variable([[30, 30, 30], [40, 40, 40]], dtype=tf.float32, name='v2')
    output = tf.add(1 * v1, 1 * v2, name='add')
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver.load_data(sess, ckpt_path='./', tensor_name_matcher=tensor_name_matcher)
        print(sess.run(output))

def test_load_struct_and_data(tensot_name):
    with tf.Session() as sess:
        saver.load_struct_and_data(sess, meta_path='./save-0.meta', ckpt_path='./')
        t = tf.get_default_graph().get_tensor_by_name(tensot_name)
        print(sess.run(t))

def main():
    test_save_all()
    # test_save_part()
    # test_save_tensor_name_exclude('v1')

    tf.reset_default_graph()

    # test_load_all()
    # test_load_part()
    # test_load_tensor_name_exclude('v2')
    # test_load_tensor_name_matcher('v1')
    test_load_struct_and_data('add:0')

if __name__ == '__main__':
    main()

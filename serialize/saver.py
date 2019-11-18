import tensorflow as tf
'''

'''
def save(sess, dir_path, global_step, var_list=None, tensor_name_list=None):
    saver = tf.train.Saver() if var_list is None else tf.train.Saver(var_list=var_list)
    if tensor_name_list is not None:
        vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        vars = [v for v in vars if v.name in tensor_name_list]
        saver = tf.train.Saver(var_list=vars)
    saver.save(sess, dir_path, global_step)

def load_data(sess, dir_path, var_list=None, tensor_name_list=None):
    saver = tf.train.Saver() if var_list is None else tf.train.Saver(var_list=var_list)
    if tensor_name_list is not None:
        vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        vars = [v for v in vars if v.name in tensor_name_list]
        saver = tf.train.Saver(var_list=vars)
    saver.restore(sess, dir_path)

def load_struct():
    pass
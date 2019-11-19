import tensorflow as tf
'''
Save all or some tensors, graph structure, index to file.
'''
def save(sess, dir_path, global_step, var_list=None, tensor_name_list=None, tensor_name_exclude=None, tensor_name_matcher=None):
    saver = tf.train.Saver() if var_list is None else tf.train.Saver(var_list=var_list)
    if tensor_name_list is not None:
        vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        if tensor_name_exclude is None:
            vars = [v for v in vars if v.name in tensor_name_list]
        else:
            vars = [v for v in vars if v.name in tensor_name_list and not v.name.startswith(tensor_name_exclude)]
        saver = tf.train.Saver(var_list=vars)
    elif tensor_name_matcher is not None:
        vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        vars = [v for v in vars if v.name in tensor_name_list and v.name.startswith(tensor_name_matcher)]
        saver = tf.train.Saver(var_list=vars)
    saver.save(sess, dir_path, global_step)
'''
Restore all or some tensors but not graph structure.
.index file is not needed.
'''
def load_data(sess, dir_path, var_list=None, tensor_name_list=None):
    saver = tf.train.Saver() if var_list is None else tf.train.Saver(var_list=var_list)
    if tensor_name_list is not None:
        vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        vars = [v for v in vars if v.name in tensor_name_list]
        saver = tf.train.Saver(var_list=vars)
    saver.restore(sess, dir_path)

def load_struct():
    pass
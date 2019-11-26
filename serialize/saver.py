import tensorflow as tf


def save(sess, dir_path, global_step, var_list=None, tensor_name_list=None, tensor_name_exclude=None,
         tensor_name_matcher=None):
    '''
    Save all or some tensors, graph structure, index to file.
    '''
    saver = build_saver(var_list=var_list, tensor_name_list=tensor_name_list, tensor_name_exclude=tensor_name_exclude,
                        tensor_name_matcher=tensor_name_matcher)
    saver.save(sess, dir_path, global_step)


def load_data(sess, dir_path=None, ckpt_path=None, var_list=None, tensor_name_list=None, tensor_name_exclude=None,
              tensor_name_matcher=None):
    '''
    Restore all or some tensors but not graph structure.
    .index file is not needed.
    '''
    assert dir_path is not None or ckpt_path is not None, 'dir_path and ckpt_path can not be None at the same time.'
    saver = build_saver(var_list=var_list, tensor_name_list=tensor_name_list, tensor_name_exclude=tensor_name_exclude,
                        tensor_name_matcher=tensor_name_matcher)
    if ckpt_path is not None:
        dir_path = tf.train.latest_checkpoint(ckpt_path)
    saver.restore(sess, dir_path)


def build_saver(var_list=None, tensor_name_list=None, tensor_name_exclude=None, tensor_name_matcher=None):
    '''
    Build a saver for serializing or deserializing values of tensors by specifying the
    variable list, the tensor name list, the excluded or the matched tensor names.
    '''
    saver = None
    if var_list is not None:
        saver = tf.train.Saver(var_list=var_list)
    elif tensor_name_list is not None:
        vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        vars = [v for v in vars if v.name in tensor_name_list]
        saver = tf.train.Saver(var_list=vars)
    elif tensor_name_exclude is not None:
        vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        vars = [v for v in vars if not v.name.startswith(tensor_name_exclude)]
        saver = tf.train.Saver(var_list=vars)
    elif tensor_name_matcher is not None:
        vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        vars = [v for v in vars if v.name.startswith(tensor_name_matcher)]
        saver = tf.train.Saver(var_list=vars)
    else:
        saver = tf.train.Saver()
    return saver


def load_struct_and_data(sess, meta_path, dir_path=None, ckpt_path=None):
    '''
    Restore all or some tensors and graph structure.
    Only .meta file is needed.
    '''
    assert meta_path is not None, 'meta_path can not be None.'
    assert dir_path is not None or ckpt_path is not None, 'dir_path and ckpt_path can not be None at the same time.'
    saver = tf.train.import_meta_graph(meta_path)
    if ckpt_path is not None:
        dir_path = tf.train.latest_checkpoint(ckpt_path)
    saver.restore(sess, dir_path)
    return saver

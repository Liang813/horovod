import tensorflow as tf
import horovod.tensorflow.keras as hvd

hvd.init()
opt = tf.keras.optimizers.Adam()
hopt = hvd.DistributedOptimizer(opt)
opt.get_config()
cfg = hopt.get_config()
opt_copy = opt.from_config(cfg)
opt_copy = opt.__class__.from_config(cfg)
hopt_copy = hopt.from_config(cfg) # TypeError: __init__() got an unexpected keyword argument 'learning_rate'
hopt_copy = hopt.__class__.from_config(cfg) # TypeError: __init__() got an unexpected keyword argument 'learning_rate'

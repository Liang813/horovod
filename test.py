import tensorflow as tf
import horovod.tensorflow as hvd

class TestModel(tf.keras.Model):

    def __init__(self):
        super().__init__()
        data_description = tf.TensorShape(
                [None, 40, 1]
            )
        layers = tf.keras.layers
        input_features = layers.Input(shape=data_description, dtype=tf.float32)
        inner = layers.Conv2D(
            filters=128,
            kernel_size=(3, 3),
            strides=(2, 2),
            padding="same",
            use_bias=False,
            data_format="channels_last",
        )(input_features)
        inner = layers.BatchNormalization()(inner)
        inner = tf.nn.relu6(inner)
        inner = layers.Conv2D(
            filters=128,
            kernel_size=(3, 3),
            strides=(2, 2),
            padding="same",
            use_bias=False,
            data_format="channels_last",
        )(inner)
        inner = layers.BatchNormalization()(inner)

        inner = tf.nn.relu6(inner)
        _, _, dim, channels = inner.get_shape().as_list()
        output_dim = dim * channels
        inner = layers.Reshape((-1, output_dim))(inner)
        inner = layers.Dense(512, activation=tf.nn.relu6)(inner)
        self.x_net = tf.keras.Model(inputs=input_features, outputs=inner, name="x_net")
        self.final_layer = layers.Dense(128, input_shape=(512,))

    def call(self, inputs, training: bool = None):
        x0 = inputs
        x = self.x_net(x0, training=training)
        x = tf.math.reduce_mean(x, axis=1)
        y = self.final_layer(x, training=training)
        return y

    def get_loss(self, logits):
        loss = self.loss_function(logits)
        return loss

    def loss_function(self, logits):
        cross_entropy = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
        hidden1, hidden2 = tf.split(logits, 2, 0)
        batch_size = tf.shape(hidden1)[0]
        hidden1_large = hvd.allgather(hidden1)
        hidden2_large = hvd.allgather(hidden2)
        enlarged_batch_size = tf.shape(hidden1_large)[0]
        replica_id = hvd.rank()
        labels_idx = tf.range(batch_size) + replica_id * batch_size
        labels = tf.one_hot(labels_idx, enlarged_batch_size * 2)
        logits_aa = tf.matmul(hidden1, hidden1_large, transpose_b=True)
        logits_ab = tf.matmul(hidden1, hidden2_large, transpose_b=True)
        loss = cross_entropy(labels, tf.concat([logits_ab, logits_aa], 1))
        return loss

class Solver():
    def __init__(self):
        self.model = TestModel()
        self.input_signature = [tf.TensorSpec(
        shape=(None, None, None, None), dtype=tf.float32
    )]
        self.optimizer = tf.keras.optimizers.SGD(learning_rate=0.1)
    def train_step(self, inputs):

        with tf.GradientTape() as tape:
            logits = self.model(inputs, training=True)
            loss = self.model.get_loss(logits)
        # Horovod: add Horovod Distributed GradientTape.
        tape = hvd.DistributedGradientTape(tape)
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

    def train(self):
        train_step = tf.function(self.train_step, input_signature=self.input_signature)
        x = tf.random.normal([32, 150, 40, 1])
        hvd.broadcast_variables(self.model.trainable_variables, root_rank=0)
        hvd.broadcast_variables(self.optimizer.variables(), root_rank=0)
        train_step(x)
        tf.print("Done")

if __name__ == "__main__":
    hvd.init()
    gpus = tf.config.experimental.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    if gpus:
        tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], "GPU")
    test_solver = Solver()
    test_solver.train()

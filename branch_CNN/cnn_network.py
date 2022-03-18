import os
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, Activation, Flatten, Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import TensorBoard, Callback
from keras import backend as K
from keras.constraints import Constraint
import numpy as np
from CNN_branch_data_generator import DataGeneratorCNNBranch 

class ConstrainedLayer(Constraint):
    def __call__(self, w):
       
        w_original_shape = w.shape
        w = w * 10000  # scale by 10k to prevent numerical issues

        # 1. Reshaping of 'w'
        x, y, z, n_kernels = w_original_shape[0], w_original_shape[1], w_original_shape[2], w_original_shape[3]
        center = x // 2 # Determine the center cell on the xy-plane.
        new_shape = [n_kernels, z, y, x]
        w = tf.reshape(w, new_shape)

        # 2. Set center values of 'w' to zero by multiplying 'w' with mask-matrix
        center_zero_mask = np.ones(new_shape)
        center_zero_mask[:, :, center, center] = 0
        w *= center_zero_mask

        # 3. Normalize values w.r.t xy-planes
        xy_plane_sum = tf.reduce_sum(w, [2, 3], keepdims=True)  # Recall new shape of w: (n_kernels, z, y, x).
        w = tf.math.divide(w, xy_plane_sum)  # Divide each element by its corresponding xy-plane sum-value

        # 4. Set center values of 'w' to negative one by subtracting mask-matrix from 'w'
        center_one_mask = np.zeros(new_shape)
        center_one_mask[:, :, center, center] = 1
        w = tf.math.subtract(w, center_one_mask)

        # Reshape 'w' to original shape and return
        return tf.reshape(w, w_original_shape)

    def get_config(self):
        return {}

class BranchCNNModel:

    def __init__(self, sector, model_files_path, tensorflow_files_path):
        self.sector = sector
        self.model = None
        self.model_name = None
        self.global_save_model_dir = self.__generate_model_path(model_files_path)
        self.global_tensorboard_dir = self.__generate_tensor_path(tensorflow_files_path)

    def __generate_model_name(self):
        model_name = f"BayayrModel-{self.sector}"

        return model_name

    def __generate_model_path(self, model_files_path):
        path_base = model_files_path
        new_path = os.path.join(path_base, self.sector)
        return new_path

    def __generate_tensor_path(self, tensorflow_files_path):
        path_base = tensorflow_files_path
        new_path = os.path.join(path_base, self.sector)
        return new_path

    def create_model(self, num_classes, model_name=None):
        input_shape = (128, 128, 3)

        input_layer = Input(shape=input_shape)

        constrained_conv_layer = Conv2D(filters=3,
                                kernel_size=(5,5),
                                strides=(1, 1),
                                padding="valid", # Intentionally
                                kernel_constraint=ConstrainedLayer(),
                                name="constrained_layer")(input_layer)

        conv2d_1 = Conv2D(96, (5,5), strides=(2,2),padding='same')(constrained_conv_layer)
        batch_norm1 = BatchNormalization()(conv2d_1)
        activation1 = Activation(tf.keras.activations.tanh)(batch_norm1)
        max_pool1 = MaxPooling2D(pool_size=(3,3), strides=2, padding='same')(activation1)

        conv2d_2 = Conv2D(64, (3,3), strides=(1,1),padding='same')(max_pool1)
        batch_norm2 = BatchNormalization()(conv2d_2)
        activation2 = Activation(tf.keras.activations.tanh)(batch_norm2)
        max_pool2 = MaxPooling2D(pool_size=(3,3), strides=2, padding='same')(activation2)

        conv2d_3 = Conv2D(64, (3,3), strides=(1,1),padding='same')(max_pool2)
        batch_norm3 = BatchNormalization()(conv2d_3)
        activation3 = Activation(tf.keras.activations.tanh)(batch_norm3)
        max_pool3 = MaxPooling2D(pool_size=(3,3), strides=2, padding='same')(activation3)

        conv2d_4 = Conv2D(128, (1,1), strides=1,padding='same')(max_pool3)
        batch_norm4 = BatchNormalization()(conv2d_4)
        activation4 = Activation(tf.keras.activations.tanh)(batch_norm4)
        max_pool4 = MaxPooling2D(pool_size=(3,3), strides=2, padding='same')(activation4)

        flatten = Flatten()(max_pool4)

        dense1 = Dense(200)(flatten)
        activation4 = Activation(tf.keras.activations.tanh)(dense1)

        dense2 = Dense(200)(activation4)
        activation4 = Activation(tf.keras.activations.tanh)(dense2)

        final_dense = Dense(num_classes, activation=tf.keras.activations.softmax)(activation4)

        model = Model(input_layer, final_dense)

        opt = tf.keras.optimizers.Adam()
        model.compile(loss="categorical_crossentropy",
                      optimizer=opt,
                      metrics=['accuracy'])

        model_name = self.__generate_model_name()

        self.model_name = model_name
        self.model = model
        
        return model

    def train(self, train_ds, val_ds_test, num_classes):
        if self.model is None:
            raise ValueError("Cannot start training! self.model is None!")

        callbacks = self.get_callbacks()
        
        self.model.fit(DataGeneratorCNNBranch(train_ds, num_classes=num_classes, batch_size=32),
                       epochs=100,
                       initial_epoch=0,
                       validation_data=DataGeneratorCNNBranch(val_ds_test, num_classes=num_classes, batch_size=32, shuffle=False),
                       callbacks=callbacks,
                       workers=12,
                       use_multiprocessing=True)

    def get_tensorboard_path(self):
        if self.model_name is None:
            raise ValueError("Model has no name specified. This is required in order to save TensorBoard log-files.")

        # Create directory if not exists
        path = os.path.join(self.global_tensorboard_dir, self.model_name)
        if not os.path.exists(path):
            os.makedirs(path)

        return path

    def print_model_summary(self):
        if self.model is None:
            print("Can't print model summary, self.model is None!")
        else:
            print(f"\nSummary of model:\n{self.model.summary()}")

    def get_save_model_path(self, file_name):
        if self.model_name is None:
            raise ValueError("Model has no name specified. This is required in order to save checkpoints.")

        # Create directory if not exists
        path = os.path.join(self.global_save_model_dir, self.model_name)
        if not os.path.exists(path):
            os.makedirs(path)

        # Append file name and return
        return os.path.join(path, file_name)
        
    def get_callbacks(self):
        default_file_name = "fm-e{epoch:05d}.h5"
        save_model_path = self.get_save_model_path(default_file_name)

        save_model_cb = tf.keras.callbacks.ModelCheckpoint(filepath=save_model_path,
                                                            monitor='val_accuracy',
                                                            save_best_only=True,
                                                            verbose=1,
                                                            save_weights_only=False,
                                                            period=1)
                                                 

        tensorboard_cb = TensorBoard(log_dir=self.get_tensorboard_path())
        print_lr_cb = PrintLearningRate()

        return [save_model_cb, tensorboard_cb, print_lr_cb]

class PrintLearningRate(Callback):
    def on_epoch_end(self, epoch, logs=None):
        lr = self.model.optimizer.lr
        decay = self.model.optimizer.decay
        iterations = self.model.optimizer.iterations
        lr_with_decay = lr / (1. + decay * K.cast(iterations, K.dtype(decay)))
        print(f"Learning rate on_epoch_end epoch {epoch}: {K.eval(lr_with_decay)}")
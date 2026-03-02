from math import pi

from keras import Layer, ops
from keras.initializers import Constant


class Degree2Radians(Layer):
    def __init__(self, **kwargs):
        super().__init__()

    def call(self, inputs, **kwargs):
        return inputs * (pi / 180.0)

    def compute_output_shape(self, input_shape):
        return input_shape


class LonLatScale(Layer):
    def build(self, input_shape):
        self.weight = self.add_weight(
            shape=input_shape[-1],
            initializer=Constant(value=[1.0 / pi, 2.0 / pi]),
            trainable=False,
        )
        self.bias = self.add_weight(
            shape=input_shape[-1],
            initializer=Constant(value=0.0),
            trainable=False,
        )

    def call(self, inputs, **kwargs):
        return inputs * self.weight + self.bias

    def compute_output_shape(self, input_shape):
        return input_shape


class Cartesian3DEncoder(Layer):
    def __init__(self, **kwargs):
        super().__init__()

    def call(self, inputs, **kwargs):
        cos_lon = ops.cos(inputs[..., 0])
        sin_lon = ops.sin(inputs[..., 0])
        cos_lat = ops.cos(inputs[..., 1])
        sin_lat = ops.sin(inputs[..., 1])
        inputs = ops.stack(
            [cos_lon * cos_lat, sin_lon * cos_lat, sin_lat], axis=-1
        )
        return inputs

    def compute_output_shape(self, input_shape):
        return input_shape[:-1] + (3,)

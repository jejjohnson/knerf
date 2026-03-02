from keras import Layer
from keras.initializers import Constant


class ScalingAndOffset(Layer):
    def __init__(self, scale: float = 1.0, offset: float = 0.0):
        super().__init__()
        self.scale = scale
        self.offset = offset

    def build(self, input_shape):
        self.weight = self.add_weight(
            shape=input_shape[-1],
            initializer=Constant(value=self.scale),
            trainable=False,
        )
        self.bias = self.add_weight(
            shape=input_shape[-1],
            initializer=Constant(value=self.offset),
            trainable=False,
        )

    def call(self, inputs, **kwargs):
        return inputs * self.weight + self.bias

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        return {
            "scale": self.scale,
            "offset": self.offset,
        }

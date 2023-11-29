from keras import Layer


class Identity(Layer):
    def __init__(self, **kwargs):
        super().__init__()

    def call(self, inputs, **kwargs):
        return inputs

    def compute_output_shape(self, input_shape):
        return input_shape

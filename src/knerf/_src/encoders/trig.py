from keras import Layer, ops


class CyclicEncoder(Layer):
    def __init__(self, **kwargs):
        super().__init__()

    def call(self, inputs, **kwargs):
        cos_lon = ops.cos(inputs[..., 0])
        sin_lon = ops.sin(inputs[..., 0])
        cos_lat = ops.cos(inputs[..., 1])
        sin_lat = ops.sin(inputs[..., 1])
        return ops.stack([cos_lon, sin_lon, cos_lat, sin_lat], axis=-1)

    def compute_output_shape(self, input_shape):
        return input_shape[:-1] + (4,)

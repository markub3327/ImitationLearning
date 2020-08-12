from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import TimeDistributed, Lambda


def BaseModel(x):
    # rescale the input channels to a range of [-1, 1]        
    normalized = TimeDistributed(Lambda(lambda x: (x / 127.5) - 1.0), name='normalization')(x)

    base_model = MobileNetV2(input_shape=x.shape[2:],
                                     include_top=False,
                                     weights='imagenet', 
                                     pooling='avg')

    # Create the base model from the pre-trained model MobileNet V2
    l1 = TimeDistributed(base_model, name='MobileNetV2')(normalized)

    return l1, base_model
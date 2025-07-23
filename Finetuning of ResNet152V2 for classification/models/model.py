
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.applications import ResNet152V2

def CNN_model(input_shape):
    inputs = Input(shape=input_shape)
    base = ResNet152V2(include_top=False, weights='imagenet', input_tensor=inputs)
    base.trainable = False
    x = base.output
    x = Flatten()(x)
    x = Dropout(0.3)(x)
    x = Dense(1024, activation='relu')(x)
    x = Dense(512, activation='relu')(x)
    x = Dense(256, activation='relu')(x)
    x = Dense(128, activation='relu')(x)
    x = Dense(2, activation='softmax')(x)
    return Model(inputs, x)

def classify_model():
    model = CNN_model((224, 224, 3))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

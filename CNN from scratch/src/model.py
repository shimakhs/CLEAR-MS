from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from keras.constraints import UnitNorm
from tensorflow.keras.optimizers import Adam

def CNN_model(input_shape=(60, 256, 3), num_classes=2):
    inputs = Input(shape=input_shape)
    x = Conv2D(128, (5, 5), activation='relu', kernel_constraint=UnitNorm(), padding='same')(inputs)
    x = Conv2D(128, (5, 5), activation='relu', kernel_constraint=UnitNorm(), padding='same')(x)
    x = MaxPooling2D((2, 2))(x)
    x = BatchNormalization()(x)

    x = Conv2D(64, (4, 4), activation='relu', kernel_constraint=UnitNorm(), padding='same')(x)
    x = Conv2D(64, (3, 3), activation='relu', kernel_constraint=UnitNorm(), padding='same')(x)
    x = MaxPooling2D((2, 2))(x)
    x = BatchNormalization()(x)

    x = Conv2D(32, (3, 3), activation='relu', kernel_constraint=UnitNorm(), padding='same')(x)
    x = Conv2D(32, (3, 3), activation='relu', kernel_constraint=UnitNorm(), padding='same')(x)
    x = MaxPooling2D((2, 2))(x)
    x = BatchNormalization()(x)

    x = Conv2D(16, (3, 3), activation='relu', kernel_constraint=UnitNorm(), padding='same')(x)
    x = Conv2D(16, (3, 3), activation='relu', kernel_constraint=UnitNorm(), padding='same')(x)
    x = MaxPooling2D((2, 2))(x)
    x = BatchNormalization()(x)

    x = Dropout(0.25)(x)
    x = Flatten()(x)
    x = Dense(256, activation='relu', kernel_constraint=UnitNorm())(x)
    x = Dense(128, activation='relu', kernel_constraint=UnitNorm())(x)
    x = Dense(64, activation='relu', kernel_constraint=UnitNorm())(x)
    x = Dense(32, activation='relu', kernel_constraint=UnitNorm())(x)
    x = Dense(16, activation='relu', kernel_constraint=UnitNorm())(x)
    outputs = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs, outputs)
    model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
    return model

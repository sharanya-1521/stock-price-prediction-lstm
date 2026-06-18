import tensorflow as tf

def get_callbacks(model_path="models/lstm_best.h5"):
    """
    Tracks network evaluation stability parameters to stop over-training.
    """
    return [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=model_path, save_best_only=True, monitor='val_loss', mode='min', verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=7, restore_best_weights=True, verbose=1
        )
    ]
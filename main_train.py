import sys
import os
sys.path.append(os.path.abspath("."))

from src.data_pipeline.window_generator import WindowGenerator
from src.model.lstm import build_lstm_model
from src.training.callbacks import get_callbacks

def train():
    INPUT_WIDTH = 60 # Check history windows back 60 days 
    LABEL_WIDTH = 1
    SHIFT = 1
    
    os.makedirs("models", exist_ok=True)
    
    print("🔄 Initializing Sequencing Pipeline Module...")
    wg = WindowGenerator(
        input_width=INPUT_WIDTH, label_width=LABEL_WIDTH, shift=SHIFT,
        train_path="data/processed/train.csv",
        val_path="data/processed/val.csv",
        test_path="data/processed/test.csv"
    )
    
    X_train, y_train = wg.get_train_data()
    X_val, y_val = wg.get_val_data()
    
    print(f"🚀 Training shape: {X_train.shape} | Targets: {y_train.shape}")
    model = build_lstm_model(input_shape=(INPUT_WIDTH, X_train.shape[2]))
    callbacks = get_callbacks("models/lstm_best.h5")
    
    model.fit(
        X_train, y_train, validation_data=(X_val, y_val),
        epochs=30, batch_size=32, callbacks=callbacks, verbose=1
    )
    print("✅ Best neural weights state saved to models/lstm_best.h5")

if __name__ == "__main__":
    train()
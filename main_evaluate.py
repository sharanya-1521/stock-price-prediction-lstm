import sys
import os
sys.path.append(os.path.abspath("."))

import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import load_model
from src.data_pipeline.window_generator import WindowGenerator

def evaluate_and_plot():
    model = load_model("models/lstm_best.h5", compile=False)
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    
    wg = WindowGenerator(
        input_width=60, label_width=1, shift=1,
        train_path="data/processed/train.csv",
        val_path="data/processed/val.csv",
        test_path="data/processed/test.csv"
    )
    
    X_test, y_test = wg.get_test_data()
    loss, mae = model.evaluate(X_test, y_test, verbose=0)
    predictions = model.predict(X_test)
    
    print("\n✅ Test Evaluation Metrics")
    print(f"📊 Mean Squared Error (MSE) : {loss:.6f}")
    print(f"📊 Root Mean Squared Error (RMSE): {np.sqrt(loss):.6f}")
    print(f"📊 Mean Absolute Error (MAE)  : {mae:.6f}")
    
    # Generate verification trend plot
    plt.figure(figsize=(12, 6))
    plt.plot(y_test[:, 0, 0], label="Actual Market Price", color="blue", alpha=0.7)
    plt.plot(predictions[:, 0], label="Predicted LSTM Trend", color="orange", linestyle="--", alpha=0.9)
    plt.title("JNTUH Project - Actual vs Predicted Stock Vector Matrix")
    plt.xlabel("Timeline Steps (Days)")
    plt.ylabel("Normalized Price Bounds")
    plt.legend()
    plt.grid(True)
    
    output_path = "models/prediction_plot.png"
    plt.savefig(output_path)
    print(f"📈 Performance line visualization graph output to: {output_path}")
    plt.show()

if __name__ == "__main__":
    evaluate_and_plot()
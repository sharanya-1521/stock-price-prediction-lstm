import numpy as np
import pandas as pd

class WindowGenerator:
    """
    Transforms sequential multi-feature arrays into overlapping window
    matrices (X) matched to future sequential continuous vector outputs (y).
    """
    def __init__(self, input_width, label_width, shift, train_path, val_path, test_path):
        self.input_width = input_width
        self.label_width = label_width
        self.shift = shift
        
        # Load your data splits
        self.train_df = pd.read_csv(train_path)
        self.val_df = pd.read_csv(val_path)
        self.test_df = pd.read_csv(test_path)

    def _split_window(self, features):
        # features shape: (batch, time_steps, features)
        inputs = features[:, :self.input_width, :]
        labels = features[:, self.input_width:, :]
        
        # Keep the 'Close' price as the primary target metric index at column index 0
        labels = labels[:, :, 0:1] 
        return inputs, labels

    def _make_dataset(self, data):
        data = np.array(data, dtype=np.float32)
        ds_sequence = []
        total_window_size = self.input_width + self.shift
        
        for i in range(len(data) - total_window_size + 1):
            ds_sequence.append(data[i:i + total_window_size])
            
        ds_sequence = np.array(ds_sequence)
        return self._split_window(ds_sequence)

    def get_train_data(self): 
        return self._make_dataset(self.train_df)
        
    def get_val_data(self): 
        return self._make_dataset(self.val_df)
        
    def get_test_data(self): 
        return self._make_dataset(self.test_df)
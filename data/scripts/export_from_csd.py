#!/usr/bin/env python3
import os
import numpy as np
import pickle
from pathlib import Path
from mmsdk import mmdatasdk

def export_aligned(csd_folder='final_aligned',
                   out_file=None,
                   seq_len=50):
    if out_file is None:
        out_file = Path(__file__).resolve().parents[1] / "aligned_mosei_dataset.pkl"
    # 1. Load the final_aligned CSDs
    ds = mmdatasdk.mmdataset(csd_folder)

    # 2. Extract sliding-window tensors for train/valid/test
    folds = [
        mmdatasdk.cmu_mosei.standard_folds.standard_train_fold,
        mmdatasdk.cmu_mosei.standard_folds.standard_valid_fold,
        mmdatasdk.cmu_mosei.standard_folds.standard_test_fold,
    ]
    tensors = ds.get_tensors(
        seq_len=seq_len,
        non_sequences=['All Labels'],
        direction=False,
        folds=folds
    )
    split_names = ['train', 'valid', 'test']

    # 3. Build a combined dict
    data = {}
    for split, tensor in zip(split_names, tensors):
        # tensor['glove_vectors'] shape (N,50,300), etc.
        # Convert one-hot labels (N,1,7) → class indices (N,)
        labels_onehot = tensor['All Labels'][:, 0, :]    # (N,7)
        labels_idx    = np.argmax(labels_onehot, axis=1) # (N,)

        data[split] = {
            'text':   tensor['glove_vectors'],
            'audio':  tensor['COVAREP'],
            'vision': tensor['FACET 4.2'],
            'labels': labels_idx,
        }

    # 4. Save the entire dict as a pickle file
    with open(out_file, 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"✔️ Saved combined dataset to `{out_file}`")

if __name__ == '__main__':
    export_aligned()

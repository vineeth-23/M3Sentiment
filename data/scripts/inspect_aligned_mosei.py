#!/usr/bin/env python3
import numpy as np

def inspect_aligned(npy_path='aligned_mosei_dataset.npy'):
    # load the dict
    data = np.load(npy_path, allow_pickle=True).item()

    for split in ['train', 'valid', 'test']:
        split_data = data.get(split, {})
        n = split_data.get('labels', np.empty((0,))).shape[0]
        print(f"\n=== {split.upper()} SPLIT ===")
        print(f"  • # samples: {n}")

        # modalities
        for mod in ['text', 'audio', 'vision']:
            arr = split_data.get(mod)
            if arr is not None:
                print(f"  • {mod:<6s}: shape={arr.shape}, dtype={arr.dtype}")

        # labels stats
        labels = split_data.get('labels')
        if labels is not None:
            print(labels)
            print(f"  • labels : shape={labels.shape}, dtype={labels.dtype}")
            mn, mx, mu, sd = labels.min(), labels.max(), labels.mean(), labels.std()
            print(f"      – min={mn:.3f}, max={mx:.3f}, mean={mu:.3f}, std={sd:.3f}")

        # optional fields
        if 'raw_text' in split_data:
            print(f"  • raw_text: {len(split_data['raw_text'])} sequences")
        if 'ids' in split_data:
            print(f"  • ids     : {len(split_data['ids'])} entries")

if __name__ == '__main__':
    inspect_aligned()

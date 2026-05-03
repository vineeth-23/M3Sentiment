
import os
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from mmsdk import mmdatasdk
from mmsdk.mmdatasdk import log

def compute_bert_embeddings(raw_ds, device="cpu"):
    """Turn each word in the raw text sequence into a BERT embedding."""
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model     = AutoModel.from_pretrained("bert-base-uncased").to(device).eval()
    bert_data = {}

    for utt, seq in raw_ds.computational_sequences["text"].data.items():
        words     = [w[0] for w in seq["features"]]        # extract each token string
        intervals = seq["intervals"]                       # (N,2) array of word timestamps
        feats     = []

        # embed each word separately
        for w in words:
            inputs = tokenizer(w, return_tensors="pt").to(device)
            with torch.no_grad():
                out = model(**inputs)
            # use the pooled CLS output as the word vector
            feats.append(out.pooler_output.cpu().numpy().squeeze(0))

        bert_data[utt] = {
            "intervals": intervals, 
            "features": np.stack(feats)  # (N, 768)
        }

    return bert_data


def download_data(base_dir="cmumosei_all/"):
    os.makedirs(base_dir, exist_ok=True)
    return {
        "raw":      mmdatasdk.mmdataset(mmdatasdk.cmu_mosei.raw,      base_dir+"raw/"),
        "high":     mmdatasdk.mmdataset(mmdatasdk.cmu_mosei.highlevel, base_dir+"high/"),
        "labels":   mmdatasdk.mmdataset(mmdatasdk.cmu_mosei.labels,    base_dir+"labels/")
    }


def process_data(use_bert: bool = True):
    cmu = download_data()

    # 1) If requested, compute BERT embeddings and inject as a new sequence:
    if use_bert:
        log.status("Computing BERT word embeddings …")
        device    = "cuda" if torch.cuda.is_available() else "cpu"
        bert_seq  = compute_bert_embeddings(cmu["raw"], device)
        # create a temp dataset just to hold BERT
        bert_ds   = mmdatasdk.mmdataset({}, "cmumosei_bert/")
        bert_ds._mmdatasdk__set_computational_sequences(bert_seq)
        # inject into the highlevel dataset
        cmu["high"].computational_sequences["bert_vectors"] = \
            bert_ds.computational_sequences["text"]
        reference = "bert_vectors"
    else:
        reference = "glove_vectors"

    # 2) Word‐level align on chosen reference
    cmu["high"].align(reference)
    cmu["high"].impute(reference)
    deploy_files = {x: x for x in cmu["high"].computational_sequences}
    cmu["high"].deploy(f"word_aligned_{reference}", deploy_files)

    # 3) Align to labels, unify, and final deploy
    cmu["high"].computational_sequences["All Labels"] = cmu["labels"]["All Labels"]
    cmu["high"].align("All Labels")
    cmu["high"].hard_unify()
    cmu["high"].deploy("final_aligned", deploy_files)

    # 4) Extract tensors for train/val/test
    folds = [
        mmdatasdk.cmu_mosei.standard_folds.standard_train_fold,
        mmdatasdk.cmu_mosei.standard_folds.standard_valid_fold,
        mmdatasdk.cmu_mosei.standard_folds.standard_test_fold
    ]
    tensors = cmu["high"].get_tensors(
        seq_len=50,
        non_sequences=["All Labels"],
        direction=False,
        folds=folds
    )

    # 5) Report shapes
    fold_names = ["train", "valid", "test"]
    for i, name in enumerate(fold_names):
        for cs in cmu["high"].computational_sequences:
            print(f"{cs:<15} | {name:5} | {tensors[i][cs].shape}")


if __name__ == "__main__":
    print("Processing with BERT word embeddings …")
    process_data(use_bert=True)
    log.success("Done: word‐aligned + BERT features generated.")

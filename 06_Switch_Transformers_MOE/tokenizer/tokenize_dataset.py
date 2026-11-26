import os
import json
import numpy as np
from datasets import load_dataset
from tqdm import tqdm

from utils.divide_dataset import divide_dataset_train_val_test
from tokenizer.hugging_face_tokenizer_wrapper import HFTokenizerWrapper

TOKENIZER_PATH = "tokenizer/tokenizer.json"


def write_bin(ds, wrapper, path, batch_size=10_000):
    """Tokenize every document, add EOS after each, append everything to one uint16 file."""
    total_tokens = 0
    with open(path, "wb") as f:
        for i in tqdm(range(0, len(ds), batch_size), desc=os.path.basename(path)):
            texts = ds[i : i + batch_size]["text"]
            for ids in wrapper.encode_batch(texts):
                ids.append(wrapper.eos_id)
                np.array(ids, dtype=np.uint16).tofile(f)
                total_tokens += len(ids)
    return total_tokens


if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    wrapper = HFTokenizerWrapper(path=TOKENIZER_PATH)

    data_set_c4 = load_dataset("allenai/c4", "realnewslike")
    train_ds, val_ds, test_ds = divide_dataset_train_val_test(data_set_c4, test_size=0.5, seed=42)

    meta = {"tokenizer": TOKENIZER_PATH, "dtype": "uint16", "split_seed": 42}
    for name, ds in [("val", val_ds), ("test", test_ds), ("train", train_ds)]:
        n_tokens = write_bin(ds, wrapper, f"data/{name}.bin")
        meta[name] = {"documents": len(ds), "tokens": n_tokens}
        print(name, meta[name])

    with open("data/meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
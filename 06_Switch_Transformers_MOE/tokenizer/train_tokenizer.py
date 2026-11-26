from datasets import load_dataset
from tqdm import tqdm

from tokenizer.hugging_face_tokenizer import train_hugging_face_tokenizer


def batch_iterator(dataset, batch_size=1000):
    with tqdm(total=len(dataset), desc="Reading documents", unit="doc", unit_scale=True) as pbar:
        for i in range(0, len(dataset), batch_size):
            batch = dataset[i : i + batch_size]["text"]
            pbar.update(len(batch))
            yield batch


if __name__ == "__main__":
    train = load_dataset("allenai/c4", "realnewslike", split="train")   # already cached: loads fast
    train_hugging_face_tokenizer(
        batch_iterator(train), vocab_size=8192, path="tokenizer/tokenizer.json")
    print("Saved tokenizer/tokenizer.json")
import os
import string
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders

SPECIALS = ["[PAD]", "[UNK]", "[EOS]"] + [f"<extra_id_{i}>" for i in range(100)]

def train_hugging_face_tokenizer(text, vocab_size=8192, path="tokenizer/tokenizer.json"):
    tokenizer = Tokenizer(models.BPE(unk_token="[UNK]", end_of_word_suffix="[END_WORD]"))
    tokenizer.pre_tokenizer = pre_tokenizers.WhitespaceSplit()          
    tokenizer.decoder = decoders.BPEDecoder(suffix="[END_WORD]")
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=SPECIALS,
        end_of_word_suffix="[END_WORD]",
        initial_alphabet=list(string.ascii_letters + string.digits + string.punctuation),  
        limit_alphabet=1000,                                          
    )
    tokenizer.train_from_iterator(text, trainer=trainer)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tokenizer.save(path)
    return tokenizer
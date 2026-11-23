from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders


class HFTokenizerWrapper:
    def __init__(self, path):
        self.tokinzer = Tokenizer.from_file(path)
        self.vocab_size = self.tokinzer.get_vocab_size()
        self.pad_id = self.tokinzer.token_to_id("[PAD]")
        self.unk_id = self.tokinzer.token_to_id("[UNK]")
        self.eos_id = self.tokinzer.token_to_id("[EOS]")
        self.sentinel_ids = [self.tokinzer.token_to_id(f"<extra_id_{i}>") for i in range(100)]
        self.special_ids = {self.pad_id, self.unk_id, self.eos_id, *self.sentinel_ids}

    def encode(self, text: str) -> list[int]:
        return self.tokinzer.encode(text).ids
    
    def encode_batch(self, texts: list[str]) -> list[list[int]]:
        return [encode.ids for encode in self.tokinzer.encode_batch(texts)]
    
    def decode(self, tokens_ids : list[int]) -> str:
        words, run = [], []

        for token_id in tokens_ids:
            if token_id in self.special_ids:
                if run:
                    words.append(self.tokinzer.decode(run))
                    run = []
                if token_id != self.pad_id:
                    words.append(self.tokinzer.id_to_token(token_id))
            else:
                run.append(token_id)

        if run:
            words.append(self.tokinzer.decode(run))
        
        return " ".join(word for word in words if word)
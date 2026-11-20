from collections import Counter
import tqdm
import string


class BPETokenizer:
    """
    Attributes:
        vocabulary: List of subword tokens including special tokens.
        vocabulary_size : Total number of tokens in vocabulary.
        token_to_index: Mapping from tokens to indices.
        index_to_token: Mapping from indices to tokens.
        pad_token_id: Index of the padding token.
        unknown_token_id: Index of the unknown token.
        tokenized_corpus: Cached tokenized corpus after BPE training.
    """
    UNKNOWN_TOKEN = "[UNK]"
    PAD_TOKEN = "[PAD]"
    SENTINEL_TOKENS = [f"<extra_id_{i}>" for i in range(0,100)]
    END_WORD_TOKEN = "[END_WORD]"
    EOS_TOKEN = "[EOS]"

    def __init__(
        self, 
        texts : list[str] | str,
        vocabulary: list[str] | None = None,
        num_merges : int = 100,
        ) -> None:
        """Inialize the BPEWordTokenizer

        Args:
            texts: A list of strings or a string representing the text corpus.
            vocabulary: Optional list of strings with unique tokens.
            num_merges: Defines how many rounds of merges should be performed
            when learning the BPE merges..
        """

        if isinstance(texts, str):
            texts = [texts]

        if vocabulary is None:
            self.merges, tokenized_corpus, vocabulary_set = self.learn_bpe(texts, num_merges)
            self.tokenized_corpus = tokenized_corpus

            #Ensuring that the basic alphanumeric characters are always included in the vocabulary
            required_chars = set(
                string.ascii_letters + string.ascii_uppercase + string.digits
            )

            vocabulary_set.update(required_chars)

            self.vocabulary = (
                [self.PAD_TOKEN, self.UNKNOWN_TOKEN, *self.SENTINEL_TOKENS, self.END_WORD_TOKEN, self.EOS_TOKEN]
                + sorted(vocabulary_set))
        else:
            self.vocabulary = vocabulary
            self.merges = [] #When there is a vocabulary not necesary
        
        #Building the mapping and the vocabulary size
        self.vocabulary_size = len(self.vocabulary)
        self.token_to_index = {token : index for index, token in enumerate(self.vocabulary)}
        self.index_to_token = {index : token for index, token in enumerate(self.vocabulary)}
        self.pad_token = self.token_to_index[self.PAD_TOKEN]
        self.unknown_token = self.token_to_index[self.UNKNOWN_TOKEN]
        self.sentinels_token = [self.token_to_index[sentinel] for sentinel in self.SENTINEL_TOKENS] 
        self.end_word_token = self.token_to_index[self.END_WORD_TOKEN]
        self.eos_token = self.token_to_index[self.EOS_TOKEN]
    def split_text(self, text : str) -> list[str]:
        """
        Split a string into a subword of tokens using a learning BPE merges
        """
        tokens = []
        for word in text.strip().split():
            #Split the string into characters and add sepcial END_WORD token "</w>"
            characters = list(word) + [self.END_WORD_TOKEN]

            # Merge individual characters according to learned BPE merges
            for pair in self.merges:
                characters = self.merge_pair_in_words(characters, pair)
            tokens.extend(characters)
        
        return tokens

    def join_text(self, tokens : list[str]) -> str:
        # Join subwords tokens into full string
        words = []
        current_word = []
        for token in tokens:
            if token.endswith(self.END_WORD_TOKEN): #Check if finish with the END_WORD token "</w>"
                current_word.append(token.replace(self.END_WORD_TOKEN, "")) #Removing the end of the word marker
                words.append("".join(current_word))
                current_word = []
            else:
                current_word.append(token)

        if current_word: #If there are any remaining tokens that do not end with the end of word marker, join them as a single word
            words.append("".join(current_word))
        return " ".join(words).strip()

    def encoder(self, text : str) -> list[int]:
        """
        Encode a string into a list of tokens indices
        """
        tokens_ids = []
        for token in self.split_text(text=text):
            token_id = self.token_to_index.get(token, self.unknown_token)
            tokens_ids.append(token_id)
        return tokens_ids

    def decoder(self, tokens_ids : int | list[int]) -> str:
        """
        Decode a list of tokens IDs back to the original text
        """
        if isinstance(tokens_ids, int):
            tokens_ids = [tokens_ids]

        tokens = []
        for token_id in tokens_ids:
            if token_id in [self.pad_token]:
                continue
            tokens.append(self.index_to_token.get(token_id, self.UNKNOWN_TOKEN))

        return self.join_text(tokens)

#  Construct a tokenizer from a given vocabulary, list of merges, and (optionally) a list of special tokens. 
#  This function should accept 3See en.wikipedia.org/wiki/Specials_(Unicode_block)
#  Replacement_character for more information about the Unicode replacement character.
#  vocab: dict[int, bytes]
#  merges: list[tuple[bytes, bytes]]
#  special_tokens: list[str] | None = None
import regex as re
from typing import Iterable, Iterator

class MyBPETokenizer:
    def __init__(self, vocab, merges, special_tokens=None):
        self.id_to_token=vocab
        self.token_to_id={v: k for k,v in vocab.items()}
        self.merge_ranks={pair: i for i,pair in enumerate(merges)} # 数字越低合并优先级越高
        self.special_tokens = special_tokens or []
        self.pattern=r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

    # Class method that constructs and return a Tokenizer from a serialized vocabulary and list of merges
    # (in the same format that your BPE training code output) and (optionally) a list of special
    # tokens. This method should accept the following additional parameters:
    # vocab_filepath: str
    # merges_filepath: str
    # special_tokens: list[str] | None = None
    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        import json

        # 读取 vocab.json
        with open(vocab_filepath, "r", encoding="utf-8") as f:
            raw_vocab = json.load(f)

        # 将 token（str）转为 bytes
        vocab = {int(v): k.encode("utf-8") for k, v in raw_vocab.items()}

        # 读取 merges.txt
        merges = []
        with open(merges_filepath, "r", encoding="utf-8") as f:
            lines = f.readlines()
            if lines[0].startswith("#"):
                lines = lines[1:]  # 跳过 '#version: 0.2'

            for line in lines:
                if line.strip():
                    t1, t2 = line.strip().split()
                    b1 = t1.encode("utf-8")
                    b2 = t2.encode("utf-8")
                    merges.append((b1, b2))

        return cls(vocab, merges, special_tokens)

    def bpe(self, token: str) -> list[str]:
        word = list(token)
        while True:
            pairs = {(word[i], word[i + 1]) for i in range(len(word) - 1)}
            mergeable = pairs & self.merge_ranks.keys()
            if not mergeable:
                break
            best = min(mergeable, key=lambda p: self.merge_ranks[p])
            i = 0
            new_word = []
            while i < len(word):
                if i < len(word) - 1 and (word[i], word[i + 1]) == best:
                    new_word.append(word[i] + word[i + 1])
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            word = new_word
        return word

    # Encode an input text into a sequence of token IDs.
    def encode(self, text: str) -> list[int]:
        special_token_regex = "|".join(re.escape(tok) for tok in self.special_tokens)
        pieces = re.split(f"({special_token_regex})", text)
        ids = []
        for piece in pieces:
            if not piece:
                continue
            if piece in self.special_tokens:
                ids.append(self.token_to_id[piece])
            else:
                for match in re.finditer(self.pattern, piece):
                    word = match.group(0)
                    bpe_tokens = self.bpe(word)
                    for token in bpe_tokens:
                        ids.append(self.token_to_id.get(token, self.token_to_id.get("<unk>", -1)))
        return ids
        

    # Given an iterable of strings (e.g., a Python file handle), return a generator that lazily yields token IDs. 
    # This is required for memory-efficient tokenization of large files that we cannot directly load into memory.
    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for line in iterable:
            yield from self.encode(line)

    # Decode a sequence of token IDs into text.
    def decode(self, ids: list[int]) -> str:
        return "".join(self.id_to_token[i] for i in ids)
import os
import regex as re
import time
import multiprocessing

from collections import defaultdict
from typing import BinaryIO
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

### ====== Part 1: 分段工具 ====== ###
def find_chunk_boundaries(
    file: BinaryIO, 
    desired_num_chunks: int, 
    split_special_token: bytes
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), (
        "Must represent special token as a bytestring"
    )

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))

### ====== Part 3: pre 分词函数 ====== ###
def update_token_freqs(text, token_freqs, pattern, special_token_bytes, tuple_cache):
    special_tokens_str = [tok.decode("utf-8") for tok in special_token_bytes]
    special_token_regex = "|".join(re.escape(tok) for tok in special_tokens_str)
    pieces = re.split(f"({special_token_regex})", text)

    for piece in pieces:
        if piece in special_tokens_str:
            token = piece.encode("utf-8")
            token_freqs[(token,)] += 1
        else:
            for match in re.finditer(pattern, piece):
                word = match.group(0).encode("utf-8")
                # 使用缓存
                if word in tuple_cache:
                    tup = tuple_cache[word]
                else:
                    tup = tuple(bytes([b]) for b in word)
                    tuple_cache[word] = tup
                token_freqs[tup] += 1



### ====== Part 4: Merge 函数 ====== ###
# 得到pair分词
def get_pairs(token_freqs):
    pair_freqs = defaultdict(int)
    for token, freq in token_freqs.items():
        for i in range(len(token) - 1):
            pair = (token[i], token[i + 1])
            pair_freqs[pair] += freq
    return pair_freqs

# 合并pair分词
def merge_pairs(best_pair, token_freqs):
    new_token_freqs = {}
    for word, freq in token_freqs.items():
        new_token = []
        i = 0
        while i < len(word):
            if i < len(word) - 1 and (word[i], word[i + 1]) == best_pair:
                new_token.append(word[i] + word[i + 1])
                i += 2
            else:
                new_token.append(word[i])
                i += 1
        new_token_freqs[tuple(new_token)] = freq
    return new_token_freqs

# chunk-level多进程
def process_chunk(args):
    start, end, file_path, special_token_bytes = args
    local_freqs = defaultdict(int)
    tuple_cache = {}  # 本地缓存
    with open(file_path, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")
        update_token_freqs(chunk, local_freqs, PAT, special_token_bytes, tuple_cache)
    return local_freqs


# ----------- 主函数 train_bpe ------------ #

def train_bpe(input_path: str, vocab_size: int, special_tokens: list[str]) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    # 初始 vocab: token(bytes) -> id(int)
    token_to_id = {}
    id_to_token = {}

    special_token_bytes = [tok.encode("utf-8") for tok in special_tokens]
    
    for idx, tok in enumerate(special_token_bytes):
        token_to_id[tok] = idx
        id_to_token[idx] = tok
    vocab_idx = len(special_token_bytes)

    for i in range(256):
        byte_tok = bytes([i])
        if byte_tok not in token_to_id:
            token_to_id[byte_tok] = vocab_idx
            id_to_token[vocab_idx] = byte_tok
            vocab_idx += 1

    num_merges = vocab_size - len(token_to_id)
    merges = []

    multiprocessing.freeze_support()
    file_path = input_path
    num_processes = min(8, cpu_count())

    with open(file_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_processes, special_token_bytes[0])

    tasks = [(start, end, file_path, special_token_bytes) for start, end in zip(boundaries[:-1], boundaries[1:])]
    with Pool(processes=num_processes) as pool:
        all_freqs = list(tqdm(pool.imap(process_chunk, tasks), total=len(tasks), desc="Parallel Pre-tokenizing"))

    token_freqs = defaultdict(int)
    for partial_freq in all_freqs:
        for token, freq in partial_freq.items():
            token_freqs[token] += freq

    for _ in tqdm(range(num_merges), desc="Merging BPE pairs"):
        pair_freqs = defaultdict(int)
        for token, freq in token_freqs.items():
            for i in range(len(token) - 1):
                pair = (token[i], token[i + 1])
                # 跳过含有特殊 token 的组合
                if pair[0] in special_token_bytes or pair[1] in special_token_bytes:
                    continue
                pair_freqs[pair] += freq

        if not pair_freqs:
            break
        max_freq = max(pair_freqs.values())
        candidates = [pair for pair, freq in pair_freqs.items() if freq == max_freq]
        best_pair = max(candidates)  # 词典序最大

        merges.append(best_pair)
        token_freqs = merge_pairs(best_pair, token_freqs)

        # 将新的 token 加入 vocab
        merged_token = best_pair[0] + best_pair[1]
        if merged_token not in token_to_id:
            token_to_id[merged_token] = vocab_idx
            id_to_token[vocab_idx] = merged_token
            vocab_idx += 1

    return id_to_token, merges

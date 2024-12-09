import pickle
import regex
from logger import Logger
from collections import Counter

class BPETokenizer():
    def __init__(self, text, vocab_size=276, logger=None):
        if logger is not None:
            self.logger = logger
            logger.info(f"Tokenizing text of length {len(text)} with vocab size {vocab_size}")
        else:
            self.logger = None
        self.pat = regex.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

        if len(text) > 100000000:
            text = text[:len(text)//10] # for big text
            if logger is not None:
                logger.info(f"truncated text to {len(text)}")
            
        tokens = text.encode("utf-8")
        tokens = list(map(int, tokens))
        self.vocab = {idx: bytes([idx]) for idx in range(256)}

        self.tokens, self.merges = self.compress(text, vocab_size=vocab_size)

        self.old_token_length = len(tokens)
        self.new_token_length = len(self.tokens)
        self.vocab_size = len(self.vocab)
        self.compression_rate = len(self.tokens)/len(tokens)

        self.save()

    def set_logger(self, logger):
        self.logger = logger

    def load_from_file(path):
        with open(path, 'rb') as f:
            tokenizer = pickle.load(f)

        return tokenizer
    
    def get_vocab_size(self):
        return self.vocab_size

    def visualzie_merge(self):
        merged_visual = ""
        for i, (p0, p1) in enumerate(self.merges):
            merged_visual += f"{i} {self.vocab[p0] + self.vocab[p1]}\n"

        return merged_visual

    def get_tokenizer_stats(self):
        merged_visual = self.visualzie_merge()
        stats = "\n"
        stats += f"===================TOKENIZATION STATS===================\n"
        stats += f"token length {self.old_token_length}\n"
        stats += f"new token length {self.new_token_length}\n"
        stats += f"compression rate {self.compression_rate}\n"
        stats += f"========================================================\n"
        stats += f"===================MERGED TOKENS===================\n"
        stats += merged_visual
        stats += f"========================================================\n"
        return stats
    
    def get_stats(self, token_list):
        pair_counts = Counter()
        for ids in token_list:
            pair_counts.update(zip(ids, ids[1:]))
        return pair_counts

    def get_stats_encode(self, ids):
        pair_counts = Counter(zip(ids, ids[1:]))
        return pair_counts

    def merge_encode(self, ids, pair, idx):
        new_ids = []
        i = 0

        while i < len(ids):
            if i < len(ids) - 1 and ids[i] == pair[0] and ids[i+1] == pair[1]:
                new_ids.append(idx)
                i += 2
            else:
                new_ids.append(ids[i])
                i += 1

        return new_ids

    def merge(self, token_list, pair, idx):
        new_token_list = []
        
        for ids in token_list:
            new_ids = []
            i = 0
            while i < len(ids):
                if i < len(ids) - 1 and ids[i] == pair[0] and ids[i+1] == pair[1]:
                    new_ids.append(idx)
                    i += 2
                else:
                    new_ids.append(ids[i])
                    i += 1
            new_token_list.append(new_ids)

        return new_token_list

    def compress(self, text, vocab_size=276):

        token_list = regex.findall(self.pat, text)
        for i in range(len(token_list)):
            token_list[i] = token_list[i].encode("utf-8")
            token_list[i] = list(map(int, token_list[i]))

        num_merges = vocab_size - 256
        merges = {}
        for i in range(num_merges):
            stats = self.get_stats(token_list)
            if stats == {}: break
            top_pair = max(stats, key=stats.get)
            new_index = 256 + i
            merges[top_pair] = new_index

            self.vocab[new_index] = self.vocab[top_pair[0]] + self.vocab[top_pair[1]]
            token_list = self.merge(token_list, top_pair, new_index)

            if self.logger is not None:
                top_pair_text = self.decode(top_pair)
                self.logger.info(f"merged {top_pair} to {new_index}, which is '{top_pair_text}'")

        tokens = []
        for ids in token_list:
            tokens.extend(ids)
        return tokens, merges

    def decode(self, ids):
        tokens =  b"".join([self.vocab[i] for i in ids])
        text = tokens.decode("utf-8", errors="replace")
        return text

    def encode(self, text):
        tokens = list(text.encode("utf-8"))
        while len(tokens) >= 2:
            stats = self.get_stats_encode(tokens)
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges: break
            idx = self.merges[pair]
            tokens = self.merge_encode(tokens, pair, idx)
        return tokens

    def save(self):
        with open('models/bpetokenizer.pkl', 'wb') as f:
            pickle.dump(self, f)
        if self.logger is not None:
            self.logger.info("saved tokenizer to models/bpetokenizer.pkl")

if __name__ == "__main__":

    with open('dataset/oscar_input.txt', 'r') as f:
        text = f.read()
    logger = Logger('logs/simplelm.log', 'SimpleLM')
    vocab_size = 1000
    bpetokenizer = BPETokenizer(text=text, vocab_size=vocab_size, logger=logger)
    print(bpetokenizer.get_tokenizer_stats())
    print(bpetokenizer.encode("hello world"))
    print(bpetokenizer.decode(bpetokenizer.encode("hello world")))

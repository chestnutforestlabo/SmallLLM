import torch
import torch.nn as nn
import torch.nn.functional as F
from tokenizer import BPETokenizer
from simplellm_comfig import SimpleLMConfig
from logger import Logger
from datasets import load_dataset
import json

class Head(nn.Module):
    def __init__(self, head_size, n_embd, context_length, dropout):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(context_length, context_length)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x)   # (B,T,hs)
        q = self.query(x) # (B,T,hs)
        wei = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5 # (B, T, hs) @ (B, hs, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x) # (B,T,hs)
        out = wei @ v # (B, T, T) @ (B, T, hs) -> (B, T, hs)
        return out

# decoder version of MultiHeadAttention
class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size, n_embd, dropout, context_length):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size, n_embd, context_length, dropout) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedForward(nn.Module):

    def __init__(self, n_embed, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed),
            nn.ReLU(),
            nn.Linear(4 * n_embed, n_embed),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, n_embed, n_head, dropout, context_length):
        super().__init__()
        head_size = n_embed // n_head
        self.attn = MultiHeadAttention(n_head, head_size, n_embed, dropout, context_length)
        self.ffwd = FeedForward(n_embed, dropout)
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)

    def forward(self,x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class SimpleLM(nn.Module):

    def __init__(self, vocab_size, config, bpetokenizer, logger=None):
        super().__init__()
        self.logger = logger

        config_summary = config.get_config_summary()
        
        tokenizer_stats = bpetokenizer.get_tokenizer_stats()
        if self.logger is not None:
            self.logger.info(config_summary)
            self.logger.info(tokenizer_stats)

        self.config = config
        self.n_embd = config.n_embd
        self.context_length = config.context_length
        self.n_layer = config.n_layer
        self.n_head = config.n_head
        self.dropout = config.dropout
        self.device = config.device
        self.learning_rate = config.learning_rate
        self.max_iters = config.max_iters
        self.eval_interval = config.eval_interval
        self.eval_iters = config.eval_iters
        self.batch_size = config.batch_size
        self.save_interval = config.save_interval

        self.margin_start = 10 * self.context_length
        self.margin_end = 5 * self.context_length  # Define margin_end before using it

        self.token_embedding_table = nn.Embedding(vocab_size, self.n_embd)
        self.position_embedding_table = nn.Embedding(self.context_length, self.n_embd)
        self.blocks = nn.Sequential(*[Block(self.n_embd, self.n_head, self.dropout, self.context_length) for _ in range(self.n_layer)])
        self.ln_f = nn.LayerNorm(self.n_embd)
        self.lm_head = nn.Linear(self.n_embd, vocab_size)

        self.loss_log = {'train': [], 'val': []}

    def forward(self, idx, targets=None):
        B, T= idx.shape
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device)) # (T,C)
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x) # (B,T,V)

        if targets is None:
            return logits, None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets) 
            return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.context_length:]
            logits, loss = self(idx_cond)
            logits = logits[:,-1,:]
            probs = torch.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, idx_next], dim=-1)
        return idx

    @torch.no_grad()
    def estimate_loss(self, train_data, valid_data):
        out = {}
        self.eval()
        for split in ['train', 'val']:
            losses = torch.zeros(self.eval_iters)
            for k in range(self.eval_iters):
                x, y = self.get_batch(train_data, valid_data, split)
                _, loss = self(x, y)
                losses[k] = loss.item()
            out[split] = losses.mean()
            self.loss_log[split].append(out[split].item())
        self.train()
        self.visualize_output()
        self.save_loss_log()
        return out

    def get_batch(self, train_data, valid_data, split):
        data = train_data if split == 'train' else valid_data
        if len(data) < self.context_length + 1:  # Ensure there's enough data for at least one full batch
            # This repeats the data to at least match context_length + 1
            data = data.repeat((self.context_length + 1 + len(data) - 1) // len(data))
        ix = torch.randint(len(data) - self.context_length, (self.batch_size,))  # Safe indexing
        x = torch.stack([data[i:i+self.context_length] for i in ix])
        y = torch.stack([data[i+1:i+self.context_length+1] for i in ix])
        x = x.to(self.device)
        y = y.to(self.device)
        return x, y

    def train_model(self):
        dataset = self.load_dataset_hf()
        optimizer = torch.optim.Adam(self.parameters(), lr=config.learning_rate)

        for iter in range(config.max_iters):

            token_sample = []
            while len(token_sample)*0.9 < self.context_length * 2:
                try:
                    text = next(dataset)["text"]
                    token_sample = bpetokenizer.encode(text)
                except StopIteration:
                    dataset = self.load_dataset_hf()
                    text = next(dataset)["text"]
                    token_sample = bpetokenizer.encode(text)

            train_data = torch.tensor(token_sample[:int(len(token_sample)*0.9)], dtype=torch.long)
            valid_data = torch.tensor(token_sample[int(len(token_sample)*0.9):], dtype=torch.long)

            if iter % self.eval_interval == 0:
                losses = self.estimate_loss(train_data, valid_data)
                self.logger.info(f'iter {iter}: train loss {losses["train"]}, val loss {losses["val"]}')

            if iter % self.save_interval == 0 and iter > 0:
                torch.save(self.state_dict(), f'models/model_{iter}.pth')

            xb, yb = self.get_batch(train_data, valid_data, 'train')
            logits, loss = self(xb, yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            
        if self.logger is not None:
            self.logger.info(f'iter {iter}: train loss {losses["train"]}, val loss {losses["val"]}')    
        torch.save(self.state_dict(), f'models/model_{iter+1}.pth')

    def load_dataset_hf(self):
        logger.info('Loading the text')
        dataset = load_dataset("oscar-corpus/OSCAR-2301",
                            use_auth_token=True,
                            language="en",
                            streaming=True,
                            split="train")
        dataset = iter(dataset)
        return dataset

    def visualize_output(self):
        self.eval()
        input_text = "I am a researcher and"
        idx = bpetokenizer.encode(input_text)
        idx = torch.tensor(idx, dtype=torch.long).unsqueeze(0).to(self.device)
        genenrated_tokens = model.generate(idx, max_new_tokens=500)[0].tolist()
        decoded_text = bpetokenizer.decode(genenrated_tokens)
        self.logger.info("The generated text is: " + decoded_text)
        self.train()

    def save_loss_log(self):
        with open('logs/loss_log.json', 'w') as f:
            json.dump(self.loss_log, f, indent=4)

if __name__ == '__main__':
    logger = Logger('logs/simplelm.log', 'SimpleLM')

    ## Preparte model and tokenizer
    logger.info('Preparing the model and tokenizer')
    config = SimpleLMConfig()
    bpetokenizer: BPETokenizer = BPETokenizer.load_from_file('models/bpetokenizer.pkl')
    bpetokenizer.set_logger(logger)
    vocab_size = bpetokenizer.get_vocab_size()

    ## Train the model
    logger.info('Training the model')
    model = SimpleLM(vocab_size, config, bpetokenizer, logger).to(config.device)
    logger.info(str(sum(p.numel() for p in model.parameters())/1e6) + 'M parameters')
    model.train_model()

    ## Test the model
    logger.info('Testing the model')
    model.visualize_output()
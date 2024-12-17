import torch
from tokenizer import BPETokenizer
from src.simplellm_config import SimpleLMConfig
from gpt import SimpleLM


def generate(model, idx, max_new_tokens, config, tokenizer):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -config.block_size:]
        logits, loss = model(idx_cond)
        logits = logits[:,-1,:]
        probs = torch.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)
        idx = torch.cat([idx, idx_next], dim=-1)
        print(tokenizer.decode(idx_next[0].tolist()), end="")

    return idx

def interact_with_simplelm(args):

    config = SimpleLMConfig()
    bpetokenizer: BPETokenizer = BPETokenizer.load_from_file('models/bpetokenizer.pkl')
    vocal_size = bpetokenizer.get_vocab_size()
    model = SimpleLM(vocal_size, config, bpetokenizer, None)
    model.load_state_dict(torch.load(args.model))
    model.eval()
    while True:
        input_text = input("\n\nEnter the text: \n\n")
        input_text = bpetokenizer.encode(input_text)
        input_text = torch.tensor(input_text, dtype=torch.long).unsqueeze(0)
        _ = generate(model, input_text, 500, config, bpetokenizer)[0].tolist()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='models/model_500.pth')
    args = parser.parse_args()

    interact_with_simplelm(args)
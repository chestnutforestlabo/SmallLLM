import torch
class SimpleLMConfig:
    def __init__(self):
        self.batch_size = 32
        self.context_length = 516
        self.max_iters = 500000
        self.eval_interval = 100
        self.save_interval = 5000
        self.learning_rate = 3e-4
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.eval_iters = 10
        self.n_embd = 768
        self.n_layer = 12
        self.n_head = 12
        self.dropout = 0.2

    def get_config_summary(self):
        config_summary = f"\nConfiguration Summary:\n"
        config_summary += f"  Batch Size: {self.batch_size}\n"
        config_summary += f"  Block Size: {self.context_length}\n"
        config_summary += f"  Max Iterations: {self.max_iters}\n"
        config_summary += f"  Evaluation Interval: {self.eval_interval}\n"
        config_summary += f"  Learning Rate: {self.learning_rate}\n"
        config_summary += f"  Device: {self.device}\n"
        config_summary += f"  Evaluation Iterations: {self.eval_iters}\n"
        config_summary += f"  Embedding Dimension: {self.n_embd}\n"
        config_summary += f"  Number of Layers: {self.n_layer}\n"
        config_summary += f"  Number of Attention Heads: {self.n_head}\n"
        config_summary += f"  Dropout: {self.dropout}"
        return config_summary
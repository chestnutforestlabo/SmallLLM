
class ViTConfig():
    def __init__(self):
        self.device = 'cpu'
        self.patch_size = 4
        self.n_embd = 768
        self.n_channels = 3
        self.n_heads = 12
        self.n_layer = 4
        self.dropout = 0.2
        self.num_classes = 10
        self.size = 32
        self.epochs = 1000000
        self.base_lr = 10e-4
        self.weight_decay = 0.03
        self.batch_size = 4

        self.eval_iters = 5
        self.eval_interval = 100
        self.save_interval = 500

    def get_config_summary(self):
        config_summary = f"\nConfiguration Summary:\n"
        config_summary += f"  Device: {self.device}\n"
        config_summary += f"  Patch Size: {self.patch_size}\n"
        config_summary += f"  Embedding Dimension: {self.n_embd}\n"
        config_summary += f"  Number of Channels: {self.n_channels}\n"
        config_summary += f"  Number of Heads: {self.n_heads}\n"
        config_summary += f"  Number of Layers: {self.n_layer}\n"
        config_summary += f"  Dropout: {self.dropout}\n"
        config_summary += f"  Number of Classes: {self.num_classes}\n"
        config_summary += f"  Size: {self.size}\n"
        config_summary += f"  Epochs: {self.epochs}\n"
        config_summary += f"  Base Learning Rate: {self.base_lr}\n"
        config_summary += f"  Weight Decay: {self.weight_decay}\n"
        config_summary += f"  Batch Size: {self.batch_size}"
        return config_summary
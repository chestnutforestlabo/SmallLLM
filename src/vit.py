import torch
from torch import nn
import einops
import torchvision
import torch.optim as optim
from torchvision.transforms import Compose, Resize, ToTensor, Normalize, RandomHorizontalFlip, RandomVerticalFlip, RandomCrop
from vit_config import ViTConfig
from logger import Logger
from datasets import load_dataset

def expand_grayscale_to_rgb(img):
    if img.size(0) == 1:  # Check if the image has only one channel
        img = img.repeat(3, 1, 1)  # Repeat the channel to make it 3-channel
    return img

def load_dataset_hf(split):
    assert split in ['train', 'test']
    dataset = load_dataset("cifar10", # other options: "imagenet-1k"
                        # use_auth_token=True,
                        streaming=True)
    dataset = iter(dataset[split])
    return dataset

class InputEmbedding(nn.Module):
    def __init__(self, patch_size, n_channels, n_embd, batch_size, device) -> None:
        super(InputEmbedding, self).__init__()
        self.n_embd = n_embd
        self.patch_size = patch_size
        self.n_channels = n_channels
        self.device = device
        self.batch_size = batch_size
        self.input_size = self.patch_size * self.patch_size * self.n_channels

        self.linearProjection = nn.Linear(self.input_size, self.n_embd)
        self.class_token = nn.Parameter(torch.randn(1, 1, self.n_embd)).to(self.device)
        num_patches = (config.size // config.patch_size) * (config.size // config.patch_size)
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches+1, self.n_embd)).to(self.device)

    def forward(self, input_data):

        input_data = input_data.to(self.device)

        # Re-arrange image into patches.
        patches = einops.rearrange(
            input_data, 'b c (h h1) (w w1) -> b (h w) (h1 w1 c)', h1=self.patch_size, w1=self.patch_size)

        linear_projection = self.linearProjection(patches).to(self.device)
        b, n, _ = linear_projection.shape

        class_tokens = self.class_token.expand(b, -1, -1)  # Expand class token for batch
        linear_projection = torch.cat((class_tokens, linear_projection), dim=1)
        linear_projection = linear_projection + self.pos_embedding[:, :n+1, :]

        return linear_projection


class EncoderBlock(nn.Module):
    def __init__(self, n_embd, n_heads, dropout, device):
        super(EncoderBlock, self).__init__()
        self.n_embd = n_embd
        self.n_heads = n_heads
        self.dropout = dropout
        self.device = device

        self.norm = nn.LayerNorm(self.n_embd)
        self.multi_head_attention = nn.MultiheadAttention(self.n_embd, self.n_heads, self.dropout).to(self.device)

        self.enc_MLP = nn.Sequential(
            nn.Linear(self.n_embd, 4*self.n_embd),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(4*self.n_embd, self.n_embd),
            nn.Dropout(self.dropout)
        )

    def forward(self, embedded_patches):

        firstNorm_out = self.norm(embedded_patches)
        attention_output = self.multi_head_attention(firstNorm_out, firstNorm_out, firstNorm_out)[0]

        # First residual connection
        first_added_output = attention_output + embedded_patches

        # Second sublayer: Norm + enc_MLP (Feed forward)
        secondNorm_out = self.norm(first_added_output)
        ff_output = self.enc_MLP(secondNorm_out)

        # Return the output of the second residual connection
        return ff_output + first_added_output

class VisionTransformer(nn.Module):
    def __init__(self, config, logger) -> None:
        super(VisionTransformer, self).__init__()
        self.n_layer = config.n_layer
        self.n_embd = config.n_embd
        self.num_classes = config.num_classes
        self.dropout = config.dropout
        self.device = config.device
        self.dropout = config.dropout
        self.patch_size = config.patch_size
        self.n_channels = config.n_channels
        self.batch_size = config.batch_size
        self.n_heads = config.n_heads
        self.eval_batch_size = config.eval_batch_size
        self.eval_interval = config.eval_interval
        self.save_interval = config.save_interval
        self.logger = logger
        self.loss_log = {'train': [], 'val': []}

        self.training_transform_pipeline = Compose([
            Resize((config.size)), 
            RandomHorizontalFlip(), 
            ToTensor(), 
            expand_grayscale_to_rgb,  
            Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])

        self.testing_transform_pipeline = Compose([
            Resize(config.size),                       # Resize the image to 224x224 pixels
            ToTensor(),    
            expand_grayscale_to_rgb,                 
            Normalize(mean=[0.485, 0.456, 0.406],  # Normalize the tensor using the ImageNet mean
                    std=[0.229, 0.224, 0.225])   # and standard deviation values
        ])


        self.embedding = InputEmbedding(patch_size=self.patch_size, n_channels=self.n_channels, n_embd=self.n_embd, batch_size=self.batch_size, device=self.device).to(self.device)
        self.enc_stack = nn.ModuleList([EncoderBlock(n_embd=self.n_embd, n_heads=self.n_heads, dropout=self.dropout, device=self.device) for _ in range(self.n_layer)])

        self.MLP_head = nn.Sequential(
            nn.LayerNorm(self.n_embd),
            # nn.Linear(self.n_embd, self.n_embd),
            nn.Linear(self.n_embd, self.num_classes)
        )

        self.logger.info("Model initialized")

    def forward(self, test_input):

        enc_output = self.embedding(test_input)

        for enc_layer in self.enc_stack:
            enc_output = enc_layer.forward(enc_output)

        cls_token_embedding = enc_output[:, 0]

        return self.MLP_head(cls_token_embedding), cls_token_embedding

    @torch.no_grad()
    def test_model(self, dataset_test, criterion):
        logger.info('Testing the model')
        self.eval()
        images = []
        labels = []
        for _ in range(self.eval_batch_size):
            try:
                sample = next(dataset_test)
            except StopIteration:
                dataset_test = load_dataset_hf('train')
                sample = next(dataset_test)
            images.append(self.testing_transform_pipeline(sample['img']))
            labels.append(sample['label'])
        images = torch.stack(images).to(self.device)
        labels = torch.tensor(labels).to(self.device)

        classification, _ = self(images)
        loss = criterion(classification, labels)
        _, predicted = torch.max(classification, 1)
        correct = (predicted == labels).sum().item()
        accuracy = correct / self.eval_batch_size
        self.logger.info(f'Validation Loss: {loss.item()} Accuracy: {accuracy}')
        self.train()

    def train_model(self):
        dataset_train = load_dataset_hf('train')
        dataset_test = load_dataset_hf('test')
        self.train()
        self.logger.info('Training the model')
        optimizer = optim.Adam(self.parameters(), betas=(0.9, 0.999), lr=config.base_lr, weight_decay=config.weight_decay)
        scheduler = optim.lr_scheduler.LinearLR(optimizer)
        criterion = nn.CrossEntropyLoss()
        running_loss = 0.0

        for epoch in range(config.epochs):
            self.train()
            images = []
            labels = []
            for _ in range(self.batch_size):
                try:
                    sample = next(dataset_train)
                except StopIteration:
                    dataset_train = load_dataset_hf('train')
                    sample = next(dataset_train)
                images.append(self.training_transform_pipeline(sample['img']))
                labels.append(torch.tensor(sample['label']))

            images = torch.stack(images).to(self.device)
            labels = torch.stack(labels).to(self.device)

            classification, _ = self(images)
            loss = criterion(classification, labels)
            loss.backward()  # Accumulate gradients
            running_loss += loss.item()

            if epoch % self.eval_interval == 0 and epoch > 0:
                optimizer.step()  # Perform a single update
                scheduler.step()  # Scheduler update
                optimizer.zero_grad()  # Zero the gradients after updating
            
                self.test_model(dataset_test, criterion)
                self.logger.info(f'Epoch: {epoch} Running Loss: {running_loss/self.eval_interval}')
                running_loss = 0.0

        self.logger.info('Saving the model')
        torch.save(self.state_dict(), f'models/vit.pth')
        self.test_model(dataset_test, criterion)


if __name__ == '__main__':
    logger = Logger('logs/vit.log', 'vit')

    # Load the configuration
    logger.info('Loading the configuration')
    config = ViTConfig()
    model = VisionTransformer(config, logger).to(config.device)

    # train the model
    model.train_model()
    logger.info('Model training complete')


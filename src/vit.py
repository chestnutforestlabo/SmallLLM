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

class PatchEmbedding(nn.Module):
    def __init__(self, n_embd, patch_size, n_patches, dropout, n_channels) -> None:
        super(PatchEmbedding, self).__init__()
        self.n_embd = n_embd
        self.patch_size = patch_size
        self.n_patches = n_patches
        self.dropout = dropout

        self.patcher = nn.Sequential(
                    nn.Conv2d(in_channels=n_channels, 
                        out_channels=self.n_embd, 
                        kernel_size=self.patch_size, 
                        stride=self.patch_size),
                    nn.Flatten(2)
                    )
        self.cls_token = nn.Parameter(torch.randn(size=(1, 1, n_embd)), requires_grad=True)
        self.pos_embedding = nn.Parameter(torch.randn(size=(1, n_patches+1, n_embd)), requires_grad=True)
        self.dropout= nn.Dropout(self.dropout)

    def forward(self, x):
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        x = self.patcher(x).permute(0, 2, 1)
        x = torch.cat([cls_token, x], dim=1)
        x = x + self.pos_embedding
        x = self.dropout(x)
        return x

class ViT(nn.Module):
    def __init__(self, config, logger) -> None:
        super(ViT, self).__init__()
        n_embd = config.n_embd
        patch_size = config.patch_size
        n_heads = config.n_heads
        dropout = config.dropout
        n_channels = config.n_channels
        num_classes = config.num_classes
        n_layer = config.n_layer
        self.device = config.device
        input_dim = patch_size * patch_size * n_channels
        num_patches = (config.size // patch_size) ** 2
        self.batch_size = config.batch_size
        self.eval_iters = config.eval_iters
        self.eval_interval = config.eval_interval
        self.logger = logger
        self.config = config

        self.training_transform_pipeline = Compose([
            Resize((config.size)), 
            RandomHorizontalFlip(), 
            ToTensor(), 
            expand_grayscale_to_rgb,  
            Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])

        self.testing_transform_pipeline = Compose([
            Resize((config.size)),                       # Resize the image to 224x224 pixels
            ToTensor(),    
            expand_grayscale_to_rgb,                 
            Normalize(mean=[0.485, 0.456, 0.406],  # Normalize the tensor using the ImageNet mean
                    std=[0.229, 0.224, 0.225])   # and standard deviation values
        ])

        self.embeddings = PatchEmbedding(input_dim, patch_size, num_patches, dropout, n_channels)
        encoder_layer = nn.TransformerEncoderLayer(d_model=input_dim, nhead=n_heads, dim_feedforward=n_embd, dropout=dropout, activation= 'gelu', batch_first=True, norm_first=True)
        self.encoder_blocks = nn.TransformerEncoder(encoder_layer, num_layers=n_layer)

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(n_embd),
            nn.Linear(n_embd, num_classes)
        )

    def forward(self, x):
        x = self.embeddings(x)
        x = self.encoder_blocks(x)
        x = x[:, 0,:]
        classification = self.mlp_head(x)
        return classification, x

    def load_dataset_hf(self, split):
        assert split in ['train', 'test']
        self.logger.info(f'Loading the {split} image dataset')
        dataset = load_dataset("cifar10", # other options: "imagenet-1k"
                            use_auth_token=True,
                            streaming=True)
        dataset = iter(dataset[split])
        return dataset

    @torch.no_grad()
    def test_model(self, dataset_test, criterion):
        logger.info('Testing the model')
        self.eval()
        for _ in range(self.eval_iters):
            images = []
            labels = []
            for _ in range(self.batch_size):
                sample = next(dataset_test)
                images.append(self.testing_transform_pipeline(sample['img']))
                labels.append(sample['label'])
            images = torch.stack(images).to(self.device)
            labels = torch.tensor(labels).to(self.device)

            classification, _ = self(images)
            loss = criterion(classification, labels)
            _, predicted = torch.max(classification, 1)
            correct = (predicted == labels).sum().item()
            accuracy = correct / self.batch_size
            self.logger.info(f'Validation Loss: {loss.item()} Accuracy: {accuracy}')
        self.train()

    def train_model(self):
        dataset_train = self.load_dataset_hf('train')
        dataset_test = self.load_dataset_hf('test')
        self.train()
        self.logger.info('Training the model')
        optimizer = optim.Adam(self.parameters(), betas=(0.9, 0.999), lr=self.config.base_lr, weight_decay=self.config.weight_decay)
        scheduler = optim.lr_scheduler.LinearLR(optimizer)
        criterion = nn.CrossEntropyLoss()
        running_loss = 0.0
        accumulation_steps = 5

        for epoch in range(self.config.epochs):
            self.train()
            images = []
            labels = []
            for _ in range(self.batch_size):
                sample = next(dataset_train)
                images.append(self.training_transform_pipeline(sample['img']))
                labels.append(torch.tensor(sample['label']))

            images = torch.stack(images).to(self.device)
            labels = torch.stack(labels).to(self.device)

            classification, _ = self(images)
            loss = criterion(classification, labels)
            loss.backward()  # Accumulate gradients
            running_loss += loss.item()

            if epoch % accumulation_steps == 0:
                optimizer.step()  # Perform a single update
                scheduler.step()  # Scheduler update
                optimizer.zero_grad()  # Zero the gradients after updating
                self.logger.info(f'Epoch: {epoch} Loss: {loss.item()}')

            if epoch % self.eval_interval == 0 and epoch > 0:
                self.test_model(dataset_test, criterion)
                # print runnign loss
                self.logger.info(f'Epoch: {epoch} Running Loss: {running_loss/self.eval_interval}')
                running_loss = 0.0

        self.logger.info('Saving the model')
        torch.save(self.state_dict(), f'models/vit.pth')
        self.test_model(dataset_test, criterion)

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
        # Random initialization of of [class] token that is prepended to the linear projection vector.
        self.class_token = nn.Parameter(torch.randn(self.batch_size, 1, self.n_embd)).to(self.device)

        # Positional embedding
        self.pos_embedding = nn.Parameter(torch.randn(self.batch_size, 1, self.n_embd)).to(self.device)

    def forward(self, input_data):

        input_data = input_data.to(self.device)

        # Re-arrange image into patches.
        patches = einops.rearrange(
            input_data, 'b c (h h1) (w w1) -> b (h w) (h1 w1 c)', h1=self.patch_size, w1=self.patch_size)

        linear_projection = self.linearProjection(patches).to(self.device)
        b, n, _ = linear_projection.shape

        # Prepend the [class] token to the original linear projection
        linear_projection = torch.cat((self.class_token, linear_projection), dim=1)
        pos_embed = einops.repeat(self.pos_embedding, 'b 1 d -> b m d', m=n+1)

        # Add positional embedding to linear projection
        linear_projection += pos_embed

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
        self.eval_iters = config.eval_iters
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

    def forward(self, test_input):

        # Apply input embedding (patchify + linear projection + position embeding)
        # to the input image passed to the model
        enc_output = self.embedding(test_input)

        # Loop through all the encoder layers
        for enc_layer in self.enc_stack:
            enc_output = enc_layer.forward(enc_output) + enc_output

        # Extract the output embedding information of the [class] token
        cls_token_embedding = enc_output[:, 0]

        # Finally, return the classification vector for all image in the batch
        return self.MLP_head(cls_token_embedding), cls_token_embedding


    def load_dataset_hf(self, split):
        assert split in ['train', 'test']
        self.logger.info(f'Loading the {split} image dataset')
        dataset = load_dataset("cifar10", # other options: "imagenet-1k"
                            use_auth_token=True,
                            streaming=True)
        dataset = iter(dataset[split])
        return dataset

    @torch.no_grad()
    def test_model(self, dataset_test, criterion):
        logger.info('Testing the model')
        self.eval()
        for _ in range(self.eval_iters):
            images = []
            labels = []
            for _ in range(self.batch_size):
                sample = next(dataset_test)
                images.append(self.testing_transform_pipeline(sample['img']))
                labels.append(sample['label'])
            images = torch.stack(images).to(self.device)
            labels = torch.tensor(labels).to(self.device)

            classification, _ = self(images)
            loss = criterion(classification, labels)
            _, predicted = torch.max(classification, 1)
            correct = (predicted == labels).sum().item()
            accuracy = correct / self.batch_size
            self.logger.info(f'Validation Loss: {loss.item()} Accuracy: {accuracy}')
        self.train()

    def train_model(self):
        dataset_train = self.load_dataset_hf('train')
        dataset_test = self.load_dataset_hf('test')
        self.train()
        self.logger.info('Training the model')
        optimizer = optim.Adam(self.parameters(), betas=(0.9, 0.999), lr=config.base_lr, weight_decay=config.weight_decay)
        scheduler = optim.lr_scheduler.LinearLR(optimizer)
        criterion = nn.CrossEntropyLoss()
        running_loss = 0.0
        accumulation_steps = 50

        for epoch in range(config.epochs):
            self.train()
            images = []
            labels = []
            for _ in range(self.batch_size):
                sample = next(dataset_train)
                images.append(self.training_transform_pipeline(sample['img']))
                labels.append(torch.tensor(sample['label']))

            images = torch.stack(images).to(self.device)
            labels = torch.stack(labels).to(self.device)

            classification, _ = self(images)
            loss = criterion(classification, labels)
            loss.backward()  # Accumulate gradients
            running_loss += loss.item()

            if epoch % accumulation_steps == 0:
                optimizer.step()  # Perform a single update
                scheduler.step()  # Scheduler update
                optimizer.zero_grad()  # Zero the gradients after updating
                self.logger.info(f'Epoch: {epoch} Loss: {loss.item()}')

            if epoch % self.eval_interval == 0 and epoch > 0:
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


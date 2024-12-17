
import json
import matplotlib.pyplot as plt
from src.simplellm_config import SimpleLMConfig

def load_data(filepath):
    with open(filepath, 'r') as file:
        data = json.load(file)
    return data

def plot_loss(data):
    config = SimpleLMConfig()
    eval_interval = config.eval_interval
    train_loss = data['train']
    val_loss = data['val']
    epochs = range(1, len(train_loss) * eval_interval, eval_interval)

    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_loss, 'b-', label='Training Loss')
    plt.plot(epochs, val_loss, 'r-', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('logs/loss_plot.png')
    plt.show()

if __name__ == '__main__':
    filepath = 'logs/loss_log.json'  # Path to the JSON file
    data = load_data(filepath)
    plot_loss(data)

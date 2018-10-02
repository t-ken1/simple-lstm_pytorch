import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.optim import Adam

from util import generate_dataset
from model import MyRNN
from trainer import Trainer


# Configuration for network.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
hidden_dim = 5
num_layers = 2

lr = 0.001
max_epoch = 300
batch_size = 100

data_size = 500
sequence_len = 100


# Prepare dataset.
xs, ys = generate_dataset(N=data_size, T=sequence_len)


# Create network.
input_dim = xs.shape[1]
output_dim = ys.shape[1]
model = MyRNN(input_dim, hidden_dim, output_dim, num_layers).to(device)
print(model)


# Train network.
criterion = nn.MSELoss()
optimizer = Adam(model.parameters(), lr=lr)

trainer = Trainer(xs, ys, model, criterion, optimizer, device)
trainer.train(max_epoch, batch_size)


# Predict sin(x)
input, _ = generate_dataset(N=1, T=sequence_len)
input_tensor = torch.tensor(input, dtype=torch.float)
input_tensor = input_tensor.transpose(1, 2)

output_tensor = model(input_tensor)
output = output_tensor.detach().numpy()

input = input.reshape(sequence_len)
output = output.reshape(sequence_len)

plt.plot(input, output)
plt.savefig('output.png')

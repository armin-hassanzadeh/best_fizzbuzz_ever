import math
from typing import NamedTuple
import torch
from torch.utils.data import DataLoader


class Instance(NamedTuple):
    n: int
    inputs: torch.Tensor
    label: int


def binary_digits(n: int) -> torch.Tensor:
    digits = [float((n >> i) & 1) for i in range(10)]
    return torch.tensor(digits)


def label(n: int) -> int:
    return [1, 3, 5, 15].index(math.gcd(n, 15))


def make_instance(n: int) -> Instance:
    inputs = binary_digits(n)
    label_for_n = label(n)
    return Instance(n, inputs, label_for_n)


training_data = [make_instance(n) for n in range(101, 1024)]
test_data = [make_instance(n) for n in range(1, 101)]

torch.manual_seed(1)

hidden_dim = 100

model = torch.nn.Sequential(
    torch.nn.Linear(in_features=10, out_features=hidden_dim),
    torch.nn.ReLU(),
    torch.nn.Linear(in_features=hidden_dim, out_features=4)
)

loss = torch.nn.CrossEntropyLoss()

optimizer = torch.optim.AdamW(model.parameters())

num_epochs = 500
batch_size = 10

for epoch in range(num_epochs):
    epoch_loss = 0.0
    batches = DataLoader(training_data, batch_size=batch_size)
    for batch in batches:
        optimizer.zero_grad()
        predicted = model(batch.inputs)
        error = loss(predicted, batch.label)
        epoch_loss += error.item()
        error.backward()
        optimizer.step()
    print(epoch, epoch_loss)


with torch.no_grad():
    num_correct = 0
    for instance in test_data:
        predicted = model(instance.inputs).argmax().item()
        actual = instance.label
        labels = [str(instance.n), 'fizz', 'buzz', 'fizzbuzz']
        num_correct += predicted == actual
        print(
            'y' if predicted == actual else 'n',
            instance.n,
            labels[predicted],
            labels[actual]
        )

print(num_correct)
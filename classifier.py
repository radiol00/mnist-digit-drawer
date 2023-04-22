import torch
import torch.nn as nn
from torch import relu
from torch.nn.functional import softmax, one_hot
from keras.datasets import mnist
import matplotlib.pyplot as plt

class MnistDigitClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.W1 = nn.Linear(784, 256)
        self.W2 = nn.Linear(256, 256)
        self.W3 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.W1(x)
        x = relu(x)

        x = self.W2(x)
        x = relu(x)

        x = self.W3(x)
        x = softmax(x, dim=1)
        return x
        
    def evaluate(self, data, labels):
        correct = 0
        for i in range(data.shape[0]):
            x = data[i].view((1, -1))
            y = labels[i]

            result = self(x).argmax()
            if result.item() == y:
                correct += 1
        return correct/data.shape[0] * 100

if __name__ == "__main__":
    # torch.manual_seed(1997)
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = torch.tensor(x_train, dtype=torch.float).view((x_train.shape[0], -1)) / 255
    y_train = torch.tensor(y_train, dtype=torch.int64)
    x_test = torch.tensor(x_test, dtype=torch.float).view((x_test.shape[0], -1)) / 255
    y_test = torch.tensor(y_test, dtype=torch.int64)

    EPOCHS = 100_000
    BATCH_SIZE = 32
    LEARNING_RATE = 0.1

    classifier = MnistDigitClassifier()
    if torch.cuda.is_available():
        classifier.cuda()
   
    losses = []
    avg_losses = []

    for i in range(EPOCHS):

        if (i + 1) % (EPOCHS/100) == 0 or i == 0:
            print(f"{((i + 1)/EPOCHS) * 100:.0f}%")

        batch_idx = torch.randint(0, x_train.shape[0], (BATCH_SIZE,))
        batch_x = x_train[batch_idx]
        batch_y = y_train[batch_idx]
        if torch.cuda.is_available():
            batch_x = batch_x.cuda()
            batch_y = batch_y.cuda()

        result = classifier(batch_x)

        expected = one_hot(batch_y, num_classes=10).float()
        loss = ((result - expected)**2).mean()

        losses.append(loss.detach().item())
        avg_losses.append(sum(losses)/len(losses))

        for p in classifier.parameters():
            p.grad = None

        loss.backward()

        for p in classifier.parameters():
            p.data += LEARNING_RATE * (-p.grad)

    if torch.cuda.is_available():
        classifier.cpu()

    print(f"Train Accuracy: {classifier.evaluate(x_train, y_train):.2f}")
    print(f"Test Accuracy: {classifier.evaluate(x_test, y_test):.2f}")

    torch.save(classifier.state_dict(), "Classifier.pt")
    plt.plot(avg_losses)
    plt.show()





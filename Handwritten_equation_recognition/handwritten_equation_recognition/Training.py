import datetime
import numpy as np


def training(train_loader, val_loader, model, optimizer, criterion, epochs, device):
    train_losses = np.zeros(epochs)
    test_losses = np.zeros(epochs)
    for i in range(epochs):
        d0 = datetime.datetime.now()
        train_loss = []
        z = 1
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            train_loss.append(loss.item())
            loss.backward()
            optimizer.step()
            print(z)
            z+=1

        train_losses[i] = np.mean(train_loss)
        test_loss = []
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            test_loss.append(loss.item())

        test_losses[i] = np.mean(test_loss)
        dt = datetime.datetime.now() - d0
        print(f"epoc {i + 1}/{epochs}, train loss={train_losses[i]}, test loss={test_losses[i]}, duration={dt}")

    return train_losses, test_losses


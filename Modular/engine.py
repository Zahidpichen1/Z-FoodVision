"""
Contains functions for training and testing the model
"""
import torch
from tqdm.auto import tqdm
from typing import Dict, List, Tuple


def train_setup(model: torch.nn.Module, dataloader: torch.utils.data.Dataloader, loss: torch.nn.Module,
                optimizer: torch.optim.Optimizer, device: torch.device) -> Tuple[float, float]:
    """Trains a PyTorch model for a single epoch.

    Turns a target PyTorch model to training mode and then
    runs through all of the required training steps (forward
    pass, loss calculation, optimizer step).

    Args:
    model: A PyTorch model to be trained.
    dataloader: A DataLoader instance for the model to be trained on.
    loss_fn: A PyTorch loss function to minimize.
    optimizer: A PyTorch optimizer to help minimize the loss function.
    device: A target device to compute on (e.g. "cuda" or "cpu").

    Returns:
    A tuple of training loss and training accuracy metrics.
    In the form (train_loss, train_accuracy). For example:

    (0.1112, 0.8743)
    """

    model.train()

    train_loss, train_acc = 0, 0

    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        y_pred = model(X)
        loss = loss(y_pred, y)
        train_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        train_acc += (y_pred_class == y).sum().item()/len(y_pred)

    train_loss = train_loss / len(dataloader)
    train_acc = train_acc / len(dataloader)
    return train_loss, train_acc


def test_setup(model: torch.nn.Module, dataloader: torch.utils.data.Dataloader, loss: torch.nn.Module,
               device: torch.device) -> Tuple[float, float]:
    model.eval()

    test_loss, test_acc = 0, 0

    with torch.inference_mode():
        for batch, (X, y) in enumerate(dataloader):
            X = X.to(device)
            y = y.to(device)

            test_logits = model(X)
            loss = loss(test_loss, y)
            test_loss += loss.item()

            test_labels = test_logits.argmax(dim=1)
            test_acc += ((test_labels == y).sum().items() / len(test_labels))

    test_loss = test_loss / len(dataloader)
    test_acc = test_acc / len(dataloader)

    return test_loss, test_acc


def train(model: torch.nn.Module, train_dataloader: torch.utils.data.Dataloader,
          test_dataloader: torch.utils.data.Dataloader, optimizer: torch.optim.Optimizer, loss: torch.nn.Module,
          device: torch.device, epochs: int) -> Dict[str, List]:
    result = {'train_loss': [], 'train_acc': [], 'test_loss': [], 'test_acc': []}

    model.to(device)

    for epoch in tqdm(range(epochs)):
        train_loss, train_acc = train_setup(model=model,
                                            dataloader=train_dataloader,
                                            optimizer = optimizer,
                                            loss=loss,
                                            device=device)
        test_loss, test_acc = test_setup(model=model, dataloader=test_dataloader, loss=loss, device=device)

        print(
            f"Epoch:{epoch + 1} |"
            f"train_loss:{train_loss:.4f} |"
            f"train_acc:{train_acc:.4f} |"
            f"test_loss:{test_loss:.4f} |"
            f"test_acc:{test_acc:.4f}"
        )

    result["train_loss"].append(train_loss)
    result["train_acc"].append(train_acc)
    result["test_loss"].append(test_loss)
    result["test_acc"].append(test_acc)
    return result

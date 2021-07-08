import torch
from pytorch_lightning import Callback
from torch.utils.data import DataLoader
import torchvision.transforms as T
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split

from torchvision import datasets
from model import AttentionFreeTransformer
from model import VisionTransformer
from torchvision.datasets import FashionMNIST
import os

epochs = 50

model_config = {
    "img_size": 384,
    "patch_size": 384,
    "in_chans": 3,
    "n_classes": 1,
    "embed_dim": 3,
    "depth": 1,
    "n_heads": 3,
    "qkv_bias": True,
    "mlp_ratio": 4
}


class LightingVisionTransformer(pl.LightningModule):
    def __init__(self, lr=1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        # self.weights = torch.tensor([20.694]).to('cuda')
        self.model = VisionTransformer(**model_config)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y = y.float()
        output = F.sigmoid(self.forward(x)).squeeze(dim=1)
        print()
        print("Output:", output)
        print("Y shape:", y)
        loss = F.binary_cross_entropy(output, y)
        pred = (output > 0.5).float()
        print("Pred:", pred)
        accuracy = torch.sum(y == pred).item() / (len(y) * 1.0)
        self.log('train_loss', loss, on_step=True, on_epoch=True)
        self.log('train_accuracy', accuracy, on_step=True, on_epoch=True)
        return {
            'loss': loss,
            'accuracy': accuracy
        }

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        y = y.float()
        output = F.sigmoid(self.forward(x)).squeeze(dim=1)
        print()
        print("Output:", output)
        print("Y shape:", y)
        loss = F.binary_cross_entropy(output, y)
        pred = (output > 0.5).float()
        print("Pred:", pred)
        accuracy = torch.sum(y == pred).item() / (len(y) * 1.0)
        self.log('val_loss', loss, on_step=True, on_epoch=True)
        self.log('val_accuracy', accuracy, on_step=True, on_epoch=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)


def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    transforms.append(T.Resize(model_config['img_size']))
    transforms.append(T.Normalize([6.4036902, 27.1573784, 31.35695405], [23.49524942, 64.09265835, 72.92890321]))
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)


def train(model, loader, device, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(loader):
        data, target = data.to(device), target.to(device).float()
        optimizer.zero_grad()
        output = model(data)
        loss = F.binary_cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            epoch, batch_idx * len(data), len(loader.dataset),
                   100. * batch_idx / len(loader), loss.item()))


def test(model, device, loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device).float()
            output = model(data)
            test_loss += F.binary_cross_entropy(output, target)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(loader.dataset),
        100. * correct / len(loader.dataset)))


def main():
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    train_dataset = dataset.LemonDataset("data/train", transforms=get_transform(True))
    val_dataset = dataset.LemonDataset("data/val", transforms=get_transform(False))

    train_loader = DataLoader(train_dataset, batch_size=8)
    val_loader = DataLoader(val_dataset, batch_size=8)

    model = VisionTransformer(**model_config).to(device)
    optimizer = torch.optim.Adam(model.parameters())

    for epoch in range(epochs):
        train(model, train_loader, device, optimizer, epoch)
        test(model, device, val_loader)


class MNISTDataModule(pl.LightningDataModule):

    def setup(self, stage):
        # transforms for images
        transform = T.Compose([T.ToTensor(), T.Normalize((0.1307,), (0.3081,))])

        # prepare transforms standard to MNIST
        mnist_train = FashionMNIST(os.getcwd(), train=True, download=True, transform=transform)
        mnist_test = FashionMNIST(os.getcwd(), train=False, download=True, transform=transform)

        self.mnist_train, self.mnist_val = random_split(mnist_train, [55000, 5000])

    def train_dataloader(self):
        return DataLoader(self.mnist_train, batch_size=256)

    def val_dataloader(self):
        return DataLoader(self.mnist_val, batch_size=256)


class ImagePredictionLogger(Callback):
    def __init__(self, val_samples, num_samples=32):
        super().__init__()
        self.num_samples = num_samples
        self.val_images, self.val_labels = val_samples

    def on_validation_epoch_end(self, trainer: 'pl.Trainer', pl_module: 'pl.LightningModule') -> None:
        val_imgs = self.val_images.to(device=pl_module.device)
        val_labels = self.val_labels.to(device=pl_module.device)

        logits = pl_module(val_imgs)
        preds = (logits > 0.5).float()
        [
            trainer.logger.experiment.add_image(f"Pred:{pred}, Label:{y}", x, pl_module.current_epoch)
            for x, pred, y in zip(val_imgs[:self.num_samples], preds[:self.num_samples], val_labels[:self.num_samples])
        ]

def main_lighting():
    train_dataset = datasets.ImageFolder('data/lemon-tiny-dataset/train', transform=get_transform(True))
    val_dataset = datasets.ImageFolder('data/lemon-tiny-dataset/val', transform=get_transform(True))

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=True)

    val_samples = next(iter(val_loader))
    image_callback = ImagePredictionLogger(val_samples)

    model = LightingVisionTransformer(lr=1e-2)
    trainer = pl.Trainer(gpus=1, max_epochs=epochs, callbacks=[image_callback])
    trainer.fit(model, train_loader, val_loader)


def main_mnist_lighting():
    model = LightingVisionTransformer()
    data = MNISTDataModule()

    trainer = pl.Trainer(max_epochs=15)

    trainer.fit(model, data)


if __name__ == '__main__':
    main_lighting()
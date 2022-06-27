import torch
from torch import nn
from torch.nn import functional as fnl
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.backends.cudnn.benchmark = True

def make_image_grid(images: torch.Tensor, nrow: int):
    image_grid = make_grid(images, nrow)
    image_grid = image_grid.permute(1, 2, 0)
    return image_grid.cpu().numpy()


def show_image_grid(images: torch.Tensor, nrow: int):
    image_grid = make_image_grid(images, nrow)
    plt.imshow(image_grid)
    plt.xticks([])
    plt.yticks([])
    plt.show()


def save_image_grid(epoch:int, images: torch.Tensor, nrow: int):
    image_grid = make_image_grid(images, nrow)
    plt.imshow(image_grid)
    plt.xticks([])
    plt.yticks([])
    plt.savefig(f"generated/generated_{epoch:03d}.jpg", bbox_inches="tight", pad_inches=0.0)
    plt.close()


generator = nn.Sequential(nn.Linear(100, 128),
                          nn.LeakyReLU(0.01),
                          nn.Linear(128, 784),
                          nn.Sigmoid()).to(device)

discriminator = nn.Sequential(nn.Linear(784, 128),
                              nn.LeakyReLU(0.01),
                              nn.Linear(128, 1)).to(device)


def calc_loss(images: torch.Tensor, targets: torch.Tensor):
    pred = discriminator(images.reshape(-1, 784))
    loss = fnl.binary_cross_entropy_with_logits(pred, targets)
    return loss


def generate_images(batch_size: int):
    z = torch.randn(batch_size, 100).to(device)
    output = generator(z)
    return output.reshape(batch_size, 1, 28, 28)


S = 64
override_epoch = 0


if __name__ == "__main__":
    if False:
        load_epoch = 105
        status = torch.load(f"./snapshots/epoch_{load_epoch:03d}.pth")
        generator.load_state_dict(status[0])
        discriminator.load_state_dict(status[1])
        override_epoch = load_epoch + 1

    transform = transforms.ToTensor()
    dataset = datasets.MNIST(root="./mnist", train=True, download=True, transform=transform)

    dataloader = DataLoader(dataset, batch_size=S, drop_last=True, num_workers=2, pin_memory=True)

    # print(images.shape)
    # print(labels)
    # show_image_grid(images, nrow=8)

    d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=1.0e-4)
    g_optimizer = torch.optim.Adam(generator.parameters(), lr=1.0e-4)

    discriminator.train(True)

    true_target = torch.ones(S, 1).to(device)
    false_target = torch.zeros(S, 1).to(device)

    for epoch in range(override_epoch, 200):
        d_losses = []
        g_losses = []

        for mnist_images_cpu, mnist_labels in tqdm(dataloader):
            mnist_images = mnist_images_cpu.to(device)
            d_loss = calc_loss(mnist_images, true_target)

            generated_images = generate_images(S)
            d_loss += calc_loss(generated_images, false_target)

            d_optimizer.zero_grad()
            d_loss.backward()
            d_optimizer.step()

            generated_images = generate_images(S)
            g_loss = calc_loss(generated_images, true_target)

            g_optimizer.zero_grad()
            g_loss.backward()
            g_optimizer.step()

            d_losses.append(d_loss.item())
            g_losses.append(g_loss.item())

        print(epoch, np.mean(d_losses), np.mean(g_losses))
        torch.save([generator.state_dict(), discriminator.state_dict()], f"./snapshots/epoch_{epoch:03d}.pth")
        save_image_grid(epoch, generate_images(S), nrow=8)

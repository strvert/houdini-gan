import torch
from torch import nn, optim, Tensor
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp
import math

from typing import Dict, List

from models import *

torch.backends.cudnn.benchmark = True
AVAILABLE_GPU = torch.cuda.is_available()
DEVICE = torch.device('cuda' if AVAILABLE_GPU else 'cpu')
DEVICE_LOAD_PARAM = {'num_workers': 3, 'pin_memory': True} if AVAILABLE_GPU else {}

BATCH_SIZE = 256
ZERO_PADDING = 2
LR = 1.0e-4

EPOCHS = 100
TEST_UNIT_SIZE = 512
TEST_SAMPLE_SIZE = 20


def load_mapping(file_path: str) -> Dict[int, str]:
    with open(file_path, "r") as f:
        lines = f.readlines()
        return {int(pair[0]): chr(int(pair[1])) for pair in [line.strip().split(' ') for line in lines]}


def image_transformer(image):
    tensor = transforms.ToTensor()(image)
    tensor = tensor.reshape([28, 28]).T
    return tensor.reshape([1, 28, 28])


def train_worker(use_model: nn.Module, train_dataset, test_dataset, status_queue: mp.Queue, snapshot: str = None):
    dataloader: DataLoader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True,
                                        **DEVICE_LOAD_PARAM)
    test_dataloader: DataLoader = DataLoader(test_dataset, batch_size=TEST_UNIT_SIZE, shuffle=True, drop_last=True,
                                             **DEVICE_LOAD_PARAM)

    cnn_net: nn.Module = use_model().to(DEVICE)
    criterion = nn.CrossEntropyLoss().to(DEVICE)
    optimizer = optim.Adam(cnn_net.parameters(), LR)

    if snapshot is not None:
        cnn_net.load_state_dict(torch.load(snapshot))

    max_epoch = EPOCHS
    for epoch in range(max_epoch):
        total_loss = 0.0

        cnn_net.train(True)
        train_images_cpu: Tensor
        train_labels: Tensor
        for index, data in enumerate(tqdm(dataloader)):
            train_images_cpu, teacher_labels_cpu = data
            train_images: Tensor = train_images_cpu.to(DEVICE)
            teacher_labels: Tensor = teacher_labels_cpu.to(DEVICE)

            optimizer.zero_grad()
            outputs: Tensor = cnn_net(train_images)
            loss: Tensor = criterion(outputs, teacher_labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        with torch.no_grad():
            cnn_net.eval()
            test_total_loss = 0

            test_images_cpu: Tensor = Tensor(0)
            ans_labels_cpu: Tensor = Tensor(0)
            outputs: Tensor = Tensor(0)
            for test_images_cpu, ans_labels_cpu in tqdm(test_dataloader):
                test_images: Tensor = test_images_cpu.to(DEVICE)
                ans_labels = ans_labels_cpu.to(DEVICE)
                outputs: Tensor = cnn_net(test_images)
                test_loss: Tensor = criterion(outputs, ans_labels)
                test_total_loss += test_loss.item()

            samples = TEST_SAMPLE_SIZE
            status_queue.put({
                'epoch': epoch,
                'max_epoch': max_epoch,
                'net_name': use_model.network_name(),
                'train_total_loss': total_loss,
                'test_total_loss': test_total_loss,
                'train_count': len(dataloader) * BATCH_SIZE,
                'test_count': len(test_dataloader) * TEST_UNIT_SIZE,
                'images': test_images_cpu[:samples].detach(),
                'results': outputs[:samples].to('cpu').detach(),
                'answers': ans_labels_cpu[:samples].detach()
            })

            torch.save(cnn_net.state_dict(), f"./snapshots/{use_model.network_name()}/epoch_{epoch:03d}.pth")


def start_monitor(class_mapping: Dict[int, str], queue: mp.Queue):
    train_loss_history = []
    test_loss_history = []
    plt.ion()

    fig, ax = plt.subplots()
    fig.tight_layout()
    ax.axis('off')

    max_horizontal_images = 10
    image_count = TEST_SAMPLE_SIZE
    image_rows = math.ceil(image_count / max_horizontal_images)
    print(image_rows)
    image_row_ratios = [1.5] * image_rows
    gridspec = plt.GridSpec(2 + image_rows, min(image_count, max_horizontal_images),
                            height_ratios=[4, *image_row_ratios, 1], figure=fig)
    loss_ax: plt.Axes = fig.add_subplot(gridspec[0, :])
    img_axes: List[plt.Axes] = []
    for index in range(image_count):
        img_ax: plt.Axes = fig.add_subplot(gridspec[1 + index // max_horizontal_images, index % max_horizontal_images])
        img_ax.axis("off")
        img_axes.append(img_ax)

    details_ax: plt.Axes = fig.add_subplot(gridspec[1 + image_rows, :])
    details_ax.axis("off")

    while True:
        if not queue.empty():
            status = queue.get()

            epoch = status['epoch']
            max_epoch = status['max_epoch']
            net_name = status['net_name']
            images = status['images']
            train_count = status['train_count']
            test_count = status['test_count']
            train_loss_history.append(status['train_total_loss'])
            test_loss_history.append(status['test_total_loss'])
            result_labels = status['results']
            ans_labels = status['answers']

            images_cpu = images.to('cpu').detach().numpy()

            loss_ax.clear()
            loss_ax.grid(linestyle="dashed")
            epochs = np.arange(len(train_loss_history))
            loss_ax.plot(epochs, [his / train_count for his in train_loss_history], label="train loss")
            loss_ax.plot(epochs, [his / test_count for his in test_loss_history], label="test loss")
            loss_ax.set_xticks(epochs, epochs)
            loss_ax.set_title("Loss history")
            loss_ax.set_xlabel("epoch")
            loss_ax.set_ylabel("loss")
            loss_ax.legend()

            details_ax.clear()
            details_ax.axis("off")
            details_ax.text(0, 0.0, f"Epoch: {epoch} / {max_epoch}", fontsize=11, horizontalalignment="left",
                            verticalalignment="top")
            details_ax.text(0, 0.2, f"Current test loss: {test_loss_history[-1]}", fontsize=11,
                            horizontalalignment="left",
                            verticalalignment="top")
            details_ax.text(0, 0.4, f"Network Name: {net_name}", fontsize=11, horizontalalignment="left",
                            verticalalignment="top")

            for index in range(image_count):
                img_ax = img_axes[index]
                img_ax.clear()
                img_ax.axis("off")
                img_ax.set_title(
                    f"""res: {class_mapping[torch.argmax(result_labels[index]).item()]}
ans: {class_mapping[ans_labels[index].item()]}""",
                    y=-1)
                img_ax.imshow(images_cpu[index][0])

            if epoch % 10:
                plt.savefig(f"./generated/{net_name}/epoch_{epoch:03d}.png")
                pass

        fig.canvas.draw()
        fig.canvas.flush_events()
        plt.pause(0.05)


def main():
    # torch.onnx.export(cnn_net, torch.randn(1, 1, 28, 28).to(DEVICE), "./cnn_net.onnx", verbose=True)
    train_dataset = datasets.EMNIST(root="./dataset/emnist", split="balanced", train=True, download=True,
                                    transform=transforms.Lambda(image_transformer))
    test_dataset = datasets.EMNIST(root="./dataset/emnist", split="balanced", train=False, download=True,
                                   transform=transforms.Lambda(image_transformer))

    worker_status_queue = mp.Queue()
    proc = mp.Process(target=train_worker, args=(
        CNNNetWithBN, train_dataset, test_dataset, worker_status_queue))
    proc.start()

    class_mapping = load_mapping("./emnist-balanced-mapping.txt")
    start_monitor(class_mapping, worker_status_queue)


if __name__ == "__main__":
    main()

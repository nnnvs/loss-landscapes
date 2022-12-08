import argparse
import torch
from models.resnet import ResNet
import torch.optim as optim
from torchvision import datasets, transforms
from utils.training import train, test
import logging 
import time 
import os 
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--opt', type=str, default='SGD')
    parser.add_argument('--depth', type=int, default=22)
    parser.add_argument('--width-multiplier', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',help='how many batches to wait before logging training status')
    args = parser.parse_args()
    out_dir = f"./trained_models/resnet_{str(args.depth)}_{str(args.seed)}/cifar10/"
    log_file_path = out_dir + f"resnet_cifar10_{str(args.depth)}_{str(args.seed)}_{str(args.width_multiplier)}.log"
    model_checkpoint_path = out_dir + f"resnet_cifar10_{str(args.depth)}_{str(args.seed)}_{str(args.width_multiplier)}_final.pt"
    logging.basicConfig(filename= log_file_path, filemode= 'w')
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    # Get data
    args = parser.parse_args()
    use_cuda = torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                        'pin_memory': True,
                        'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)


    test_transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    
    train_transforms = transforms.Compose(
    [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]
)

    trainset = datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=train_transforms)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
                                            shuffle=True, num_workers=2)

    testset = datasets.CIFAR10(root='./data', train=False,
                                        download=True, transform=test_transform)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size,
                                            shuffle=False, num_workers=2)

    model = ResNet(args.depth, args.width_multiplier, 0, num_classes=10).to(device)
    if args.opt == "adam":
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
    else:
        optimizer = optim.SGD(model.parameters(), momentum=0.9, lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    start = time.time()
    end = time.time()
    total_time = end - start
    for epoch in range(1, args.epochs + 1):
        start = time.time()
        train(args, model, device, train_loader, optimizer, epoch, logger, True)
        end = time.time()
        total_time += end - start
        test(model, device, test_loader, logger, True)
        scheduler.step()
        if epoch%10 == 0:
            torch.save(model.state_dict(),out_dir + f"resnet_cifar10_{str(args.depth)}_{str(args.seed)}_{str(args.width_multiplier)}_{str(epoch)}.pt")
    logger.info(f"total_trainnig_time: {str(total_time)}")
    torch.save(model.state_dict(), model_checkpoint_path)

if __name__ == "__main__":
  main()
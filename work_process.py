import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import csv

import torch
import math
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from torch import optim
from torchvision import transforms

from models.ResNet import TSEncoder
from models.Encoder import Encoder
from models.mmm4tsc import TSMultiModalMamba
from models.img_encoder import IMG_Encoder

from timm.loss import LabelSmoothingCrossEntropy
from data_provider import UCR_data_provider, M4TSC_UCR_data_provider, IMG_UCR_data_provider
import thop


def work_process(args):
    # 数据处理
    UCR_dataset_train = UCR_data_provider(args=args, dataset_type='train')
    UCR_dataset_test = UCR_data_provider(args=args, dataset_type='test')

    # length使用dataset，dataloader的length被batch_size除过
    # 注意drop_last, false则下取整, true上取整
    train_len = (len(UCR_dataset_train) // args.train_batch_size) * args.train_batch_size
    test_len = math.ceil(len(UCR_dataset_test) / args.test_batch_size) * args.test_batch_size

    # 返回一个三维tensor, (batch_size, seq_len, S=1/MS=7)
    train_loader = DataLoader(UCR_dataset_train, batch_size=args.train_batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(UCR_dataset_test, batch_size=args.test_batch_size, shuffle=False, drop_last=False)

    if args.model == 'CNN':  # 未实现
        model = TSEncoder(dim=UCR_dataset_train.input_size, n_classes=UCR_dataset_train.output_size)
    elif args.model == 'Encoder':
        model = Encoder(input_size=UCR_dataset_train.input_size, output_size=UCR_dataset_train.output_size)
    model.to(args.device)

    model_parameters = []
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            model_parameters.append(parameter.numel())
            print(f"{name} has {parameter.numel()} parameters")

    total_params = sum(model_parameters)
    print(f"Total number of trainable parameters: {total_params}")

    loss_fn = LabelSmoothingCrossEntropy()  # 多维float32, 一维int64
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)

    eva_file = args.eva_store_path + '/' + args.model + '/'
    train_epoch_acc = []
    train_epoch_loss = []
    test_epoch_acc = []
    test_epoch_loss = []

    for e in range(args.epochs):
        # print("epoch number: {}".format(e + 1))
        train_acc = 0
        train_loss = 0
        test_acc = 0
        test_loss = 0

        if args.learning_decay and (e + 1) % 50 == 0:
            optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate / 10)

        # train
        for i, (x_train, y_train) in enumerate(train_loader):
            x_train = x_train.float().to(args.device)
            y_train = y_train.to(torch.int64).to(args.device)
            y_pred = model(x_train)
            predicted = torch.max(y_pred.data, 1)[1]
            loss = loss_fn(y_pred, y_train)

            train_loss += loss.item()
            train_acc += (predicted == y_train).sum()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print('Epoch: {}, Cur_lr: {:.5f}, Train Loss: {:.5f}, Train Acc: {:.3f}'.format(e, args.learning_rate,
                                                                                        (train_loss / train_len),
                                                                                        (100 * (
                                                                                                train_acc.item() / train_len))
                                                                                        ))

        train_epoch_acc.append(100 * (train_acc.item() / train_len))
        train_epoch_loss.append(train_loss / train_len)

        # test

        for i, (x_test, y_test) in enumerate(test_loader):
            x_test = x_test.float().to(args.device)
            y_test = y_test.to(torch.int64).to(args.device)
            y_val = model(x_test)
            predicted = torch.max(y_val.data, 1)[1]
            loss = loss_fn(y_val, y_test)

            test_loss += loss.item()
            test_acc += (predicted == y_test).sum()

        print('Epoch: {}, Cur_lr: {:.5f}, Test Loss: {:.5f}, Test Acc: {:.3f}'.format(e, args.learning_rate,
                                                                                      (test_loss / test_len),
                                                                                      (100 * (
                                                                                              test_acc.item() / test_len))
                                                                                      ))

        test_epoch_acc.append(100 * (test_acc.item() / test_len))
        test_epoch_loss.append(test_loss / test_len)

    with open('result/' + args.model + '/' + f'{args.sub_data}.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['train_loss', 'train_accuracy', 'test_loss', 'test_accuracy'])
        for train_loss, train_accuracy, test_loss, test_accuracy in zip(train_epoch_loss, train_epoch_acc,
                                                                        test_epoch_loss, test_epoch_acc):
            writer.writerow([train_loss, train_accuracy, test_loss, test_accuracy])
    path_img = eva_file + f'{args.sub_data}.jpg'
    time = list(range(args.epochs))
    fig, (a1, a2) = plt.subplots(1, 2, sharex=False, sharey=False)
    fig.set_figheight(10)
    fig.set_figwidth(30)
    plt.subplots_adjust(wspace=0.3, hspace=0)

    x_major_locator = plt.MultipleLocator(5)

    a1.plot(time, train_epoch_acc, color='r', marker='o', label='Train_ACC', linewidth=2.5, markersize=10)
    a1.plot(time, test_epoch_acc, color='g', marker='v', label='Test_ACC', linewidth=2.5, markersize=10)
    a1.set_xlabel('Time', fontdict={'size': 18}, labelpad=-1)
    a1.set_ylabel('ACC:%', fontdict={'size': 18})
    a1.tick_params(labelsize=14)
    a1.xaxis.set_major_locator(x_major_locator)
    a1.set_title('Accuarcy', fontsize=20)
    a1.legend(loc=0, prop={'size': 18})

    a2.plot(time, train_epoch_loss, color='r', marker='o', label='Train_Loss', linewidth=2.5, markersize=10)
    a2.plot(time, test_epoch_loss, color='g', marker='v', label='Test_Loss', linewidth=2.5, markersize=10)
    a2.set_xlabel('Time', fontdict={'size': 18}, labelpad=-1)
    a2.set_ylabel('Loss', fontdict={'size': 18})
    a2.tick_params(labelsize=14)
    a2.xaxis.set_major_locator(x_major_locator)
    a2.set_title('Loss', fontsize=20)
    a2.legend(loc=0, prop={'size': 18})

    fig.tight_layout()
    plt.savefig(path_img)


def multimodal_work_process(args):
    img_transform = {
        'train': transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize([0.1307], [0.3081])
        ]),
        'test': transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize([0.1307], [0.3081])
        ])
    }

    # 数据处理
    UCR_dataset_train = M4TSC_UCR_data_provider(args=args, dataset_type='train', transform=img_transform['train'])
    UCR_dataset_test = M4TSC_UCR_data_provider(args=args, dataset_type='test', transform=img_transform['test'])

    # length使用dataset，dataloader的length被batch_size除过
    # 注意drop_last, false则下取整, true上取整
    train_len = (len(UCR_dataset_train) // args.train_batch_size) * args.train_batch_size
    test_len = math.ceil(len(UCR_dataset_test) / args.test_batch_size) * args.test_batch_size

    # 返回一个三维tensor, (batch_size, seq_len, S=1/MS=7)
    train_loader = DataLoader(UCR_dataset_train, batch_size=args.train_batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(UCR_dataset_test, batch_size=args.test_batch_size, shuffle=False, drop_last=False)

    model = TSMultiModalMamba(dim=UCR_dataset_train.input_size,
                              depth=6,
                              dropout=0.2,
                              heads=8,
                              d_state=128,
                              image_size=128,
                              patch_size=32,
                              encoder_dim=UCR_dataset_train.input_size,
                              encoder_depth=6,
                              encoder_heads=8,
                              n_classes=UCR_dataset_train.output_size,
                              fusion_method='concat')

    model_parameters = []
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            model_parameters.append(parameter.numel())
            print(f"{name} has {parameter.numel()} parameters")

    total_params = sum(model_parameters)
    print(f"Total number of trainable parameters: {total_params}")

    # print(model)

    model.to(args.device)

    loss_fn = LabelSmoothingCrossEntropy()  # 多维float32, 一维int64
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)

    eva_file = args.eva_store_path + '/' + args.model + '/'
    train_epoch_acc = []
    train_epoch_loss = []
    test_epoch_acc = []
    test_epoch_loss = []

    for e in range(args.epochs):
        train_acc = 0
        train_loss = 0
        test_acc = 0
        test_loss = 0

        if args.learning_decay and (e + 1) % 50 == 0:
            optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate / 10)

        output = 10

        # train
        for i, (ts, img, label) in enumerate(train_loader):
            model.train()
            ts = ts.float().to(args.device)
            img = img.float().to(args.device)
            label = label.to(torch.int64).to(args.device)
            y_pred = model(ts, img)
            predicted = torch.max(y_pred.data, 1)[1]
            loss = loss_fn(y_pred, label)

            train_loss += loss.item()
            train_acc += (predicted == label).sum()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # if output >= train_loss:
            #     print('Saving model')
            #     save_path = './result/%s.pth' % (args.sub_data)
            #     train_loss = output
            #     torch.save(model.state_dict(), save_path)

        print('Epoch: {}, Cur_lr: {:.5f}, Train Loss: {:.5f}, Train Acc: {:.3f}'.format(e, args.learning_rate,
                                                                                        (train_loss / train_len),
                                                                                        (100 * (
                                                                                                train_acc.item() / train_len))
                                                                                        ))

        ts = ts.unsqueeze(dim=1)
        image_macs, image_params = thop.profile(model.img_encoder, inputs=(img,), verbose=False)
        ts_macs, ts_params = thop.profile(model.ts_encoder, inputs=(ts,), verbose=False)

        # 计算总的FLOPs
        total_macs = image_macs + ts_macs
        total_params = image_params + ts_params

        from thop import clever_format
        total_macs, total_params = clever_format([total_macs, total_params], "%.3f")

        print(f"Total FLOPs: {total_macs}", f"Total params: {total_params}")

        train_epoch_acc.append(100 * (train_acc.item() / train_len))
        train_epoch_loss.append(train_loss / train_len)

        # test
        for i, (ts, img, label) in enumerate(test_loader):
            model.eval()
            with torch.no_grad():
                ts = ts.float().to(args.device)
                img = img.float().to(args.device)
                label = label.to(torch.int64).to(args.device)
                y_val = model(ts, img)
                predicted = torch.max(y_val.data, 1)[1]
                loss = loss_fn(y_val, label)

                test_loss += loss.item()
                test_acc += (predicted == label).sum()

        print('Epoch: {}, Cur_lr: {:.5f}, Test Loss: {:.5f}, Test Acc: {:.3f}'.format(e, args.learning_rate,
                                                                                      (test_loss / test_len),
                                                                                      (100 * (
                                                                                              test_acc.item() / test_len))
                                                                                      ))

        test_epoch_acc.append(100 * (test_acc.item() / test_len))
        test_epoch_loss.append(test_loss / test_len)

    with open('result/' + args.model + '/' + f'{args.sub_data}.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['train_loss', 'train_accuracy', 'test_loss', 'test_accuracy'])
        for train_loss, train_accuracy, test_loss, test_accuracy in zip(train_epoch_loss, train_epoch_acc,
                                                                        test_epoch_loss, test_epoch_acc):
            writer.writerow([train_loss, train_accuracy, test_loss, test_accuracy])

    path_img = eva_file + f'{args.sub_data}.jpg'
    time = list(range(args.epochs))
    fig, (a1, a2) = plt.subplots(1, 2, sharex=False, sharey=False)
    fig.set_figheight(10)
    fig.set_figwidth(30)
    plt.subplots_adjust(wspace=0.3, hspace=0)

    x_major_locator = plt.MultipleLocator(5)

    a1.plot(time, train_epoch_acc, color='r', marker='o', label='Train_ACC', linewidth=2.5, markersize=10)
    a1.plot(time, test_epoch_acc, color='g', marker='v', label='Test_ACC', linewidth=2.5, markersize=10)
    a1.set_xlabel('Time', fontdict={'size': 18}, labelpad=-1)
    a1.set_ylabel('ACC:%', fontdict={'size': 18})
    a1.tick_params(labelsize=14)
    a1.xaxis.set_major_locator(x_major_locator)
    a1.set_title('Accuarcy', fontsize=20)
    a1.legend(loc=0, prop={'size': 18})

    a2.plot(time, train_epoch_loss, color='r', marker='o', label='Train_Loss', linewidth=2.5, markersize=10)
    a2.plot(time, test_epoch_loss, color='g', marker='v', label='Test_Loss', linewidth=2.5, markersize=10)
    a2.set_xlabel('Time', fontdict={'size': 18}, labelpad=-1)
    a2.set_ylabel('Loss', fontdict={'size': 18})
    a2.tick_params(labelsize=14)
    a2.xaxis.set_major_locator(x_major_locator)
    a2.set_title('Loss', fontsize=20)
    a2.legend(loc=0, prop={'size': 18})

    fig.tight_layout()
    plt.savefig(path_img)

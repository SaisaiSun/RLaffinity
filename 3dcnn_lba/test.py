import argparse
import datetime
import json
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import time
import tqdm

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from lba.datasets import LMDBDataset
from scipy.stats import spearmanr

from model import CNN3D_LBA
from data import CNN3D_TransformLBA
import resnet


# Construct model
def conv_model(in_channels, spatial_size, args):
    num_conv = args.num_conv
    conv_filters = [32 * (2**n) for n in range(num_conv)]
    conv_kernel_size = 3
    max_pool_positions = [0, 1]*int((num_conv+1)/2)
    max_pool_sizes = [2]*num_conv
    max_pool_strides = [2]*num_conv
    fc_units = [512]

    model = CNN3D_LBA(
        in_channels, spatial_size,
        args.conv_drop_rate,
        args.fc_drop_rate,
        conv_filters, conv_kernel_size,
        max_pool_positions,
        max_pool_sizes, max_pool_strides,
        fc_units,
        batch_norm=args.batch_norm,
        dropout=not args.no_dropout)
    return model


@torch.no_grad()
def test(pre_model, model, loader, device):
    model.eval()

    losses = []

    ids = []
    y_true = []
    y_pred = []

    for data in loader:
        print(data['id'])
        feature = data['feature'].to(device).to(torch.float32)
        new_feature = pre_model(feature)
        label = data['label'].to(device).to(torch.float32)
        output = model(new_feature)
        batch_losses = F.mse_loss(output, label, reduction='none')
        losses.extend(batch_losses.tolist())
        ids.extend(data['id'])
        y_true.extend(label.tolist())
        y_pred.extend(output.tolist())

    results_df = pd.DataFrame(
        np.array([ids, y_true, y_pred]).T,
        columns=['structure', 'true', 'pred'],
        )
    r_p = np.corrcoef(y_true, y_pred)[0,1]
    r_s = spearmanr(y_true, y_pred)[0]

    return np.sqrt(np.mean(losses)), r_p, r_s, results_df


def run_test(args, device):
    print("Testing model with config:")
    print(str(json.dumps(args.__dict__, indent=4)) + "\n")

    # Save config
    with open(os.path.join(args.output_dir, 'config.json'), 'w') as f:
        json.dump(args.__dict__, f, indent=4)

    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)

    test_dataset = LMDBDataset(os.path.join(args.data_dir, 'test'),
                               transform=CNN3D_TransformLBA(random_seed=args.random_seed))
    test_loader = DataLoader(test_dataset, args.batch_size, shuffle=False)

    # for data in test_loader:
    #     in_channels, spatial_size = data['feature'].size()[1:3]
    #     print('num channels: {:}, spatial size: {:}'.format(in_channels, spatial_size))
    #     #print(data['feature'].size())
    #     break


    pre_model = resnet.generate_model(18).to(device)
    pre_model.load_state_dict(torch.load('pth/model_stage1_epoch20.pth'), strict=False) #model_stage1_epoch20.pth can be downloaded through https://drive.google.com/file/d/1qly7uKTEvh_wVXHYvil-ZV0w3JUcAEHB/view?usp=drive_link
    #print(pre_model)
    in_channels = 32
    spatial_size = 23
    model = conv_model(in_channels, spatial_size, args)
    #print(model)
    model.to(device)

    model.load_state_dict(torch.load(os.path.join(args.data_dir, f'best_weights.pt')))
    rmse, pearson, spearman, test_df = test(pre_model, model, test_loader, device)
    test_df.to_pickle(os.path.join(args.output_dir, 'test_results.pkl'))
    print('Test RMSE: {:.7f}, Pearson R: {:.7f}, Spearman R: {:.7f}'.format(
        rmse, pearson, spearman))
    test_file = os.path.join(args.output_dir, f'test_results.txt')
    with open(test_file, 'a+') as out:
        out.write('{}\t{:.7f}\t{:.7f}\t{:.7f}\n'.format(
            args.random_seed, rmse, pearson, spearman))



if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--output_dir', type=str, default='output_test')
    parser.add_argument('--unobserved', action='store_true', default=False)

    parser.add_argument('--learning_rate', type=float, default=0.0001)
    parser.add_argument('--conv_drop_rate', type=float, default=0.1)
    parser.add_argument('--fc_drop_rate', type=float, default=0.25)

    parser.add_argument('--num_conv', type=int, default=4)
    parser.add_argument('--batch_norm', action='store_true', default=False)
    parser.add_argument('--no_dropout', action='store_true', default=False)

    parser.add_argument('--batch_size', type=int, default=20)
    parser.add_argument('--random_seed', type=int, default=int(np.random.randint(1, 10e6)))

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Set up output dir
    args.output_dir = os.path.join(args.output_dir, 'output')
    assert args.output_dir != None
    if args.unobserved:
        args.output_dir = os.path.join(args.output_dir, 'None')
        os.makedirs(args.output_dir, exist_ok=True)
    else:
        num = 0
        while True:
            dirpath = os.path.join(args.output_dir, str(num))
            if os.path.exists(dirpath):
                num += 1
            else:
                args.output_dir = dirpath
                print('Creating output directory {:}'.format(args.output_dir))
                os.makedirs(args.output_dir)
                break

    run_test(args, device)

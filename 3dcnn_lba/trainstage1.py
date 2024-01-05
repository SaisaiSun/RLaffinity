# trainstage1.py
import torch,argparse,os
import net,config,loaddataset
from lba.datasets import LMDBDataset
from data1 import CNN3D_TransformLBA


# train stage one
def train(args):
    if torch.cuda.is_available() and config.use_gpu:
        DEVICE = torch.device("cuda:" + str(config.gpu_name))
        torch.backends.cudnn.benchmark = True
    else:
        DEVICE = torch.device("cpu")
    print("current deveice:", DEVICE)

    dataset_path = '/Users/saisaisun/Downloads/RNA-affinity-pretrain/data' #replace it with the pre-training data (.mdb) path'
    train_dataset = LMDBDataset(dataset_path, transform=CNN3D_TransformLBA(radius=20.0))
    train_data = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, drop_last=True)
    
    for data1,data2 in train_data:
        in_channels, spatial_size = data1.size()[1:3]
        print('num channels: {:}, spatial size: {:}'.format(in_channels, spatial_size))
        #print(data)
        break

    #train_dataset=loaddataset.PreDataset(root='dataset', train=True, transform=config.train_transform, download=True)
    #train_data=torch.utils.data.DataLoader(train_dataset,batch_size=args.batch_size, shuffle=True, num_workers=16 , drop_last=True)

    model =net.SimCLRStage1().to(DEVICE)
    print(model)
    lossLR=net.Loss(args.batch_size).to(DEVICE)
    optimizer=torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-6)

    os.makedirs(config.save_path, exist_ok=True)
    for epoch in range(1,args.max_epoch+1):
        model.train()
        total_loss = 0
        for batch,(imgL,imgR) in enumerate(train_data):
            imgL,imgR=imgL.to(DEVICE),imgR.to(DEVICE)
            #print(model(imgL))

            pre_L=model(imgL)
            pre_R=model(imgR)

            loss=lossLR(pre_L,pre_R)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print("epoch", epoch, "batch", batch, "loss:", loss.detach().item())
            total_loss += loss.detach().item()

        print("epoch loss:",total_loss/len(train_dataset)*args.batch_size)

        with open(os.path.join(config.save_path, "stage1_loss.txt"), "a") as f:
            f.write(str(total_loss/len(train_dataset)*args.batch_size) + " ")

        if epoch % 5==0:
            torch.save(model.state_dict(), os.path.join(config.save_path, 'model_stage1_epoch' + str(epoch) + '.pth'))
        #if epoch == max_epoch:
        #    torch.save(model, os.path.join(config.save_path, 'model_stage1_final.pth'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train SimCLR')
    parser.add_argument('--batch_size', default=100, type=int, help='')
    parser.add_argument('--max_epoch', default=20, type=int, help='')

    args = parser.parse_args()
    train(args)


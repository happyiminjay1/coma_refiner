import pickle
import argparse
import os
import os.path as osp
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch_geometric.transforms as T
from psbody.mesh import Mesh
import openmesh as om
import trimesh

from models import AE
from datasets import MeshData, HOI_info
from utils import utils, writer, train_eval, DataLoader, mesh_sampling


class ConcatDataset(torch.utils.data.Dataset):
    def __init__(self, *datasets):
        self.datasets = datasets

    def __getitem__(self, i):
        return tuple(d[i] for d in self.datasets)

    def __len__(self):
        return min(len(d) for d in self.datasets)

parser = argparse.ArgumentParser(description='mesh autoencoder')
parser.add_argument('--exp_name', type=str, default='interpolation_exp')
parser.add_argument('--ori_exp_name', type=str, default='ori_interpolation_exp')
parser.add_argument('--dataset', type=str, default='CoMA')
parser.add_argument('--split', type=str, default='interpolation')
parser.add_argument('--test_exp', type=str, default='bareteeth')
parser.add_argument('--n_threads', type=int, default=4)
parser.add_argument('--device_idx', type=int, default=0)

# network hyperparameters
parser.add_argument('--out_channels',
                    nargs='+',
                    default=[16, 16, 16, 32],
                    type=int)
parser.add_argument('--loss_weight',
                    nargs='+',
                    default=[1, 1, 1, 1],
                    type=int)
parser.add_argument('--latent_channels', type=int, default=64)
parser.add_argument('--in_channels', type=int, default=3)
parser.add_argument('--output_channels', type=int, default=3)
parser.add_argument('--K', type=int, default=6)

# optimizer hyperparmeters
parser.add_argument('--optimizer', type=str, default='SGD')
parser.add_argument('--lr', type=float, default=8e-3)
parser.add_argument('--lr_decay', type=float, default=0.99)
parser.add_argument('--decay_step', type=int, default=1)
parser.add_argument('--weight_decay', type=float, default=0.0005)

# training hyperparameters
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--epochs', type=int, default=300)


# others
parser.add_argument('--seed', type=int, default=1)

args = parser.parse_args()

args.work_dir = osp.dirname(osp.realpath(__file__))
args.data_fp = osp.join(args.work_dir, 'data', args.dataset)
args.out_dir = osp.join(args.work_dir, 'out', args.exp_name)
args.checkpoints_dir = osp.join(args.out_dir, 'checkpoints')
print(args)

# Namespace(K=6, batch_size=16, checkpoints_dir='/scratch/minjay/coma/out/interpolation_exp/checkpoints', 
# data_fp='/scratch/minjay/coma/data/CoMA', dataset='CoMA', decay_step=1, device_idx=0, epochs=300, 
# exp_name='interpolation_exp', in_channels=3, latent_channels=8, lr=0.008, lr_decay=0.99, n_threads=4, 
# optimizer='Adam', out_channels=[16, 16, 16, 32], out_dir='/scratch/minjay/coma/out/interpolation_exp', 
# seed=1, split='interpolation', test_exp='bareteeth', weight_decay=0, work_dir='/scratch/minjay/coma')

utils.makedirs(args.out_dir)
utils.makedirs(args.checkpoints_dir)

writer = writer.Writer(args)
device = torch.device('cuda', args.device_idx)
torch.set_num_threads(args.n_threads)

# deterministic
torch.manual_seed(args.seed)
cudnn.benchmark = False
cudnn.deterministic = True

# load dataset
template_fp = osp.join('template', 'hand_mesh_template.obj')

meshdata = MeshData(args.data_fp,
                    template_fp,
                    split=args.split,
                    test_exp=args.test_exp)

#print(meshdata.mean.shape)
#print(meshdata.std.shape)

HOI_info_train = HOI_info('/scratch/minjay/coma_refiner/data/CoMA/processed/interpolation/training_total.pkl')
HOI_info_test = HOI_info('/scratch/minjay/coma_refiner/data/CoMA/processed/interpolation/test_total.pkl')

train_loader_hoi = torch.utils.data.DataLoader(
                 HOI_info_train,batch_size=args.batch_size)

test_loader_hoi = torch.utils.data.DataLoader(
                 HOI_info_test,batch_size=args.batch_size)

train_loader = DataLoader(meshdata.train_dataset,
                          batch_size=args.batch_size)

test_loader = DataLoader(meshdata.test_dataset, batch_size=args.batch_size)

# for hoi, hand_mesh in zip(train_loader_hoi,train_loader) :
#     print(hoi[0][0].shape)
#     print(hoi[1][0].shape)
#     print(hoi[2][0].shape)
#     print(hoi[3][1])
#     #self.data[idx]['hand_mesh'].x, 
#     #self.data[idx]['hoi_feature'], 
#     #self.data[idx]['contactmap'], 
#     #self.data[idx]['taxonomy']

#     print(hand_mesh.x[0].shape)
#     torch.Size([778, 3])
#     torch.Size([778, 24])
#     torch.Size([778, 1])
#     tensor(6)
#     torch.Size([778, 3])
#     exit(0)
#     #om.write_mesh('verts1.obj', om.TriMesh(verts2.numpy(), meshdata.template_face))
#     #om.write_mesh('verts2.obj', om.TriMesh(verts1.numpy(), meshdata.template_face))
#     #exit(0)

# generate/load transform matrices
transform_fp = osp.join(args.data_fp, 'transform.pkl')
print(transform_fp)
if not osp.exists(transform_fp):
    print('Generating transform matrices...')
    mesh = Mesh(filename=template_fp)
    ds_factors = [4, 2, 2, 2]
    _, A, D, U, F = mesh_sampling.generate_transform_matrices(mesh, ds_factors)
    tmp = {'face': F, 'adj': A, 'down_transform': D, 'up_transform': U}
    with open(transform_fp, 'wb') as fp:
        pickle.dump(tmp, fp)
    print('Done!')
    print('Transform matrices are saved in \'{}\''.format(transform_fp))
else:
    with open(transform_fp, 'rb') as f:
        tmp = pickle.load(f, encoding='latin1')

edge_index_list = [utils.to_edge_index(adj).to(device) for adj in tmp['adj']]
down_transform_list = [
    utils.to_sparse(down_transform).to(device)
    for down_transform in tmp['down_transform']
]
up_transform_list = [
    utils.to_sparse(up_transform).to(device)
    for up_transform in tmp['up_transform']
]

model = AE(args.in_channels,
           args.out_channels,
           args.output_channels,
           args.latent_channels,
           edge_index_list,
           down_transform_list,
           up_transform_list,
           K=args.K).to(device)

print(model)

#checkpoint = torch.load(f'/scratch/minjay/coma_refiner/out/{args.ori_exp_name}/checkpoints/checkpoint_300.pt')

if 'new' not in args.exp_name :
    print('############## pretrained_model_loaded #################')
    model.load_state_dict(checkpoint['model_state_dict'])

testing_env = False

if testing_env :
    checkpoint = torch.load(f'/scratch/minjay/coma_refiner/out/{args.exp_name}/checkpoints/checkpoint_300.pt')

    print('############## pretrained_model_loaded #################')
    model.load_state_dict(checkpoint['model_state_dict'])

    #/scratch/minjay/coma_refiner/out/interpolation_exp_mano_contact[32, 32, 32, 64] 128 new both_none 2

    #/scratch/minjay/coma/out/interpolation_exp_mano_contact[32, 32, 32, 64] 128 new both_none 

else :
    print('start_new!!!!!')
    

if args.optimizer == 'Adam':
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=args.lr,
                                 weight_decay=args.weight_decay)
elif args.optimizer == 'SGD':
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=args.lr,
                                weight_decay=args.weight_decay,
                                momentum=0.9)
else:
    raise RuntimeError('Use optimizers of SGD or Adam')
scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                            args.decay_step,
                                            gamma=args.lr_decay)

train_eval.run(model,train_loader, train_loader_hoi, test_loader, test_loader_hoi, args.epochs, optimizer, scheduler, writer, meshdata,  args.exp_name, device, args)

train_eval.tester(model,train_loader, train_loader_hoi, test_loader, test_loader_hoi, args.epochs, optimizer, scheduler, writer, meshdata,  args.exp_name, device, args)

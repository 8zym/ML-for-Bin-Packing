import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Model Params')
    # 训练设置
    parser.add_argument('--batch', default=128, type=int, help='batch size')
    parser.add_argument('--epoch', default=30000, type=int, help='number of epochs')
    parser.add_argument('--device', type=str, default='cpu', help='cuda device')
    parser.add_argument('--tstEpoch', default=1, type=int, help='number of epoch to test while training')
    parser.add_argument('--save', type=str, default='./Save', help='save path')
    parser.add_argument('--log-url', type=str, default = 'log/test.log', help='log-url')
    parser.add_argument('--log_interval', default=1000, type=int, help='log interval')
    parser.add_argument('--full_eval_mode', default=0, type=int, help='do evaluation on the whole validation and the test data')
    # 模型设置
    parser.add_argument('--actor_lr', default=4e-5, type=float, help='actor learning rate')
    parser.add_argument('--critic_lr', default=1e-4, type=float, help='critic learning rate')
    parser.add_argument('--lr_warmup', default=100, type=int, help='linea+y increase LR from 0 during first lr_warmup updates warmup_epochs=lr_warmup/(block_size/nsteps)')
    parser.add_argument('--nsteps', default=5, type=int, help='GAE rolling out steps')
    parser.add_argument('--encoder_layers', default=3, type=int, help='number of layers in encoder')
    parser.add_argument('--decoder_layers', default=3, type=int, help='number of layers in decoder')
    parser.add_argument('--c_encoder_layers', default=2, type=int, help='number of layers')
    parser.add_argument('--hidden_size', default=128, type=int, help='hidden size in heightmap encoder')
    parser.add_argument('--inner_hidden_size', default=512, type=int, help='inner_hidden_size of FF layers')
    parser.add_argument('--head_hidden', default=128, type=int, help='head hidden dim(select and rotate)')
    parser.add_argument('--nb_heads', default=8, type=int, help='number of heads in transformer')
    parser.add_argument('--gamma', default=1, type=float, help='GAE gamma: reward discount factor')
    parser.add_argument('--lam', default=0.98, type=float, help='lam for General Advantage Estimation')
    parser.add_argument('--target_entropy', default=-0.6, type=float, help='position target entropy for entropy regularization')
    parser.add_argument('--grad_clip', default=5, type=float, help='clip gradient of each module parameters by a given value')
    parser.add_argument('--normalization', default='batch', type=str, help='batch or instance normalization')
    parser.add_argument('--dropout', default=0.4, type=float, help='dropout rate of ReLU and attention')
    # 数据集设置
    parser.add_argument('--dataset_path', type=str, default=None, help='dataset path, None for generating dataset')
    parser.add_argument('--dataset_size', default=128, type=int, help='number of samples in dataset, only used for generating dataset')
    parser.add_argument('--bin_x', default=20, type=int, help='bin length, used for generating dataset and training')
    parser.add_argument('--bin_y', default=20, type=int, help='bin width, used for generating dataset and training')
    parser.add_argument('--bin_z', default=20, type=int, help='bin height, only uesd for generating dataset, training will ignore z limit')
    parser.add_argument('--box_min', default=5, type=int, help='minimum box size, used for generating dataset and training')
    parser.add_argument('--box_max', default=10, type=int, help='maximum box size, used for generating dataset and training')
    return parser.parse_args()
args = parse_args()
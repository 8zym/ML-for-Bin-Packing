import torch
from typing import NamedTuple
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import move_to
from Params import args

class PackAction():
    # (batch, 1)

    def __init__(self, batch_size):
        self.index = torch.zeros(batch_size, 1, device=args.device)
        self.x = torch.empty(batch_size, 1, device=args.device).fill_(-2)  # set to -2
        self.y = torch.empty(batch_size, 1, device=args.device).fill_(-2)
        self.z = torch.empty(batch_size, 1, device=args.device).fill_(-2)
        self.rotate = torch.zeros(batch_size, 1, device=args.device)
        self.updated_shape = torch.empty(batch_size, 3, device=args.device)
        self.sp = torch.FloatTensor(batch_size, 1).fill_(1)
        self.sp = move_to(self.sp, args.device)
        '''
        0: no rotate
        1: (x,y,z) -> (y,x,z)
        2: (x,y,z) -> (y,z,x)
        3: (x,y,z) -> (z,y,x)
        4: (x,y,z) -> (z,x,y)
        5: (x,y,z) -> (x,z,y)
        '''

    def set_index(self, selected):
        self.index = selected

    def set_rotate(self, rotate):
        self.rotate = rotate

    def set_shape(self, length, width, height):
        # (batch, 3)
        self.updated_shape = torch.stack([length, width, height], dim=-1)

    def get_shape(self):
        return self.updated_shape

    def set_pos(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

    def get_packed(self):
        return torch.cat((self.sp,self.updated_shape, self.x, self.y, self.z), dim=-1)

    def reset(self):
        self.__init__(self.index.size(0))

    def __call__(self):
        return {'index': self.index,
                'rotate': self.rotate,
                'x': self.x,
                'y': self.y}

    def __len__(self):
        return self.index.size(0)

def push_to_tensor_alternative(tensor, x):
    return torch.cat((tensor[:, 1:, :], x), dim=1)

class StatePack3D():

    def __init__(self, batch_size, box_num):
        self.container_size = [args.bin_x, args.bin_y, 2*args.bin_z]
        self.action = PackAction(batch_size)
        self.blocks_num = box_num
        self.block_dim = 3
        self.heightmap = np.zeros([batch_size,args.bin_x,args.bin_y]).astype(int)
        self.batch_size= batch_size
        self.packed_state = torch.zeros(
            batch_size, box_num, 7, dtype=torch.float, device=args.device)
        self.index=0
        self.packed_rotate = torch.zeros(
            batch_size, box_num, 1, dtype=torch.int64, device=args.device)
        self.total_rewards = torch.zeros(
            batch_size, dtype=torch.float, device=args.device)
        self.container = np.zeros([batch_size,args.bin_x,args.bin_y,2*args.bin_z]).astype(int)
        self.positions = np.zeros((batch_size,box_num, self.block_dim)).astype(int)
        self.reward_type = "C-soft"
        self.stable = np.zeros((batch_size,box_num),dtype=bool)
        self.valid_size = [0]*batch_size
        self.empty_size = [0]*batch_size
        
    def put_reward(self, reward):
        self.total_rewards += reward

    def get_rewards(self):
        return self.total_rewards

    def update_env(self, batch, batch_size):
        # batch, box_num, 3
        sp_initial=torch.FloatTensor(batch_size,self.blocks_num,1).fill_(0)
        sp_initial=move_to(sp_initial,args.device)
        position_initial=torch.FloatTensor(batch_size,self.blocks_num,3).fill_(0)
        position_initial=move_to(position_initial,args.device)
        self.packed_state=torch.cat([sp_initial,batch,position_initial],dim=2)
    
    def update_select(self, selected):
        self.action.set_index(selected)
        box_length, box_width, box_height = self._get_action_box_shape()
        self.action.set_shape(box_length, box_width, box_height)

    def _get_action_box_shape(self):
        select_index = self.action.index.long()

        box_raw_l = self.packed_state[:, :, 1].squeeze(-1)
        box_raw_w = self.packed_state[:, :, 2].squeeze(-1)
        box_raw_h = self.packed_state[:, :, 3].squeeze(-1)

        box_length = torch.gather(box_raw_l, -1, select_index).squeeze(-1)
        box_width = torch.gather(box_raw_w, -1, select_index).squeeze(-1)
        box_height = torch.gather(box_raw_h, -1, select_index).squeeze(-1)

        return box_length, box_width, box_height

    def update_rotate(self, rotate):

        self.action.set_rotate(rotate)

        # there are 5 rotations except the original one
        rotate_types = 5
        batch_size = rotate.size()[0]

        rotate_mask = torch.empty((rotate_types, batch_size), dtype=torch.bool)
        rotate_mask = move_to(rotate_mask, args.device)
        select_index = self.action.index.long()

        box_raw_x = self.packed_state[:, :, 1].squeeze(-1)
        box_raw_y = self.packed_state[:, :, 2].squeeze(-1)
        box_raw_z = self.packed_state[:, :, 3].squeeze(-1)

        # (batch)  get the original box shape
        box_length = torch.gather(box_raw_x, -1, select_index).squeeze(-1)
        box_width = torch.gather(box_raw_y, -1, select_index).squeeze(-1)
        box_height = torch.gather(box_raw_z, -1, select_index).squeeze(-1)

        for i in range(rotate_types):
            rotate_mask[i] = rotate.squeeze(-1).eq(i + 1)

        # rotate in 5 directions one by one
        # (x,y,z)->(y,x,z)
        # (x,y,z)->(y,z,x)
        # (x,y,z)->(z,y,x)
        # (x,y,z)->(z,x,y)
        # (x,y,z)->(x,z,y)
        inbox_length = box_length
        inbox_width = box_width
        inbox_height = box_height
        for i in range(rotate_types):

            if i == 0:
                box_l_rotate = box_width
                box_w_rotate = box_length
                box_h_rotate = box_height
            elif i == 1:
                box_l_rotate = box_width
                box_w_rotate = box_height
                box_h_rotate = box_length
            elif i == 2:
                box_l_rotate = box_height
                box_w_rotate = box_width
                box_h_rotate = box_length
            elif i == 3:
                box_l_rotate = box_height
                box_w_rotate = box_length
                box_h_rotate = box_width
            elif i == 4:
                box_l_rotate = box_length
                box_w_rotate = box_height
                box_h_rotate = box_width

            box_l_rotate = torch.masked_select(
                box_l_rotate, rotate_mask[i])
            box_w_rotate = torch.masked_select(
                box_w_rotate, rotate_mask[i])
            box_h_rotate = torch.masked_select(
                box_h_rotate, rotate_mask[i])

            inbox_length = inbox_length.masked_scatter(
                rotate_mask[i], box_l_rotate)
            inbox_width = inbox_width.masked_scatter(
                rotate_mask[i], box_w_rotate)
            inbox_height = inbox_height.masked_scatter(
                rotate_mask[i], box_h_rotate)
        self.packed_rotate[torch.arange(0, rotate.size(
            0)), select_index.squeeze(-1), 0] = rotate.squeeze(-1)

        self.action.set_shape(inbox_length, inbox_width, inbox_height)

    def update_pack(self):
        select_index = self.action.index.squeeze(-1).long().tolist()

        x=torch.tensor(self.positions[:,self.index,0]).unsqueeze(-1).float()
        y=torch.tensor(self.positions[:,self.index,1]).unsqueeze(-1).float()
        z=torch.tensor(self.positions[:,self.index,2]).unsqueeze(-1).float()
        x=move_to(x,args.device)
        y=move_to(y,args.device)
        z=move_to(z,args.device)
        self.action.set_pos(x, y, z)

        packed_box = self.action.get_packed()
       
        mid=self.packed_state.clone()
        self.packed_state=mid
        for i in range(self.batch_size):
            self.packed_state[i][select_index[i]]=packed_box[i]

        self.index += 1

    def _get_z(self, x, y):

        inbox_length = self.action.get_packed()[:, 0]
        inbox_width = self.action.get_packed()[:, 1]

        in_back = torch.min(x.squeeze(-1), x.squeeze(-1) + inbox_length)
        in_front = torch.max(x.squeeze(-1), x.squeeze(-1) + inbox_length)
        in_left = torch.min(y.squeeze(-1), y.squeeze(-1) + inbox_width)
        in_right = torch.max(y.squeeze(-1), y.squeeze(-1) + inbox_width)

        box_length = self.packed_cat[:, :, 0]
        box_width = self.packed_cat[:, :, 1]
        box_height = self.packed_cat[:, :, 2]

        box_x = self.packed_cat[:, :, 3]
        box_y = self.packed_cat[:, :, 4]
        box_z = self.packed_cat[:, :, 5]

        box_back = torch.min(box_x, box_x + box_length)
        box_front = torch.max(box_x, box_x + box_length)
        box_left = torch.min(box_y, box_y + box_width)
        box_right = torch.max(box_y, box_y + box_width)
        box_top = torch.max(box_z, box_z + box_height)

        in_back = in_back.unsqueeze(-1).repeat([1, self.packed_cat.size()[1]])
        in_front = in_front.unsqueeze(-1).repeat(
            [1, self.packed_cat.size()[1]])
        in_left = in_left.unsqueeze(-1).repeat([1, self.packed_cat.size()[1]])
        in_right = in_right.unsqueeze(-1).repeat(
            [1, self.packed_cat.size()[1]])

        is_back = torch.gt(box_front, in_back)
        is_front = torch.lt(box_back, in_front)
        is_left = torch.gt(box_right, in_left)
        is_right = torch.lt(box_left, in_right)

        is_overlaped = is_back * is_front * is_left * is_right
        non_overlaped = ~is_overlaped

        overlap_box_top = box_top.masked_fill(non_overlaped, 0)

        max_z, _ = torch.max(overlap_box_top, -1, keepdim=True)

        return max_z

    def get_boundx(self):
        batch_size = self.packed_state.size()[0]
        front_bound = torch.ones(
            batch_size, device=self.packed_state.device) - self.action.get_shape()[:, 0]

        return front_bound

    def get_boundy(self):

        batch_size = self.packed_state.size()[0]
        right_bound = torch.ones(
            batch_size, device=self.packed_state.device) - self.action.get_shape()[:, 1]

        return right_bound

    def get_height(self):


        return np.max(self.heightmap,axis=(1,2))

    def get_gap_size(self):

        bin_volumn = self.get_height() * 100.0

        gap_volumn = bin_volumn - self.valid_size
        gap_volumn=torch.tensor(gap_volumn)
        gap_volumn=move_to(gap_volumn,args.device)
        return gap_volumn

    def get_gap_ratio(self):

        bin_volumn = self.get_height() *100.0
        bin_volumn=torch.tensor(bin_volumn)
        bin_volumn=move_to(bin_volumn,args.device)
        ebselong=torch.tensor([0.0001])
        ebselong=move_to(ebselong,args.device)

        gap_ratio = self.get_gap_size() /( bin_volumn+ebselong)

        return gap_ratio

    def get_graph(self):
        return self.packed_cat
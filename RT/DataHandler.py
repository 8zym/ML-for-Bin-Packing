import csv
import numpy as np
from Params import args
import random

def BPP_generator(box_num, bin_x, bin_y, bin_z):
    I = [
        {
            "l": bin_x,   # 物品长
            "w": bin_y,   # 物品宽
            "h": bin_z,   # 物品高
        }
    ]
    # 在达到N之前持续分裂物品
    while len(I) < box_num:
        # 按体积加权从I中选取物品
        volumes = [item["l"] * item["w"] * item["h"] for item in I]
        total_volume = sum(volumes)
        r = random.random() * total_volume
        cumulative = 0.0
        chosen_index = None
        for idx, vol in enumerate(volumes):
            cumulative += vol
            if r <= cumulative:
                chosen_index = idx
                break
        
        # 选中的物品
        chosen_item = I.pop(chosen_index)
        
        # 根据边长选择分割轴(以边长为权重)
        edges = [chosen_item["l"], chosen_item["w"], chosen_item["h"]]
        edge_sum = sum(edges)
        r_edge = random.random() * edge_sum
        cumulative_edge = 0.0
        axis = None  # 0->x, 1->y, 2->z
        for i, e_len in enumerate(edges):
            cumulative_edge += e_len
            if r_edge <= cumulative_edge:
                axis = i
                break
        
        # 沿所选轴方向的中点附近随机选择分割位置
        # 为了保证不会出现太极端的值，以物品一半长度为参考，在中点附近±1/4范围随机选取
        axis_length = edges[axis]
        half_axis_length = axis_length / 2.0
        split_offset = int((random.random() - 0.5) * (axis_length / 2.0))
        split_pos1 = half_axis_length + split_offset
        split_pos2 = half_axis_length - split_offset
        
        # 将物品分割为两个子物品
        if axis == 0:
            child1 = {"l": split_pos1,"w": chosen_item["w"],"h": chosen_item["h"]}
            child2 = {"l": split_pos2,"w": chosen_item["w"],"h": chosen_item["h"]}
        if axis == 1:
            child1 = {"l": chosen_item["l"],"w": split_pos1,"h": chosen_item["h"]}
            child2 = {"l": chosen_item["l"],"w": split_pos2,"h": chosen_item["h"]}
        if axis == 2:    
            child1 = {"l": chosen_item["l"],"w": chosen_item["w"],"h": split_pos1}
            child2 = {"l": chosen_item["l"],"w": chosen_item["w"],"h": split_pos2}
        
        # 旋转子物品
        def rotate_item(l, w, h):
            # 随机旋转操作：随机打乱(l,w,h)顺序
            dims = [l, w, h]
            random.shuffle(dims)
            return dims[0], dims[1], dims[2]
        
        child1["l"], child1["w"], child1["h"] = rotate_item(child1["l"], child1["w"], child1["h"])
        child2["l"], child2["w"], child2["h"] = rotate_item(child2["l"], child2["w"], child2["h"])

        # 加入到I中
        I.append(child1)
        I.append(child2)                
    
    array = np.array([[d["l"], d["w"], d["h"]] for d in I])
    return array

class DataHandler:
    def __init__(self):
        if args.dataset_path is None:
            # generate dataset: 
            self.datalist = []
            if args.box_min >= args.box_max:
                raise ValueError('min should be less than max.')
            steps = int(np.ceil(args.dataset_size / args.batch))
            for i in range(steps):
                N = random.randint(args.box_min, args.box_max)
                # batch_size, box_num, 3
                batch_data = np.array([BPP_generator(N, args.bin_x, args.bin_y, args.bin_z) for _ in range(min(args.batch, args.dataset_size - i * args.batch))])
                self.datalist.append(batch_data)
        else:
            # read dataset from csv
            # TODO: implement reading dataset from csv
            # TODO: init args such as box_num, dataset_size
            raise NotImplementedError('Dataset not supported yet.')
        
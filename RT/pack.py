import torch
from torch import nn
import copy
from torch.nn import DataParallel
import torch.nn.functional as F
from utils import move_to
import torch
import itertools
from matplotlib.path import Path
import numpy as np
from scipy.spatial import ConvexHull
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from Params import args
import logging

def calc_one_position_lb_greedy_3d(block, block_index, container_size, positions, heightmap, valid_size):

    block_x, block_y, block_z = block
    valid_size += block_x * block_y * block_z

    # get empty-maximal-spaces list from heightmap
    # each ems represented as a left-bottom corner
    ems_list = []
    # hm_diff: height differences of neightbor columns, padding 0 in the front
    # x coordinate
    hm_diff_x = np.insert(heightmap, 0, heightmap[0, :], axis=0)
    hm_diff_x = np.delete(hm_diff_x, len(hm_diff_x) - 1, axis=0)
    hm_diff_x = heightmap - hm_diff_x
    # y coordinate
    hm_diff_y = np.insert(heightmap, 0, heightmap[:, 0], axis=1)
    hm_diff_y = np.delete(hm_diff_y, len(hm_diff_y.T) - 1, axis=1)
    hm_diff_y = heightmap - hm_diff_y

    # get the xy coordinates of all left-deep-bottom corners
    ems_x_list = np.array(np.nonzero(hm_diff_x)).T.tolist()
    ems_y_list = np.array(np.nonzero(hm_diff_y)).T.tolist()
    ems_xy_list = []
    ems_xy_list.append([0, 0])
    for xy in ems_x_list:
        x, y = xy
        if y != 0 and [x, y - 1] in ems_x_list:
            if heightmap[x, y] == heightmap[x, y - 1] and \
                    hm_diff_x[x, y] == hm_diff_x[x, y - 1]:
                continue
        ems_xy_list.append(xy)
    for xy in ems_y_list:
        x, y = xy
        if x != 0 and [x - 1, y] in ems_y_list:
            if heightmap[x, y] == heightmap[x - 1, y] and \
                    hm_diff_x[x, y] == hm_diff_x[x - 1, y]:
                continue
        if xy not in ems_xy_list:
            ems_xy_list.append(xy)
    logging.debug(f"ems_xy_list: {ems_xy_list}")

    # sort by y coordinate, then x
    def y_first(pos):
        return pos[1]
    # 升序
    ems_xy_list.sort(key=y_first, reverse=False)

    # get ems_list
    for xy in ems_xy_list:
        x, y = xy
        if x + block_x > container_size[0] or \
                y + block_y > container_size[1]: continue
        # print(heightmap[x:x + block_x, y:y + block_y])
        z = np.max(heightmap[x:x + block_x, y:y + block_y])
        ems_list.append([x, y, z])

    logging.debug(f"ems_list: {ems_list}")

    # firt consider the most bottom, sort by z coordinate, then y last x
    ems_list.sort(key=lambda pos: (pos[2], pos[1], pos[0]))

    # if no ems found
    if len(ems_list) == 0:
        valid_size -= block_x * block_y * block_z
        # TODO: how to deal with stable and return values
        return positions, heightmap, valid_size

    # update the dynamic parameters
    best_ems_index = 0
    _x, _y, _z = ems_list[best_ems_index]
    # container[_x:_x + block_x, _y:_y + block_y, _z:_z + block_z] = block_index + 1
    # container[_x:_x + block_x, _y:_y + block_y, 0:_z][under_space_mask[best_ems_index]] = -1
    positions[block_index] = torch.tensor([_x, _y, _z])
    heightmap[_x:_x + block_x, _y:_y + block_y] = _z + block_z
    # stable[block_index] = is_stable_ems[best_ems_index]
    # heightmap_ems[block_index][_x:_x + block_x, _y:_y + block_y] = _z + block_z

    # empty_size = empty_ems[best_ems_index]

    return positions, heightmap, valid_size

def is_stable(block, position, container):
    '''
    check for 3D packing
    ----
    '''
    if (position[2]==0):
        return True
    x_1 = position[0]
    x_2 = x_1 + block[0] - 1
    y_1 = position[1]
    y_2 = y_1 + block[1] - 1
    z = position[2] - 1
    obj_center = ( (x_1+x_2)/2, (y_1+y_2)/2 )

    # valid points right under this object
    points = []
    for x in range(x_1, x_2+1):
        for y in range(y_1, y_2+1):
            if (container[x][y][z] > 0):
                points.append([x, y])
    if(len(points) > block[0]*block[1]/2):
        return True
    if(len(points)==0 or len(points)==1):
        return False
    elif(len(points)==2):
        # whether the center lies on the line of the two points
        a = obj_center[0] - points[0][0]
        b = obj_center[1] - points[0][1]
        c = obj_center[0] - points[1][0]
        d = obj_center[1] - points[1][1]
        # same ratio and opposite signs
        if (b==0 or d==0):
            if (b!=d): return False
            else: return (a<0)!=(c<0)
        return ( a/b == c/d and (a<0)!=(c<0) and (b<0)!=(d<0) )
    else:
        # calculate the convex hull of the points
        points = np.array(points)
        try:
            convex_hull = ConvexHull(points)
        except:
            # error means co-lines
            min_point = points[np.argmin( points[:,0] )]
            max_point = points[np.argmax( points[:,0] )]
            points = np.array( (min_point, max_point) )
            a = obj_center[0] - points[0][0]
            b = obj_center[1] - points[0][1]
            c = obj_center[0] - points[1][0]
            d = obj_center[1] - points[1][1]
            if (b==0 or d==0):
                if (b!=d): return False
                else: return (a<0)!=(c<0)
            return ( a/b == c/d and (a<0)!=(c<0) and (b<0)!=(d<0) )

        hull_path = Path(points[convex_hull.vertices])
        return hull_path.contains_point((obj_center))

def pack_step(modules, state):
    hm = np.zeros((state.batch_size, 2, args.bin_x, args.bin_y))
    for i in range(state.batch_size):
        hm_diff_x = np.insert(state.heightmap[i], 0, state.heightmap[i][0, :], axis=0)
        hm_diff_x = np.delete(hm_diff_x, len(hm_diff_x) - 1, axis=0)
        hm_diff_x = state.heightmap[i] - hm_diff_x

        hm_diff_y = np.insert(state.heightmap[i], 0, state.heightmap[i][:, 0], axis=1)
        hm_diff_y = np.delete(hm_diff_y, len(hm_diff_y.T) - 1, axis=1)
        hm_diff_y = state.heightmap[i] - hm_diff_y
        # combine the two heightmaps
        hm[i][0] = hm_diff_x
        hm[i][1] = hm_diff_y

    # (batch, 2, bin_x, bin_y)
    hm=torch.tensor(hm).float()
    hm=move_to(hm, args.device)
    actor_modules = modules['actor']

    logging.debug(f"packed_state_shape: {state.packed_state.shape}")
    # batch, box_num, 7
    actor_encoder_out = actor_modules['encoder'](state.packed_state)
    logging.debug(f"actor_encoder_out_shape: {actor_encoder_out.shape}")
    # batch, box_num, hidden_size
    if state.index == 0:
        actor_encoder_out_select = torch.zeros(state.batch_size, 1, args.hidden_size)
        actor_encoder_out_select = move_to(actor_encoder_out_select, args.device)
    else:
        actor_encoder_out_select = torch.masked_select(actor_encoder_out, state.packed_state[:, :, 0].unsqueeze(-1).bool())
        actor_encoder_out_select = actor_encoder_out_select.view(state.batch_size, state.index, args.hidden_size)
    # batch, index, hidden_size

    actor_encoderheightmap_out=actor_modules["encoderheightmap"](hm)
    logging.debug(f"actor_encoderheightmap_out_shape: {actor_encoderheightmap_out.shape}")
    # batch, 1, hidden_size
    q_select=torch.cat((actor_encoder_out_select,actor_encoderheightmap_out),dim=1)
    logging.debug(f"q_select_shape: {q_select.shape}")
    # (batch, index+1, hidden_size)
    q_select=torch.mean(q_select,dim=1).unsqueeze(1)
    logging.debug(f"q_select_shape: {q_select.shape}")
    # (batch, 1, hidden_size)
    s_out = actor_modules['s_decoder'](q_select, actor_encoder_out)
    logging.debug(f"s_out_shape: {s_out.shape}")
    # (batch, 1, box_num)
    s_log_p, selected = _select_step(s_out.squeeze(1), state.packed_state[:,:,0].bool())
    logging.debug(f"s_log_p_shape: {s_log_p.shape}")
    # batch, box_num
    logging.debug(f"selected_shape: {selected.shape}")
    # (batch, 1)

    # select (batch)
    logging.debug(f"select box{state.index}")
    state.update_select(selected)

    selected=selected.expand(state.batch_size,args.hidden_size).unsqueeze(1)
    logging.debug(f"selected_shape(after expand): {selected.shape}")
    # batch, 1, hidden_size
    actor_encoder_out_rotation=torch.gather(actor_encoder_out,1,selected)
    logging.debug(f"actor_encoder_out_rotation_shape: {actor_encoder_out_rotation.shape}")
    # batch, 1, hidden_size

    q_rotation=torch.cat((actor_encoder_out_rotation,actor_encoderheightmap_out),dim=1)
    logging.debug(f"q_rotation_shape: {q_rotation.shape}")
    # batch, 2, hidden_size
    q_rotation=torch.mean(q_rotation,dim=1).unsqueeze(1)
    logging.debug(f"q_rotation_shape(after mean): {q_rotation.shape}")
    # batch, 1, hidden_size

    r_out = actor_modules['r_decoder'](q_rotation, actor_encoder_out).squeeze(1)
    logging.debug(f"r_out_shape: {r_out.shape}")
    # batch, 6

    r_log_p, rotation = _rotate_step(r_out.squeeze(-1))
    logging.debug(f"r_log_p_shape: {r_log_p.shape}")
    # batch, 6
    logging.debug(f"rotation_shape: {rotation.shape}")
    # batch, 1

    # rotation
    logging.debug(f"rotate box{state.index}")
    state.update_rotate(rotation)
    blocks=state.action.get_shape()
    # batch,3

    for i,j in enumerate(blocks):
        # TODO: extend to float
        block = j.int().tolist()
        logging.debug(f"block{i}_shape: {block}")
        # [x, y, z]
        block_index = state.index
        container_size = [args.bin_x, args.bin_y, args.bin_z]
        positions = state.positions[i]
        valid_size = state.valid_size[i]
        empty_size =state.empty_size[i]
        heightmap = state.heightmap[i]
        state.positions[i], state.heightmap[i], state.valid_size[i] = calc_one_position_lb_greedy_3d(
            block,
            block_index,
            container_size,
            positions,
            heightmap,
            valid_size
            )

    value = modules['critic'](actor_encoderheightmap_out, actor_encoder_out)
    value = value.squeeze(-1).squeeze(-1)

    state.update_pack()

    return s_log_p, r_log_p, value

def _select_step(s_logits, mask):
    # (batch, box_num) \ (batch, box_num)
    s_logits = s_logits.masked_fill(mask, -np.inf)

    s_log_p = F.log_softmax(s_logits, dim=-1)
    logging.debug(f"s_log_p_shape: {s_log_p.shape}")
    # (batch, box_num)

    # (batch)
    selected = _select(s_log_p.exp()).unsqueeze(-1)
    logging.debug(f"selected_shape: {selected.shape}")
    # (batch, 1)

    # do not reinforce masked and avoid entropy become nan
    s_log_p = s_log_p.masked_fill(mask, 0)

    return s_log_p, selected

def _rotate_step(r_logits):

    r_log_p = F.log_softmax(r_logits, dim=-1)

    # rotate (batch, 1)
    rotate = _select(r_log_p.exp()).unsqueeze(-1)
    
    return r_log_p, rotate

def _select(probs, decode_type="sampling"):
    assert (probs == probs).all(), "Probs should not contain any nans"
    
    if decode_type == "greedy":
        _, selected = probs.max(-1)
    elif decode_type == "sampling":
        selected = probs.multinomial(1).squeeze(1)
    
    else:
        assert False, "Unknown decode type"
        
    return selected



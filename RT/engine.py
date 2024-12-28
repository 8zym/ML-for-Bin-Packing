import torch
import torch.optim as optim
from model import *
import numpy as np
import utils
from Params import args
from DataHandler import DataHandler
from state import StatePack3D
from pack import pack_step
from model import build_model
from utils import move_to, clip_grad_norms, explained_variance
from tqdm import tqdm
import logging

class trainer():
    def __init__(self):
        self.handler = DataHandler()
        self.model = build_model()
        critic_params = [param for name, param in self.model['critic'].named_parameters() if 'log_alpha' not in name]
        self.optimizer = optim.Adam([
                {'params': self.model['actor'].parameters()},
                {'params': critic_params, 'lr': args.critic_lr},
                {'params': self.model['critic'].log_alpha, 'lr': args.critic_lr}
            ], lr=args.actor_lr)
        lambda1 = lambda epoch: min(1, epoch / args.lr_warmup)
        self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda1)
        self.loglikelihood = utils._calc_log_likelihood_3d
        self.entropy = utils._calc_entropy_3d

    def train(self, epoch):

        # 计算所需的批次数量 ceil用于向上取整
        steps = int(np.ceil(args.dataset_size / args.batch))

        for i in tqdm(range(steps)):
            # 1. 该 batch 的起始下标
            st = i * args.batch
            ed = min((i + 1) * args.batch, args.dataset_size)
            bt = ed - st

            # 2. 构造待训练的 batch 数据: batch, box_num, 3
            batch_trn = self.handler.datalist[i] 
            box_num = batch_trn.shape[1]
            if i == 0:
                logging.debug(f"box_num: {box_num}")
                for j in range(box_num):
                    logging.debug(f"box_shape: {batch_trn[0,j]}")
            batch_trn = torch.Tensor(batch_trn).to(args.device)
            state = StatePack3D(batch_size=bt, box_num=box_num)
            state.update_env(batch_trn, bt)
            
            # 3. 训练该 batch
            state, values, returns, losses, entropy, grad_norms = self.train_instance(state)  
            epoch_logger(epoch, state, values, returns, losses, entropy, grad_norms)

    def train_instance(self, state):
        total_losses = torch.tensor([0,0,0,0], dtype=torch.float, device=args.device)
        total_entropy = torch.zeros(1, dtype=torch.float, device=args.device)
        update_mb_number=int(state.box_num//args.nsteps)
        for mb_id in tqdm(range(update_mb_number), disable=True):
            state, entropy, values, returns, losses, grad_norms = self.train_minibatch(state)

            total_entropy += entropy
            total_losses += losses
        
        average_entropy = total_entropy / update_mb_number
        average_losses = total_losses / update_mb_number

        return state, values, returns, average_losses, average_entropy, grad_norms
    
    def train_minibatch(self, state):
        modules = self.model
        returns, advs, values, log_likelihoods, entropys = self.get_mb_data(state)

        alpha_loss = -1 * torch.mv((-1*entropys + args.target_entropy).detach(), modules['critic'].log_alpha).mean()

        value_loss = F.mse_loss(values, returns.float().detach())

        advs = returns - values
        # Normalize the advantages
        advs = (advs - advs.mean()) / (advs.std() + 1e-8)
        # do not backward critic in actor
        advantages = advs.detach()
        # Calculate loss (gradient ascent)  loss=A*log
        actor_loss = -1 * (advantages * log_likelihoods).mean()
        
        loss = actor_loss + value_loss + alpha_loss

        # TODO: Clip gradient norms
        if args.full_eval_mode == 0:
            # Perform backward pass and optimization step
            self.optimizer.zero_grad()
            loss.backward()
            # Clip gradient norms and get (clipped) gradient norms for logging
            grad_norms = clip_grad_norms(self.optimizer.param_groups, args.grad_clip)
            self.optimizer.step()

            if self.scheduler is not None:
                self.scheduler.step()
        else:
            grad_norms = None

        losses = torch.tensor([actor_loss, value_loss, alpha_loss, loss], device=args.device)

        return state, entropys.mean(), values, returns, losses, grad_norms
    
    def get_mb_data(self, state):
    
        def sf01(arr):
            """
            swap and then flatten axes 0 and 1
            """
            s = arr.size()
            return arr.transpose(0, 1).reshape(s[0] * s[1], *s[2:]) 

        mb_rewards, mb_values, mb_log_likelihoods, mb_entropy = [],[],[],[]
        modules = self.model

        for _ in range(args.nsteps):
            
            ll, entropy, value, reward = self._run_batch(state)
            mb_values.append(value)
            mb_log_likelihoods.append(ll)
            mb_rewards.append(reward)
            mb_entropy.append(entropy)

        # batch of steps to batch of roll-outs (nstep, batch)
        mb_rewards = torch.stack(mb_rewards)
        mb_values = torch.stack(mb_values)
        mb_log_likelihoods = torch.stack(mb_log_likelihoods)
        # (nstep, batch, 3)
        mb_entropys = torch.stack(mb_entropy)
        if (state.packed_state[:,:,0]==1).all():
            last_values=torch.zeros(state.batch_size,dtype=torch.float)
            last_values=move_to(last_values,args.device)
            logging.debug("all boxes are packed, last_values is 0")
        else:
            hm = np.zeros((state.batch_size, 2, args.bin_x, args.bin_y))
            for i in range(state.batch_size):
                hm_diff_x = np.insert(state.heightmap[i], 0, state.heightmap[i][0, :], axis=0)
                hm_diff_x = np.delete(hm_diff_x, len(hm_diff_x) - 1, axis=0)
                hm_diff_x = state.heightmap[i] - hm_diff_x

                hm_diff_y = np.insert(state.heightmap[i], 0, state.heightmap[i][:, 0], axis=1)
                hm_diff_y = np.delete(hm_diff_y, len(hm_diff_y.T) - 1, axis=1)
                hm_diff_y = state.heightmap[i] - hm_diff_y
                # combine 
                hm[i][0] = hm_diff_x
                hm[i][1] = hm_diff_y

            hm = torch.tensor(hm).float()
            hm = move_to(hm, args.device)
            actor_modules = modules['actor']

            actor_encoder_out = actor_modules['encoder'](state.packed_state)
            actor_encoderheightmap_out = actor_modules["encoderheightmap"](hm)
            last_values = modules['critic'](actor_encoderheightmap_out, actor_encoder_out)
            last_values = last_values.squeeze(-1).squeeze(-1)

        mb_returns = torch.zeros_like(mb_rewards)
        mb_advs = torch.zeros_like(mb_rewards)

        lastgaelam = 0

        for t in reversed(range(args.nsteps)):
            if t == args.nsteps - 1:
                nextvalues = last_values
            else:
                nextvalues = mb_values[t+1]

            delta = mb_rewards[t] + args.gamma * nextvalues - mb_values[t]
            mb_advs[t] = lastgaelam = delta + args.gamma * args.lam * lastgaelam

        # use return to supervise critic
        mb_returns = mb_advs + mb_values

        # (batch * nstep, )
        returns, advs, values, log_likelihoods, entropys = map(sf01, (mb_returns, mb_advs, mb_values, mb_log_likelihoods, mb_entropys))

        return  returns, advs, values, log_likelihoods, entropys
    

    def _run_batch(self, state):
        modules = self.model
        # update pack candidates for next packing step
        # batch, 1
        last_gap = state.get_gap_size()

        s_log_p, r_log_p, value= pack_step(self.model, state)

        actions = state.action()
        
        ll = self.loglikelihood(actions, s_log_p, r_log_p)
        entropys = self.entropy(s_log_p, r_log_p)

        # (batch)
        new_gap = state.get_gap_size()
        reward = (last_gap - new_gap)

        alpha = torch.exp(modules['critic'].log_alpha)

        reward +=  torch.mv(entropys, alpha).detach()

        state.put_reward(reward)

        return ll, entropys, value, reward
    
def epoch_logger(epoch, state, values, returns, losses, entropy, grad_norms):
    gap_ratio = state.get_gap_ratio()
    rewards = state.get_rewards()
    
    avg_gap_ratio = gap_ratio.mean().item()
    var_gap_ratio = gap_ratio.var().item()
    avg_rewards = rewards.mean().item()
    min_gap = torch.min(gap_ratio)
    max_gap = torch.max(gap_ratio)

    grad_norms, grad_norms_clipped = grad_norms

    ev = explained_variance(values.detach().cpu().numpy(), returns.detach().cpu().numpy())

    # Log values to screen
    if epoch % args.log_interval == 0:
        print("state.packed_state:{}".format(state.packed_state))
        print('\nepoch: {}, avg_rewards: {}, gap_ratio: {}, var_gap_ratio: {}, ev: {}, loss: {}'.\
              format(epoch, avg_rewards, avg_gap_ratio, var_gap_ratio, ev, losses[3]))
        print('min gap ratio: {}, max gap ratio: {}'.format(min_gap, max_gap))
        print('grad_norm: {}, clipped: {}'.format(grad_norms[0], grad_norms_clipped[0]))
        print('grad_norm_c: {}, clipped_c: {}'.format(grad_norms[1], grad_norms_clipped[1]))
        
        print("entropy:{}".format(entropy))
        logging.info(f"epoch: {epoch}")
        logging.info(f"explained_variance: {float(ev)}")
        logging.info(f"entropy: {entropy.item()}")   
        logging.info(f"actor_loss: {losses[0].item()}")
        logging.info(f"value_loss: {losses[1].item()}")
        logging.info(f"alpha_loss: {losses[2].item()}")
        logging.info(f"avg_rewards: {avg_rewards}")
        logging.info(f"gap_ratio: {avg_gap_ratio}")
        logging.info(f"var_gap_ratio: {var_gap_ratio}")



import numpy as np
from env.util import detecting_corner,detecting_corners_batch
import torch
import torch.nn as nn
from torch.distributions import Categorical
from typing import Optional
from einops import reduce


class CategoricalMasking(Categorical):
    def __init__(self, logits: torch.Tensor, mask: Optional[torch.Tensor] = None):
        if mask is not None:
            self.mask = mask
            self.batch, self.nb_actions = logits.size()
            mask_value = torch.tensor(torch.finfo(logits.dtype).min, dtype=logits.dtype, device=logits.device)
            logits = torch.where(mask, logits, mask_value)
        super().__init__(logits=logits)

    def entropy(self):
        if hasattr(self, 'mask'):
            p_log_p = torch.einsum("ij,ij->ij", self.logits, self.probs)
            p_log_p = torch.where(self.mask, p_log_p, torch.tensor(0.0, dtype=p_log_p.dtype, device=p_log_p.device))
            return -reduce(p_log_p, "b a -> b", "sum", b=self.batch, a=self.nb_action)
        else:
            return super().entropy()






class ml_agent(nn.Module):
    def __init__(self):
        super(ml_agent, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=3)
        self.conv2 = nn.Conv2d(in_channels=3, out_channels=5, kernel_size=3)
        self.conv3 = nn.Conv2d(in_channels=5, out_channels=7, kernel_size=3)
        self.fc1 = nn.Linear(7*19*19,25*25)
        self.fc2 = nn.Linear(25*25,25*25)




    def forward(self, observation):
        eroded = observation[0].reshape(-1,25,25)
        new_box = observation[1]
        x = self.conv1(torch.tensor(eroded).unsqueeze(0).float())
        print('x1',x.shape)
        x = self.conv2(x)
        print('x2',x.shape)
        x = self.conv3(x)
        print('x3',x.shape)
        print('x4',x.shape)
        x = self.fc1(x.view(-1, 7*19*19))
        print('x5',x.shape)
        logits = self.fc2(x)


        mask = torch.zeros_like(logits, dtype=torch.bool)
        print('모델 안 eroded',eroded)
        corner_xy = detecting_corners_batch(eroded)
        print('corner_xy',corner_xy)
        for i, corner in enumerate(corner_xy):
            mask[i, 25*corner[0]+corner[1]] = True
        
        # print('mask',mask)
        dist = CategoricalMasking(logits, mask)
        masked_action = dist.probs
        # print('masked_action',masked_action)




        # empty_space_xy = np.array(np.where(eroded[0] == 0)).T
        # action = empty_space_xy[np.argmin(empty_space_xy[:, 0] + empty_space_xy[:, 1])] # Bottom Left Agent
        # action = empty_space_xy[np.random.choice(len(empty_space_xy))] # Random Agent
        
        corner_xy = corner_xy[0] # 임시임 이거 나중에 바꿔야함
        action = corner_xy[np.random.choice(len(corner_xy))] # Placing Candidate Agent
        print('action ::: ',action)
 
        return action , corner_xy
    


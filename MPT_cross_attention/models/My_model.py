
import os,sys
sys.path.append('/home/kimseungjun/task/PointTransformer')
from MPT_cross_attention.models.PT3_model import PointTransformerV3

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence

class PTCrossATT(nn.Module):
    def __init__(self,hidden_dim=512, nhead=4):
        super().__init__()
        """
        scene encoder option
        stride=(2, 2),                   # <--- 2개로 줄이세요
        enc_depths=(2, 2, 6),            # <--- 깊이도 3단계로 맞춤
        enc_channels=(64, 128, 512),     # <--- 채널도 3단계 (마지막은 512 유지)
        enc_num_head=(4, 8, 32),         # <--- 헤드 개수 조정
        enc_patch_size=(128, 128, 128),  # <--- 패치 사이즈 조정
        """
        self.scene_PT = PointTransformerV3(in_channels=6,
                                           stride=(1, 2),enc_depths=(2, 2, 2),
                                           enc_channels=(64, 128, 512), enc_num_head=(4, 8, 32),
                                           enc_patch_size=(128, 128, 128),
                                           )
        self.target_PT = PointTransformerV3(in_channels=6,
                                            stride=(2, 2,2),enc_depths=(2, 2, 2,6),
                                            enc_channels=(64, 128, 256, 512), enc_num_head=(4, 8,16, 32),
                                            enc_patch_size=(128, 128, 128,128),
                                            norm_layer=nn.LayerNorm)

        self.cross_attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=nhead, batch_first=True)
        
        self.pick_proj = nn.Linear(6, hidden_dim)
        self.query_fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), # 1024 -> 512
            nn.ReLU(),                             # 비선형성 추가 (섞어주는 효과)
            # nn.Linear(hidden_dim, hidden_dim)    # (선택사항) 한 번 더 섞어주면 더 좋음
        )

        self.score_head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1) # scalar output (Grasp Quality)
        )
        
        self.collision_head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1) # scalar output (Collision Probability)
        )

    
    def forward(self,input,pick_feature):
        total_pcd = input['scene']
        target_pcd = input['target']
        scene_feat = self.scene_PT(total_pcd)
        target_feat = self.target_PT(target_pcd)
        scene_feature = scene_feat.feat
        #scene_feature = scene_feat.feat.unsqueeze(0)
        target_feature = target_feat.feat #2,512
        num_picks = pick_feature.shape[1]
        pick_emb = self.pick_proj(pick_feature)
        target_emb = target_feature.unsqueeze(1).expand(-1, num_picks, -1)
        
        raw_query = torch.cat([pick_emb, target_emb], dim=-1)
        query = self.query_fusion(raw_query)

        current_batch_size = pick_feature.shape[0]
        scene_list = []
        batch_idx = scene_feat['batch']

        for b in range(current_batch_size):
            # b번째 배치에 해당하는 점들만 마스킹해서 가져옴
            mask = (batch_idx == b)
            scene_list.append(scene_feature[mask])

        scene_feature = pad_sequence(scene_list, batch_first=True)
        
        key_padding_mask = torch.zeros(
            current_batch_size, scene_feature.shape[1], 
            dtype=torch.bool, device=scene_feature.device
        )
        
        for b in range(current_batch_size):
            valid_len = scene_list[b].shape[0]
            key_padding_mask[b, valid_len:] = True # 유효 길이 이후는 True(Masking)


        attn_output, attn_weights = self.cross_attn(
            query=query, 
            key=scene_feature, 
            value=scene_feature,
            key_padding_mask=key_padding_mask
        )

        #logits = self.classifier(attn_output.squeeze(0)) # (N_scene, 2)
        score_logit = self.score_head(attn_output)         # (B, N_pick, 1)
        collision_logit = self.collision_head(attn_output) # (B, N_pick, 1)
        return score_logit,collision_logit
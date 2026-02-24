
import os,sys
sys.path.append('/home/kimseungjun/task/PointTransformer')
from pathlib import Path
current_file_path = Path(__file__).resolve()
path = Path(current_file_path)
work_path = str(path.parent.parent)
sys.path.insert(0, work_path)

from models.PT3_model import PointTransformerV3,PointTransformerV3_decoder,PointTransformerV3_encoder
from model_utils.evidence_utils import DenseNormalGamma
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F

class PTCrossATT(nn.Module):
    def __init__(self,hidden_dim=512, nhead=4,dropout=0.1):
        super().__init__()
        """
        scene encoder option
        stride=(2, 2),                   # <--- 2개로 줄이세요
        enc_depths=(2, 2, 6),            # <--- 깊이도 3단계로 맞춤
        enc_channels=(64, 128, 512),     # <--- 채널도 3단계 (마지막은 512 유지)
        enc_num_head=(4, 8, 32),         # <--- 헤드 개수 조정
        enc_patch_size=(128, 128, 128),  # <--- 패치 사이즈 조정
        """
        ptv3_config = {
            'stride': (2, 2, 2, 2),
            'enc_depths': (2, 2, 2, 2, 2),
            'enc_channels': (32, 64, 128, 256, 256),
            'enc_num_head': (2, 4, 8, 16, 32),
            'enc_patch_size': (128, 128, 128, 128, 128),
            'dec_depths': (2, 2, 2, 2),
            'dec_channels': (64, 64, 128, 256), # User code setting
            'dec_num_head': (4, 4, 8, 16),
            'dec_patch_size': (128, 128, 128, 128)
        }

        # self.scene_PT = PointTransformerV3_encoder(in_channels=3,
        #                                     cls_mode=False,
        #                                     **ptv3_config
        #                                    )
        self.scene_PT = PointTransformerV3(
                                            in_channels=3,
                                            cls_mode=False,
                                            **ptv3_config
                                           )


        self.target_PT = PointTransformerV3_encoder(in_channels=3,
                                            cls_mode=True,
                                            **ptv3_config
                                            )
                                            #norm_layer=nn.LayerNorm)

        # self.PT_decoder = PointTransformerV3_decoder(in_channels=3,
        #                                     cls_mode=False,
        #                                     **ptv3_config
        #                                     )


        self.cross_attn = nn.MultiheadAttention(embed_dim=64, num_heads=nhead, batch_first=True)
        
        self.pick_proj = nn.Linear(6, 256)
        self.pick_proj2 = nn.Linear(6, 64)

        self.query_fusion = nn.Sequential(
            nn.Linear(256 * 2, 256), # 1024 -> 512
            nn.ReLU(),                             # 비선형성 추가 (섞어주는 효과)
            nn.Linear(256, 64),
            nn.ReLU()
        )

        self.pick_fusion = nn.Sequential(
            nn.Linear(64 * 2, 64), # 1024 -> 512
            nn.ReLU(),                             # 비선형성 추가 (섞어주는 효과)
            # nn.Linear(hidden_dim, hidden_dim)    # (선택사항) 한 번 더 섞어주면 더 좋음
        )

        self.evidential_head = nn.Sequential(
            nn.Linear(64, 32), # 공통 특징 추출
            nn.ReLU(),
            DenseNormalGamma(in_features=32, units=1) # (mu, v, alpha, beta) 출력
        )
        self.pos_mlp = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, 64)
        )

        self.scene_context_fusion = nn.Sequential(
            nn.Linear(256+64, 64), # 1024 -> 512
            nn.ReLU(),                             # 비선형성 추가 (섞어주는 효과)
            # nn.Linear(hidden_dim, hidden_dim)    # (선택사항) 한 번 더 섞어주면 더 좋음
        )


    def forward(self,input,pick_feature):
        total_pcd = input['scene']
        target_pcd = input['target']
        scene_out = self.scene_PT(total_pcd) #decoder까지 통과한 feature
        target_feat = self.target_PT(target_pcd) #encoder만 통과한 feature
        flat_scene_feat = scene_out.feat

        target_feature = target_feat.feat #타겟 인코더 정보
        target_batch = target_feat.batch
        batch_idx = input['scene']['batch'] 

        target_per_point = target_feature[batch_idx]

        combined_feat = torch.cat([flat_scene_feat, target_per_point], dim=-1) # Shape: [29290, 320]

        updated_flat_scene = self.scene_context_fusion(combined_feat) # [29290, 64]

        num_picks = pick_feature.shape[1] #1024개 점 sampling
        batch = pick_feature.shape[0]
        pick_emb = self.pick_proj(pick_feature) #1024 point embedding
        B, N, C = pick_emb.shape #

        scene_feats_list = []
        for b in range(B):
            mask = (batch_idx == b)
            scene_feats_list.append(updated_flat_scene[mask])
            #scene_coords_list.append(scene_coord_flat[mask])


        # 배치마다 다른 포인트 수를 최대 길이에 맞춰 패딩 (B, Max_N, C)
        pad_scene_feats = pad_sequence(scene_feats_list, batch_first=True)
        Max_N = pad_scene_feats.shape[1]

        # Key Padding Mask 생성 (실제 점이 없는 패딩 구역은 True)
        lengths = torch.tensor([len(f) for f in scene_feats_list], device=flat_scene_feat.device)
        global_key_mask = torch.arange(Max_N, device=flat_scene_feat.device).unsqueeze(0) >= lengths.unsqueeze(1)


        target_expanded = target_feature.unsqueeze(1).expand(-1, num_picks, -1)

        raw_query = torch.cat([pick_emb, target_expanded], dim=-1)
        raw_query =raw_query.to(pick_feature.device)
        pick_context = self.query_fusion(raw_query)


        attn_output, attn_weights = self.cross_attn(
            query=pick_context,          # [8, 1024, 64] (탐색대)
            key=pad_scene_feats,     # [8, 6190, 64] (현장 정보)
            value=pad_scene_feats,   # [8, 6190, 64] (현장 정보)
            key_padding_mask=global_key_mask # [8, 6190] (가짜 점은 보지 마!)
        )

        #attn_out = attn_output.view(B, N_pick, C)
        
        pick_emb2 = self.pick_proj2(pick_feature)
        
        pick_wise_emb = self.pick_fusion(torch.concat((pick_emb2, attn_output), dim=-1))

        evidential_output = self.evidential_head(pick_wise_emb)
        #collision_logit = self.collision_head(attn_output) # (B, N_pick, 1)
        return evidential_output 




        # current_batch_size = pick_feature.shape[0]

        # scene_coord_flat = input['scene']['coord'] # (Total_Points, 3) 혹은 input['scene'] 자체가 좌표라면 그것 사용
        #        # (Total_Points,)
        # pick_coord = pick_feature
        
        # neighbor_feat_list = []
        # relative_pos_list = []
        # target_k = 1000
        # key_padding_mask_list = []
        # for b in range(current_batch_size):
        #     # b번째 배치에 해당하는 점들만 마스킹해서 가져옴
        #     mask = (batch_idx == b)
        #     scene_feat_b = flat_scene_feat[mask]  # (N_scene_b, C_out) -> 아까 만드신 것
        #     scene_xyz_b = scene_coord_flat[mask]      # (N_scene_b, 3) -> 좌표도 똑같이 잘라야 함!
        #     pick_xyz_b = pick_coord[b][:,:3]
        #     dist = torch.cdist(pick_xyz_b, scene_xyz_b)
        #     num_scene_points = scene_xyz_b.shape[0]

        #     # 실제 추출할 개수는 100개 혹은 전체 점 개수 중 작은 값
        #     actual_k = min(target_k, num_scene_points)
        #     _, indices = torch.topk(dist, k=actual_k, largest=False, dim=-1)
            
        #     # 3. Padding 처리 (부족할 경우)
        #     if actual_k < target_k:
        #         num_pad = target_k - actual_k
        #         # 부족한 만큼 마지막 인덱스를 반복해서 채움
        #         indices = torch.cat([indices, indices[:, -1:].repeat(1, num_pad)], dim=-1)
        #         # 마스크 생성: 실제 점=False, 패딩 점=True (PyTorch 사양)
        #         mask = torch.cat([
        #             torch.zeros((pick_xyz_b.shape[0], actual_k), device=dist.device, dtype=torch.bool),
        #             torch.ones((pick_xyz_b.shape[0], num_pad), device=dist.device, dtype=torch.bool)
        #         ], dim=-1)
        #     else:
        #         mask = torch.zeros((pick_xyz_b.shape[0], target_k), device=dist.device, dtype=torch.bool)

        #     neighbor_feat_list.append(scene_feat_b[indices])
        #     relative_pos_list.append(scene_xyz_b[indices] - pick_xyz_b.unsqueeze(1))
        #     key_padding_mask_list.append(mask)

        # # neighbor_tensor = torch.stack(neighbor_feat_list, dim=0)
        # # relative_tensor = torch.stack(relative_pos_list, dim=0)
        # # (B*N_pick, K) 형태의 마스크
        # key_padding_mask = torch.stack(key_padding_mask_list, dim=0).view(-1, target_k)
        # ###############################################3
        # # 리스트를 텐서로 변환
        # neighbor_tensor = torch.stack(neighbor_feat_list, dim=0) # (B, 512, K, 64)
        # relative_tensor = torch.stack(relative_pos_list, dim=0)  # (B, 512, K, 3)
        # B, N_pick, K, C = neighbor_tensor.shape
        # # B: 배치 크기
        # # N_pick: 1024 (사용자 설정값)
        # # K: 100 (이웃 개수)
        # # C: 64 (채널)

        # query_pos_emb = self.pos_mlp(relative_tensor)
        # query_with_pos = neighbor_tensor + query_pos_emb

        # query_flat = query_with_pos.view(B * N_pick, K, C)

        # key_flat = pad_scene_feats.unsqueeze(1).expand(-1, N_pick, -1, -1).reshape(B * N_pick, -1, C)
        # value_flat = key_flat


        # total_queries = B * N_pick
        # query_flat = pick_context.view(total_queries, 1, C)
        # value_flat = neighbor_tensor.view(total_queries, K, C)
        # rel_pos_flat = relative_tensor.view(total_queries, K, 3)
        # pos_emb = self.pos_mlp(rel_pos_flat)
        # key_flat = value_flat + pos_emb

        # attn_output, attn_weights = self.cross_attn(
        #     query=query_flat, 
        #     key=key_flat, 
        #     value=value_flat,
        #     key_padding_mask=key_padding_mask,
        #     need_weights=True
        # )
        
        # attn_out = attn_output.view(B, N_pick, C)
        
        # pick_emb2 = self.pick_proj2(pick_feature)
        
        # pick_wise_emb = self.pick_fusion(torch.concat((pick_emb2, attn_out), dim=-1))

        # evidential_output = self.evidential_head(pick_wise_emb)
        # #collision_logit = self.collision_head(attn_output) # (B, N_pick, 1)
        # return evidential_output 
    

    def visual_forward(self,input,pick_feature):
        total_pcd = input['scene']
        target_pcd = input['target']
        scene_out = self.scene_PT(total_pcd) #decoder까지 통과한 feature
        target_feat = self.target_PT(target_pcd) #encoder만 통과한 feature
        flat_scene_feat = scene_out.feat

        target_feature = target_feat.feat #129,512
        target_batch = target_feat.batch

        num_picks = pick_feature.shape[1] #512-> 512개의 점
        batch = pick_feature.shape[0]
        pick_emb = self.pick_proj(pick_feature)
        B, N, C = pick_emb.shape

        target_expanded = target_feature.unsqueeze(1).expand(-1, num_picks, -1)

        raw_query = torch.cat([pick_emb, target_expanded], dim=-1)
        raw_query =raw_query.to(pick_feature.device)
        pick_context = self.query_fusion(raw_query)

        current_batch_size = pick_feature.shape[0]

        scene_coord_flat = input['scene']['coord'] # (Total_Points, 3) 혹은 input['scene'] 자체가 좌표라면 그것 사용
        batch_idx = input['scene']['batch']        # (Total_Points,)
        pick_coord = pick_feature
        
        neighbor_feat_list = []
        relative_pos_list = []
        target_k = 1000
        key_padding_mask_list = []
        global_indices_list = []
        all_indices = torch.arange(scene_coord_flat.shape[0], device=scene_coord_flat.device)

        for b in range(current_batch_size):
            # b번째 배치에 해당하는 점들만 마스킹해서 가져옴
            mask = (batch_idx == b)
            scene_feat_b = flat_scene_feat[mask]  # (N_scene_b, C_out) -> 아까 만드신 것
            scene_xyz_b = scene_coord_flat[mask]      # (N_scene_b, 3) -> 좌표도 똑같이 잘라야 함!
            
            global_idx_b = all_indices[mask]
            
            pick_xyz_b = pick_coord[b][:,:3]
            dist = torch.cdist(pick_xyz_b, scene_xyz_b)
            num_scene_points = scene_xyz_b.shape[0]

            # 실제 추출할 개수는 100개 혹은 전체 점 개수 중 작은 값
            actual_k = min(target_k, num_scene_points)
            _, local_indices = torch.topk(dist, k=actual_k, largest=False, dim=-1)
            batch_global_indices = global_idx_b[local_indices]

            # 3. Padding 처리 (부족할 경우)
            if actual_k < target_k:
                num_pad = target_k - actual_k
                # 부족한 만큼 마지막 인덱스를 반복해서 채움
                batch_global_indices = torch.cat([batch_global_indices, batch_global_indices[:, -1:].repeat(1, num_pad)], dim=-1)
                # 마스크 생성: 실제 점=False, 패딩 점=True (PyTorch 사양)
                mask = torch.cat([
                    torch.zeros((pick_xyz_b.shape[0], actual_k), device=dist.device, dtype=torch.bool),
                    torch.ones((pick_xyz_b.shape[0], num_pad), device=dist.device, dtype=torch.bool)
                ], dim=-1)
            else:
                mask = torch.zeros((pick_xyz_b.shape[0], target_k), device=dist.device, dtype=torch.bool)

            neighbor_feat_list.append(scene_feat_b[local_indices])
            global_indices_list.append(batch_global_indices)
            relative_pos_list.append(scene_xyz_b[local_indices] - pick_xyz_b.unsqueeze(1))
            key_padding_mask_list.append(mask)

        neighbor_tensor = torch.stack(neighbor_feat_list, dim=0)
        relative_tensor = torch.stack(relative_pos_list, dim=0)
        # (B*N_pick, K) 형태의 마스크
        key_padding_mask = torch.stack(key_padding_mask_list, dim=0).view(-1, target_k)
        ###############################################3
        # 리스트를 텐서로 변환
        neighbor_tensor = torch.stack(neighbor_feat_list, dim=0) # (B, 512, K, 64)
        global_indices_tensor = torch.stack(global_indices_list, dim=0) # (B, N_pick, K)
        relative_tensor = torch.stack(relative_pos_list, dim=0)  # (B, 512, K, 3)
        B, N_pick, K, C = neighbor_tensor.shape
        # B: 배치 크기
        # N_pick: 1024 (사용자 설정값)
        # K: 100 (이웃 개수)
        # C: 64 (채널)
        total_queries = B * N_pick
        query_flat = pick_context.view(total_queries, 1, C)
        value_flat = neighbor_tensor.view(total_queries, K, C)
        rel_pos_flat = relative_tensor.view(total_queries, K, 3)
        pos_emb = self.pos_mlp(rel_pos_flat)
        key_flat = value_flat + pos_emb

        attn_output, attn_weights = self.cross_attn(
            query=query_flat, 
            key=key_flat, 
            value=value_flat,
            key_padding_mask=key_padding_mask,
            need_weights=True
        )
        
        attn_out = attn_output.view(B, N_pick, C)
        
        pick_emb2 = self.pick_proj2(pick_feature)
        
        pick_wise_emb = self.pick_fusion(torch.concat((pick_emb2, attn_out), dim=-1))

        evidential_output = self.evidential_head(pick_wise_emb)
        attn_weights = attn_weights.view(B, N_pick, K)

        #collision_logit = self.collision_head(attn_output) # (B, N_pick, 1)
        return evidential_output, (attn_weights, global_indices_tensor, neighbor_tensor)
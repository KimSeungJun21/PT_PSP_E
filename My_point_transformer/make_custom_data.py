import numpy as np
import torch
import open3d as o3d
import os
import cv2
from tqdm import tqdm
import glob
import json
import matplotlib.pyplot as plt

class Makedata:
    def __init__(self,path):
        self.data_folder_path = path
        self.img_size = (1024, 1224, 3)
        self.mask_size = (1024, 1224)
    
    def make_pth_data(self,path):
        pcd = o3d.io.read_point_cloud(path)
        coords = np.asarray(pcd.points).astype(np.float32)
        colors = np.asarray(pcd.colors).astype(np.float32)
        normals = np.asarray(pcd.normals).astype(np.float32)
        


        #target mask extract        
        zero_mask_for_segment = np.zeros(self.mask_size)
        zero_mask_for_target = np.zeros(self.mask_size)
        #info path
        info_file_name = os.path.basename(path).replace(".ply", "_result_data.json")
        parent_dir = os.path.dirname(path)
        open_path = os.path.join(parent_dir,info_file_name)

        # 파일이 없으면 메시지 출력 후 함수 종료 (다음 파일로 넘어감)
        if not os.path.exists(open_path):
            print(f"Skipping: {open_path} not found.")
            return

        # 2. 파일이 있을 경우에만 실행
        try:
            with open(open_path, 'r') as f:
                info_data = json.load(f)
        except Exception as e:
            print(f"Error reading {open_path}: {e}")
            return

        # with open(open_path , 'r') as f:
        #     info_data = json.load(f)
        segment_masks_polygone = info_data['result_data']
        for poly in segment_masks_polygone:
            if len(poly) == 2:
                polygon = poly[0]
                np_polygon = np.array([polygon], dtype=np.int32)
                cv2.fillPoly(zero_mask_for_segment, [np_polygon], color=1)
            elif len(poly) > 2:
                polygon = poly
                np_polygon = np.array([polygon], dtype=np.int32)
                cv2.fillPoly(zero_mask_for_segment, np_polygon, color=1)

        pick_obj_id = info_data.get('pick_obj_id', -1) # pick_obj_id가 없을 경우를 대비해 기본값 설정

        if pick_obj_id < 0 or pick_obj_id >= len(segment_masks_polygone):
                    print(f"Skipping: pick_obj_id ({pick_obj_id}) is out of range for {open_path}")
                    return
        target_masks_polygone = info_data['result_data'][info_data['pick_obj_id']]
        if len(target_masks_polygone) == 2:
            polygon = target_masks_polygone[0]
            np_polygon = np.array([polygon], dtype=np.int32)
            cv2.fillPoly(zero_mask_for_target, [np_polygon], color=1)
        elif len(target_masks_polygone) > 2:
            polygon = target_masks_polygone
            np_polygon = np.array([polygon], dtype=np.int32)
            cv2.fillPoly(zero_mask_for_target, np_polygon, color=1)


        flat_target_mask = zero_mask_for_target.reshape(-1)
        flat_semantic_mask = zero_mask_for_segment.reshape(-1)

        assert len(flat_target_mask) == len(coords), \
            f"데이터 불일치! Mask: {len(flat_target_mask)}, Coords: {len(coords)}"

        data_dict = {
                        "coord": coords,
                        "normal": normals,
                        "color": colors,
                        "target_mask": flat_target_mask,
                        "semantic_gt": flat_semantic_mask,
                        'label' :info_data['pick_ok']
                    }



        # 저장 경로 설정 (.ply -> .pth)
        save_file_name = os.path.basename(path).replace(".ply", ".pth")
        
        # 저장할 폴더 설정 (예: 데이터셋 폴더 내 'processed' 폴더)
        save_dir = '/home/kimseungjun/datasets/My_PT_data/PT_data/train'
        os.makedirs(save_dir, exist_ok=True) # 폴더가 없으면 생성
        
        save_path = os.path.join(save_dir, save_file_name)

        # 데이터 저장
        torch.save(data_dict, save_path)
        print(f"Successfully saved: {save_path}")


        # 시각화 부분
        # plt.figure(figsize=(12, 6))

        # # 1. 전체 세그멘테이션 마스크
        # plt.subplot(1, 2, 1)
        # plt.title("Segment Mask (All)")
        # plt.imshow(zero_mask_for_segment, cmap='gray') # 0은 검정, 1은 흰색으로 보임
        # plt.axis('off')

        # # 2. 타겟 오브젝트 마스크
        # plt.subplot(1, 2, 2)
        # plt.title("Target Mask (Pick ID)")
        # plt.imshow(zero_mask_for_target, cmap='jet') # 강조를 위해 jet 컬러맵 사용 가능
        # plt.axis('off')

        # plt.tight_layout()
        # plt.show()




    def forward(self):
        for filepath in tqdm(glob.glob(self.data_folder_path+"/*.ply"), ncols=200):
            self.make_pth_data(filepath)


        # data_dict = {
        #                 "coord": coords,
        #                 "normal": normals,
        #                 "color": colors,
        #                 "target_mask": target_mask,
        #                 "semantic_gt": labels
                    # }

"""
# 4. Dictionary 형태로 묶어서 .pth 저장
            data_dict = {
                "coord": coords,
                "normal": normals,
                "color": colors,
                "target_mask": target_mask,
                "semantic_gt": labels
            }
            
            torch.save(data_dict, os.path.join(save_dir, f"{file_name}.pth"))
"""




def extract_depth_png_normal(ply_path, save_path, img_size=(1024, 1224, 3)): #zivid:(1024, 1224, 3), orbeg:(1080, 1920, 3)
    os.makedirs(save_path, exist_ok=True)
    for filepath in tqdm(glob.glob(ply_path+"/*.ply"), ncols=200):
        try:
            file_name = os.path.basename(filepath).rstrip(".ply")
            pcd = o3d.io.read_point_cloud(filepath)
            # img_rgb = np.reshape(np.asarray(pcd.colors)*255, (480, 848, 3)).astype(np.uint8)
            img_rgb = np.reshape(np.asarray(pcd.colors)*255, img_size).astype(np.uint8)
            img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB)
            cv2.imwrite(f"{save_path}/{file_name}_rgb.png", img_rgb)
            img_depth =  np.reshape(np.asarray(pcd.points)[:, 2], (img_size[0], img_size[1]))
            img_depth = np.uint16(img_depth)
            cv2.imwrite(f"{save_path}/{file_name}_depth.png", img_depth)
            img_normal =  np.reshape(np.asarray(pcd.normals), img_size).astype(np.float32)
            height_bin = np.array([img_normal.shape[0]])
            width_bin = np.array([img_normal.shape[1]])
            channels_bin = np.array([img_normal.shape[2]])
            data_bin = np.array(img_normal.flatten())
            with open(f"{save_path}/{file_name}_normal.bin", 'wb') as fp:
                height_bin.tofile(fp, "", format="int32")
                width_bin.tofile(fp, "", format="int32")
                channels_bin.tofile(fp, "", format="int32")
                data_bin.tofile(fp, "", format="float32")
        except Exception as e:
            print(e)
            continue








if __name__ == "__main__":    
    # PLY에서 이미지 추출
    ROOT_PATH = "/home/kimseungjun/datasets/CMES_DATA/12_23/251029"
    save_path = os.path.join(ROOT_PATH, "raw")
    datas = Makedata(ROOT_PATH)
    datas.forward()
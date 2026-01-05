import numpy as np
import torch
import open3d as o3d
import os
import cv2
from tqdm import tqdm
import glob
import json
import matplotlib.pyplot as plt

def safe_torch_load(fn, map_location="cpu"):
    """
    Load a torch serialized file with a clearer error message for corrupted files.

    The original crash was ``ValueError: buffer size does not match array size``
    coming from ``torch.load`` when a ``.pth`` file was partially written.  The
    wrapped loader attempts a CPU load first and, on failure, re-raises with the
    file path and size so the caller knows which sample needs to be regenerated.
    """

    try:
        return torch.load(fn, map_location=map_location)
    except ValueError as exc:
        try:
            file_size = os.stat(fn).st_size
        except OSError:
            file_size = None

        size_info = f" ({file_size} bytes)" if file_size is not None else ""
        raise ValueError(
            f"Failed to load corrupted sample '{fn}'{size_info}: {exc}. "
            "Please regenerate or re-download the dataset file."
        ) from exc
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





    def forward(self):
        label_json = {}
        labels_only = []
        for filepath in tqdm(glob.glob(self.data_folder_path+"/*.pth"), ncols=200):
            #label = self.make_pth_data(filepath)
            datas = safe_torch_load(filepath, map_location="cpu")
            label = datas['label']
            fn = os.path.basename(filepath)
            label_int = 1 if label is True else 0
            label_json[fn] = label_int
            labels_only.append(label_int)
        
        # --- 통계 계산 ---
        labels_np = np.array(labels_only)
        neg_count = int(np.sum(labels_np == 0))
        pos_count = int(np.sum(labels_np == 1))
        # 0번 개수 / 1번 개수로 가중치 계산 (1번이 0개일 경우 대비)
        suggested_pos_weight = neg_count / max(pos_count, 1)

        # --- 최종 저장 데이터 구성 ---
        output_data = {
            "stats": {
                "total": len(labels_only),
                "neg_count": neg_count,
                "pos_count": pos_count,
                "suggested_pos_weight": round(suggested_pos_weight, 4)
            },
            "labels": label_json
        }



        save_path = os.path.join(self.data_folder_path, "label_stats.json")
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=4)



if __name__ == "__main__":    
    # PLY에서 이미지 추출
    ROOT_PATH = "/home/kimseungjun/datasets/My_PT_data/PT_data/train"
    save_path = os.path.join(ROOT_PATH, "raw")
    datas = Makedata(ROOT_PATH)
    datas.forward()
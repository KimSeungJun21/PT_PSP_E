
import os,sys
import suctionnetAPI
import open3d as o3d
import numpy as np
from suctionnetAPI import SuctionNet
import torch
sys.path.append('/home/kimseungjun/task/PointTransformer/suctionnetAPI')
from suctionnetAPI.utils.utils import (
    generate_scene_model, 
    plot_sucker_collision, 
    transform_points, 
    parse_posevector, 
    create_table_cloud, 
    get_model_suctions
)

from suctionnetAPI.utils.rotation import (
    viewpoint_to_matrix
)
from tqdm import tqdm  # 상단에 추가


def get_visible_pcd(pcd, camera_location=[0, 0, 0]):
    """
    pcd: open3d.geometry.PointCloud 객체
    camera_location: 카메라의 위치 (보통 [0, 0, 0])
    """
    # 1. 가시성 체크를 위한 파라미터 설정
    # 점구름의 전체 크기(대각선 길이)를 계산하여 적절한 radius 설정
    diameter = np.linalg.norm(np.asarray(pcd.get_max_bound()) - np.asarray(pcd.get_min_bound()))
    
    # radius가 너무 작으면 구멍이 뚫리고, 너무 크면 뒷면이 안 가려집니다. 
    # 보통 diameter의 100배 정도가 적당합니다.
    radius = diameter * 100 
    
    # 2. Hidden Point Removal 수행 (현재 점들 중 카메라에서 보이는 점의 인덱스 반환)
    _, pt_map = pcd.hidden_point_removal(camera_location, radius)
    
    # 3. 보이는 점들만 선택하여 반환
    visible_pcd = pcd.select_by_index(pt_map)
    return visible_pcd


def main():
    visu_num_each = 10
    suction_file_path = '/home/kimseungjun/datasets/graspnet_data/suctionnet'
    scene_path = os.path.join(suction_file_path,'scenes')
    scene_id_list = os.listdir(scene_path)
    dataset = []

    save_path = '/home/kimseungjun/datasets/My_PT_data/PT_data/sn_train'
    scene_pbar = tqdm(scene_id_list, desc="Overall Scenes", unit="scene")
    for i in scene_pbar:
        scene_id = int(i.split('_')[-1])
        # scene_id=0
        # i='scene_%04d' % scene_id
        camera='kinect'
        sn = SuctionNet(root=suction_file_path, camera=camera)

        scene_kinect_path = os.path.join(scene_path,i,camera)
        annotation_list = os.listdir(os.path.join(scene_kinect_path,'annotations'))
        #for ann_l in annotation_list:
        for ann_l in tqdm(annotation_list, desc=f"Scene {scene_id} Annotations", leave=False):
            ann = int(ann_l.replace('.xml',''))
            #ann=0
            model_list, obj_list, pose_list = generate_scene_model(suction_file_path, i, ann, return_poses=True, camera=camera, align=True)

            camera_poses = np.load(os.path.join(suction_file_path, 'scenes', i, camera, 'camera_poses.npy'.format(camera)))
            camera_pose = camera_poses[ann]
            T_world_to_cam = np.linalg.inv(camera_pose)

            align_mat = np.load(os.path.join(suction_file_path, 'scenes', i, camera, 'cam0_wrt_table.npy'))
            scene_pcd = sn.loadScenePointCloud(sceneId=scene_id, camera=camera, annId=ann, format='open3d')

            real_camera_pose = np.matmul(align_mat, camera_poses[ann])
            cam_pos_in_aligned_space = np.linalg.inv(real_camera_pose)[:3, 3]

            scene_pcd.transform(real_camera_pose) # 배경을 물체와 같은 '정렬된 카메라 좌표계'로 변환
            scene_pcd = scene_pcd.voxel_down_sample(voxel_size=0.005)


            plane_model, inliers = scene_pcd.segment_plane(distance_threshold=0.015,
                                             ransac_n=3,
                                             num_iterations=1000)
            table = scene_pcd.select_by_index(inliers)
            table.paint_uniform_color([0.2, 0.2, 0.2])

            [a, b, c, d] = plane_model # 평면 방정식: ax + by + cz + d = 0

            # 물체 주변 영역에만 그리드 생성 (예: -0.5 ~ 0.5 범위)
            grid_size = 0.01
            x = np.arange(-0.5, 0.5, grid_size)
            y = np.arange(-0.5, 0.5, grid_size)
            gx, gy = np.meshgrid(x, y)

            # 평면 방정식에 맞춰 z값 계산 (z = -(ax + by + d) / c)
            gz = -(a * gx + b * gy + d) / c -0.01

            # 가상 포인트들을 Open3D 객체로 변환
            points = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1)
            table = o3d.geometry.PointCloud()
            table.points = o3d.utility.Vector3dVector(points)
            table.paint_uniform_color([0.1, 0.1, 0.1])




            collision_dir = os.path.join(suction_file_path, 'suction_collision_label')
            
            collision_dump = np.load(os.path.join(collision_dir, '{:04d}_collision.npz'.format(scene_id)))

            radius = 0.01
            height = 0.1

            num_obj = len(obj_list)
            visible_model_list = []
            full_scene_pcd = o3d.geometry.PointCloud()
            full_scene_pcd += table

            for m in model_list:
                v_pcd = get_visible_pcd(m, camera_location=cam_pos_in_aligned_space)
                # 시각적 구분을 위해 랜덤 색상을 입힐 수도 있습니다.
                # v_pcd.paint_uniform_color(np.random.rand(3)) 
                visible_model_list.append(v_pcd)
                full_scene_pcd += v_pcd


            full_points = np.asarray(full_scene_pcd.points).copy()
            full_normals = np.asarray(full_scene_pcd.normals).copy()
            full_colors = np.asarray(full_scene_pcd.colors).copy()
            
            scene_save_data = {
                'scene_points': torch.tensor(full_points, dtype=torch.float32),
                'scene_normals': torch.tensor(full_normals, dtype=torch.float32),
                'scene_colors': torch.tensor(full_colors, dtype=torch.float32)}

            
            scene_file_id =f'scene_{scene_id:04d}' 
            save_path_id = f'{scene_file_id}.pth'
            
            scene_file_path = os.path.join(save_path,scene_file_id)

            if not os.path.exists(scene_file_path):
                os.mkdir(scene_file_path)

            scene_p = os.path.join(scene_file_path,save_path_id)
            torch.save(scene_save_data, scene_p)
            #print(f"Successfully saved {scene_save_data}")

            
            #for obj_i in range(len(obj_list)):
            for obj_i in tqdm(range(len(obj_list)), desc="Objects in Scene", leave=False):
                target_pcd = o3d.geometry.PointCloud()
                
                suckers = []
                #print('Checking ' + str(obj_i+1) + ' / ' + str(num_obj))
                obj_idx = obj_list[obj_i]
                trans = pose_list[obj_i]

                target_model = visible_model_list[obj_i]
                target_points = np.asarray(target_model.points).copy()
                target_normals = np.asarray(target_model.normals).copy()
                #target_colors = np.asarray(target_model.colors)
                target_colors = np.asarray(target_model.colors).copy() if target_model.has_colors() else np.zeros_like(target_points)
                

                # --- [수정 포인트] 포인트별 데이터를 모을 리스트 준비 ---
                all_suction_targets = []
                all_collisions = []
                all_scores = []


                object_file_id =f'target_objects_{obj_idx}' 
                object_save_path_id = f'{object_file_id}.pth'
                
                object_save_data = {
                    'points': torch.tensor(target_points, dtype=torch.float32),
                    'normals': torch.tensor(target_normals, dtype=torch.float32),
                    'colors': torch.tensor(target_colors, dtype=torch.float32)}

                object_file_path = os.path.join(scene_file_path,object_save_path_id)

                if not os.path.exists(object_file_path):
                    os.mkdir(object_file_path)


                obj_p = os.path.join(object_file_path,object_save_path_id)
                torch.save(object_save_data, obj_p)
                #print(f"Successfully saved {object_save_data}")


                #target_visible_pcd = get_visible_pcd(target_model, camera_location=cam_pos_in_aligned_space)
                seal_dir = os.path.join(suction_file_path, 'seal_label')
                sampled_points, normals, scores, _ = get_model_suctions('%s/%03d_seal.npz'%(seal_dir, obj_idx))
                collisions = collision_dump['arr_{}'.format(obj_i)]

                sucker_params = []
                point_pbar = tqdm(range(len(sampled_points)), desc=f"Obj {obj_idx} Points", leave=False)
                #for point_ind in range(len(sampled_points)):
                for point_ind in point_pbar:
                    target_point = sampled_points[point_ind]
                    normal = normals[point_ind]
                    #score = scores[point_ind]
                    #collision = collisions[point_ind]
                    R = viewpoint_to_matrix(normal)
                    t = transform_points(target_point[np.newaxis,:], trans).squeeze()
                    R = np.dot(trans[:3,:3], R)
                    #sucker = plot_sucker_collision(R, t, collision, radius, height)
                    #suckers.append(sucker)
                    sucker_params.append([target_point[0],target_point[1],target_point[2],normal[0],normal[1],normal[2],radius, height])
                    sucker_pr = [target_point[0],target_point[1],target_point[2],normal[0],normal[1],normal[2],radius, height]
                    # o3d.visualization.draw_geometries([table, *visible_model_list], width=1536, height=864) #total scene pcd
                    # o3d.visualization.draw_geometries([target_model, *suckers], #target pcd
                    #                   window_name=f"Object {obj_idx} Only",
                    #                   width=1536, height=864)
                    # o3d.visualization.draw_geometries([table, *visible_model_list, *suckers], width=1536, height=864)
                    
                    # 리스트에 수집
                    all_suction_targets.append(sucker_pr)
                    all_collisions.append(collisions[point_ind])
                    all_scores.append(scores[point_ind])

                save_data = {
                    # 해당 물체의 정답 정보 (석션 가능 지점, 점수 등)
                    'suction_targets': torch.tensor(all_suction_targets, dtype=torch.float32),
                    'collisions': torch.tensor(all_collisions, dtype=torch.int32),
                    'scores': torch.tensor(all_scores, dtype=torch.float32),                    
                    # 메타데이터
                    'scene_id': scene_id,
                    'ann_id': ann,
                    'obj_idx': obj_idx,
                    'pose': torch.tensor(trans, dtype=torch.float32),
                }
                save_path_id = f'scene_id_{scene_id:04d}_ann_{ann:04d}_obj_{obj_idx:03d}.pth'
                
                suction_d_p = os.path.join(object_file_path,'suction_data')
                sp = os.path.join(suction_d_p,save_path_id)
                
                if not os.path.exists(suction_d_p):
                    os.mkdir(suction_d_p)
                
                
                torch.save(save_data, sp)
                    #print(f"Successfully saved {save_path}")
            
            #o3d.visualization.draw_geometries([scene_pcd])
            #print(1)



if __name__ == "__main__":
    main()

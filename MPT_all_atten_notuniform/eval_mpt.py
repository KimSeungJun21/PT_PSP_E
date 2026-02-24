from suctionnetAPI import SuctionNetEval
import os
import numpy as np
import argparse
import random

def setup_seed(seed):
    np.random.seed(seed)
    random.seed(seed)

# Set random seed
setup_seed(0)


def evaluate(cfgs):
    dataset_root = cfgs.dataset_root
    camera = cfgs.camera
    dump_dir = cfgs.dump_dir
    split = cfgs.split
    suctionnet_eval = SuctionNetEval(root=dataset_root, camera=camera)

    result_path = os.path.join('experiment', dump_dir)
    # Evaluate all the test splits
    # res is the raw evaluation results, ap_top50 and ap_top1 are average precision of top 50 and top 1 suctions
    # see our paper for details
    if split == 'test_seen':
        res, ap_top50, ap_top1 = suctionnet_eval.eval_seen(dump_folder=result_path, proc=cfgs.num_workers)
    elif split == 'test_similar':
        res, ap_top50, ap_top1 = suctionnet_eval.eval_similar(dump_folder=result_path, proc=cfgs.num_workers)
    elif split == 'test_novel':
        res, ap_top50, ap_top1 = suctionnet_eval.eval_novel(dump_folder=result_path, proc=cfgs.num_workers)
    else:
        print(f"Invalid split: {split}")
        return
    
    save_dir = os.path.join('experiment', cfgs.dump_dir, 'ap_{}_{}.npy'.format(cfgs.split, cfgs.camera))
    np.save(save_dir, res)
    print(f"Results saved to: {save_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', default='test_novel', help='dataset split [default: test_seen]')
    parser.add_argument('--camera', default='realsense', help='camera to use [default: realsense]')
    parser.add_argument('--dump_dir', default='mpt_evidence', help='where to save')
    parser.add_argument('--dataset_root', default='/home/kimseungjun/datasets/graspnet_data/suctionnet', help='where dataset is')
    parser.add_argument('--num_workers', type=int, default=10, help='Number of workers used in evaluation [default: 10]')
    cfgs = parser.parse_args()

    print(cfgs)
    evaluate(cfgs)


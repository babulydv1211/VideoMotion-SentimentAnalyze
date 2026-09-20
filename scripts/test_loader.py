import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scene_motion_llm.utils.dataset import VideoFrameDataset as MOSEIDataset
from torch.utils.data import DataLoader
import traceback

if __name__ == '__main__':
    try:
        ds = MOSEIDataset(dataset_path='./dataset/CMU-MOSEI', frames_path='./frames', labels_path=None, split='train')
        dl = DataLoader(ds, batch_size=2, num_workers=2)
        batch = next(iter(dl))
        print('Batch keys:', list(batch.keys()))
        print('Frames shape:', batch['frames'].shape)
        print('Labels shape:', batch['label'].shape)
    except Exception:
        traceback.print_exc()

import math
from pathlib import Path
import bisect
from typing import Dict, Tuple
import weakref

import cv2
from numba import jit
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from utils import visualization as visu
from matplotlib import pyplot as plt
from utils import transformers
import os

from utils.dsec_utils import RepresentationType, VoxelGrid, flow_16bit_to_float

VISU_INDEX = 1

class EventSlicer:
    def __init__(self, directory):
        self.events = dict()
        self.folder_directory = directory
        self.events['t'] = np.load(str(self.folder_directory) + '/dataset_events_t.npy', mmap_mode='r')
        self.events['p'] = np.load(str(self.folder_directory) + '/dataset_events_p.npy', mmap_mode='r')
        self.events['x'] = np.load(str(self.folder_directory) + '/dataset_events_xy.npy', mmap_mode='r')[:, 0]
        self.events['y'] = np.load(str(self.folder_directory) + '/dataset_events_xy.npy', mmap_mode='r')[:, 1]

    def get_final_time_us(self):
        return self.events['t'][-1]

    def get_events(self, t_start_us: int, t_end_us: int) -> Dict[str, np.ndarray]:
        """Get events (p, x, y, t) within the specified time window
        Parameters
        ----------
        t_start_us: start time
        t_end_us: end time
        Returns
        -------
        events: dictionary of (p, x, y, t) or None if the time window cannot be retrieved
        """
        index_start = bisect.bisect_left(self.events['t'], t_start_us)
        index_end = bisect.bisect_left(self.events['t'], t_end_us)
        events = dict()
        events['t'] = np.asarray(self.events['t'][index_start:index_end])
        for dset_str in ['p', 'x', 'y']:
            events[dset_str] = np.asarray(self.events[dset_str][index_start:index_end])
            assert events[dset_str].size == events['t'].size
        return events
    
class Sequence(Dataset):
    def __init__(self, seq_path: Path, representation_type: RepresentationType, mode: str='test', delta_t_us: int=0.000001,
                 num_bins: int=15, transforms=None, name_idx=0, visualize=False):
        assert num_bins >= 1
        assert seq_path.is_dir()
        assert mode in {'train', 'test'}
        '''
        Directory Structure:

        Dataset
        └── test
            ├── scene_03_00_000000
            │   ├── dataset_events_p.npy
            │   ├── dataset_events_t.npy
            │   ├── dataset_events_xy.npy
            │   ├── dataset_flow.npz
            │   └── dataset_mask.npz

        '''

        self.mode = mode
        self.name_idx = name_idx
        self.visualize_samples = visualize
        self.delta_t_us = delta_t_us * 10
        # # Get Test Timestamp File
        # test_timestamp_file = seq_path / 'test_forward_flow_timestamps.csv'
        # assert test_timestamp_file.is_file()
        # file = np.genfromtxt(
        #     test_timestamp_file,
        #     delimiter=','
        # )
        # self.idx_to_visualize = file[:,2]

        # Save output dimensions
        self.height = 480
        self.width = 640
        self.num_bins = num_bins

        # Just for now, we always train with num_bins=15
        assert self.num_bins==15

        # Set event representation
        self.voxel_grid = None
        if representation_type == RepresentationType.VOXEL:
            self.voxel_grid = VoxelGrid((self.num_bins, self.height, self.width), normalize=True)

        #Load and compute timestamps and indices
        # timestamps_images = np.loadtxt(seq_path / 'image_timestamps.txt', dtype='int64')
        # image_indices = np.arange(len(timestamps_images))
        # But only use every second one because we train at 10 Hz, and we leave away the 1st & last one
        # self.timestamps_flow = timestamps_images[::2][1:-1]
        # self.indices = image_indices[::2][1:-1]

        # Left events only
        ev_dir_location = seq_path
        self.event_slicer = EventSlicer(ev_dir_location)
        flow = np.load(ev_dir_location / 'dataset_flow.npz', mmap_mode='r')
        self.timestamps_flow = flow['t'][2:-2]
        self.indices = np.arange(len(self.timestamps_flow))
        self.idx_to_visualize = self.indices[::5][50:-50]
        meta = np.load(ev_dir_location / 'dataset_info.npz', allow_pickle=True, mmap_mode='r')['meta'].item()
        self.meta = meta['meta']

    def events_to_voxel_grid(self, p, t, x, y, device: str='cpu'): 
        t = (t - t[0]).astype('float32')
        t = (t/t[-1])
        x = x.astype('float32')
        y = y.astype('float32')
        pol = p.astype('float32')
        event_data_torch = {
            'p': torch.from_numpy(pol),
            't': torch.from_numpy(t),
            'x': torch.from_numpy(x),
            'y': torch.from_numpy(y),
        }
        return self.voxel_grid.convert(event_data_torch)

    def getHeightAndWidth(self):
        return self.height, self.width

    # def load_flow(self, flowfile: Path, maskfile: Path):
    #     assert flowfile.exists()
    #     assert flowfile.suffix == '.npz'
    #     return flow, valid2D

    def get_image_width_height(self):
        return self.height, self.width

    def __len__(self):
        return len(self.timestamps_flow)

    def rectify_events(self, x: np.ndarray, y: np.ndarray):
        # assert location in self.locations
        # From distorted to undistorted
        assert x.max() < self.width
        assert y.max() < self.height
        K = np.array([[self.meta['fx'], 0, self.meta['cx']],
                      [0, self.meta['fy'], self.meta['cy']],
                      [0, 0, 1]])
        Coeffs = np.array([self.meta['k1'], self.meta['k2'], self.meta['p1'], self.meta['p2']])
        x = x.reshape(-1, 1)
        y = y.reshape(-1, 1)
        events = np.concatenate((x, y), axis=1).reshape(1, -1, 2)
        rectify_events = np.squeeze(cv2.undistortPoints(events.astype(np.float32), K, Coeffs))
        rectify_events = np.concatenate((rectify_events.T, np.ones((1, len(rectify_events)))), axis=0)
        rectify_events = (K @ rectify_events)[0:2, :].T
        return rectify_events

    def get_data_sample(self, index, crop_window=None, flip=None):
        # First entry corresponds to all events BEFORE the flow map
        # Second entry corresponds to all events AFTER the flow map (corresponding to the actual fwd flow)
        names = ['event_volume_old', 'event_volume_new']
        ts_start = [self.timestamps_flow[index] - self.delta_t_us, self.timestamps_flow[index]]
        ts_end = [self.timestamps_flow[index], self.timestamps_flow[index] + self.delta_t_us]

        file_index = self.indices[index]

        output = {
            'file_index': file_index,
            'timestamp': self.timestamps_flow[index]
        }
        # Save sample for benchmark submission
        # output['save_submission'] = file_index in self.idx_to_visualize
        output['save_submission'] = file_index in self.idx_to_visualize
        output['visualize'] = self.visualize_samples


        for i in range(len(names)):
            event_data = self.event_slicer.get_events(ts_start[i], ts_end[i])

            p = event_data['p']
            t = event_data['t']
            x = event_data['x']
            y = event_data['y']

            xy_rect = self.rectify_events(x, y)
            x_rect = xy_rect[:, 0]
            y_rect = xy_rect[:, 1]

            if crop_window is not None:
                # Cropping (+- 2 for safety reasons)
                x_mask = (x_rect >= crop_window['start_x']-2) & (x_rect < crop_window['start_x']+crop_window['crop_width']+2)
                y_mask = (y_rect >= crop_window['start_y']-2) & (y_rect < crop_window['start_y']+crop_window['crop_height']+2)
                mask_combined = x_mask & y_mask
                p = p[mask_combined]
                t = t[mask_combined]
                x_rect = x_rect[mask_combined]
                y_rect = y_rect[mask_combined]

            if self.voxel_grid is None:
                raise NotImplementedError
            else:
                event_representation = self.events_to_voxel_grid(p, t, x_rect, y_rect)
                output[names[i]] = event_representation
            output['name_map']=self.name_idx
        return output

    def __getitem__(self, idx):
        sample =  self.get_data_sample(idx)
        return sample
    
class SequenceRecurrent(Sequence):
    def __init__(self, seq_path: Path, representation_type: RepresentationType, mode: str='test', delta_t_us: int=100,
                 num_bins: int=15, transforms=None, sequence_length=1, name_idx=0, visualize=False):
        super(SequenceRecurrent, self).__init__(seq_path, representation_type, mode, delta_t_us, transforms=transforms,
                                                name_idx=name_idx, visualize=visualize)
        self.sequence_length = sequence_length
        self.valid_indices = self.get_continuous_sequences()

    def get_continuous_sequences(self):
        continuous_seq_idcs = []
        if self.sequence_length > 1:
            for i in range(len(self.timestamps_flow)-self.sequence_length+1):
                diff = self.timestamps_flow[i+self.sequence_length-1] - self.timestamps_flow[i]
                if diff < np.max([100000 * (self.sequence_length-1) + 1000, 101000]):
                    continuous_seq_idcs.append(i)
        else:
            for i in range(len(self.timestamps_flow)-1):
                diff = self.timestamps_flow[i+1] - self.timestamps_flow[i]
                if diff < np.max([100000 * (self.sequence_length-1) + 1000, 101000]):
                    continuous_seq_idcs.append(i)
        return continuous_seq_idcs

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        assert idx >= 0
        assert idx < len(self)

        # Valid index is the actual index we want to load, which guarantees a continuous sequence length
        valid_idx = self.valid_indices[idx]

        sequence = []
        j = valid_idx

        ts_cur = self.timestamps_flow[j]
        # Add first sample
        sample = self.get_data_sample(j)
        sequence.append(sample)

        # Data augmentation according to first sample
        crop_window = None
        flip = None
        if 'crop_window' in sample.keys():
            crop_window = sample['crop_window']
        if 'flipped' in sample.keys():
            flip = sample['flipped']

        for i in range(self.sequence_length-1):
            j += 1
            ts_old = ts_cur
            ts_cur = self.timestamps_flow[j]
            assert(ts_cur-ts_old < 100000 + 1000)
            sample = self.get_data_sample(j, crop_window=crop_window, flip=flip)
            sequence.append(sample)

        # Check if the current sample is the first sample of a continuous sequence
        if idx==0 or self.valid_indices[idx]-self.valid_indices[idx-1] != 1:
            sequence[0]['new_sequence'] = 1
            print("Timestamp {} is the first one of the next seq!".format(self.timestamps_flow[self.valid_indices[idx]]))
        else:
            sequence[0]['new_sequence'] = 0
        return sequence
    
class DatasetProvider:
    def __init__(self, dataset_path: Path, representation_type: RepresentationType, delta_t_us: int=0.000001, num_bins=15,
                 type='standard', config=None, visualize=False):
        path = dataset_path / 'evimo'
        assert dataset_path.is_dir(), str(dataset_path)
        assert path.is_dir(), str(path)
        self.config=config
        self.name_mapper = []

        test_sequences = list()
        for child in path.iterdir():
            self.name_mapper.append(str(child).split("/")[-1])
            if type == 'standard':
                test_sequences.append(Sequence(child, representation_type, 'test', delta_t_us, num_bins,
                                               transforms=[],
                                               name_idx=len(self.name_mapper)-1,
                                               visualize=visualize))
            elif type == 'warm_start':
                test_sequences.append(SequenceRecurrent(child, representation_type, 'test', delta_t_us, num_bins,
                                                        transforms=[], sequence_length=1,
                                                        name_idx=len(self.name_mapper)-1,
                                                        visualize=visualize))
            else:
                raise Exception('Please provide a valid subtype [standard/warm_start] in config file!')

        self.test_dataset = torch.utils.data.ConcatDataset(test_sequences)

    def get_test_dataset(self):
        return self.test_dataset


    def get_name_mapping_test(self):
        return self.name_mapper

    def summary(self, logger):
        logger.write_line("================================== Dataloader Summary ====================================", True)
        logger.write_line("Loader Type:\t\t" + self.__class__.__name__, True)
        logger.write_line("Number of Voxel Bins: {}".format(self.test_dataset.datasets[0].num_bins), True)
        
if __name__ == '__main__':
    # slice = EventSlicer(directory='/fs/nexus-scratch/zhenglin/E-RAFT/data/evimo/scene_03_00_000000')
    # Seq = Sequence(seq_path=Path('/fs/nexus-scratch/zhenglin/E-RAFT/data/evimo'), representation_type=RepresentationType.VOXEL)
    # print("Success!")
    loader = DatasetProvider(
            dataset_path=Path('/fs/nexus-scratch/zhenglin/E-RAFT/data/'),
            representation_type=RepresentationType.VOXEL,
            delta_t=0.001,
            type='warm_start')
    test_set = loader.get_test_dataset()
    additional_loader_returns = {'name_mapping_test': loader.get_name_mapping_test()}
    loader = DataLoader(test_set,
                                 batch_size=1,
                                 shuffle=False,
                                 num_workers=4,
                                 drop_last=True)
    for i, data in enumerate(loader):
        # print(len(data))
        data = data[0]
        print(f'Batch {i} - Inputs type: {type(data)}-Dict keys: {data.keys()}')
        print(f'file_index: {data["file_index"]}')
        print(f'save_submission: {data["save_submission"]}')
        print(f'timestamp: {data["timestamp"]}')
        print(f'volume_old: {data["event_volume_old"].shape}')
        print(f'volume_new: {data["event_volume_new"].shape}')
        print(f'name_map: {data["name_map"]}')
        
        # Optional: break the loop after a few batches to avoid printing too much data
        if i == 2:
            break
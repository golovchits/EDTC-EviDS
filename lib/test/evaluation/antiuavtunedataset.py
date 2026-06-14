import numpy as np
import os
import json
from lib.test.evaluation.data import Sequence, BaseDataset, SequenceList


class AntiUAVTuneDataset(BaseDataset):
    """30-sequence tune split from AntiUAV600 training data (antiuav_train_val_split.json)."""

    def __init__(self):
        super().__init__()
        self.base_path = self.env_settings.antiuav_train_path
        self.sequence_list = self._get_sequence_list()

    def get_sequence_list(self):
        return SequenceList([self._construct_sequence(s) for s in self.sequence_list])

    def _construct_sequence(self, sequence_name):
        anno_path = '{}/{}/IR_label.json'.format(self.base_path, sequence_name)
        with open(anno_path, 'r') as f:
            label_res = json.load(f)
        gt = label_res['gt_rect']
        for i in range(len(gt)):
            if gt[i] in ([], [0]):
                gt[i] = [0, 0, 0, 0]

        ground_truth_rect = np.array(gt)
        target_visible = np.array(label_res['exist'], dtype=bool)

        frames_path = '{}/{}'.format(self.base_path, sequence_name)
        frame_list = sorted([f for f in os.listdir(frames_path) if f.endswith('.jpg')],
                            key=lambda f: int(f[:-4]))
        frames_list = [os.path.join(frames_path, f) for f in frame_list]

        return Sequence(sequence_name, frames_list, 'antiuav',
                        ground_truth_rect.reshape(-1, 4),
                        target_visible=target_visible)

    def __len__(self):
        return len(self.sequence_list)

    def _get_sequence_list(self):
        import json as _json
        split_json = os.path.join(
            os.path.dirname(__file__), '..', '..', '..', 'antiuav_train_val_split.json'
        )
        with open(split_json) as f:
            split = _json.load(f)
        return split['tune']

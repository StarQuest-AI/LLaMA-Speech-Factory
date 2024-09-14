from typing import Iterable, Optional, List, Union, Iterable, Dict, Any
import numpy as np
import torch
import torch.nn.functional as F
from lhotse import CutSet
from lhotse.cut.base import Cut
from lhotse.utils import ifnone
from torch.utils.data import Dataset, DataLoader
from ..data.speech.utils import is_scalar, concat_list, apply_to_sample


class CutSetPtDataset(Dataset, CutSet):
    def __init__(self,
                 cuts: Optional[Iterable[Cut]] = None,
                 mixed_method: str = 'Unweighted',
                 mixed_weights: Optional[List[float]]=None,
                 read_keys: List[str] = ['audio', 'feat',],
                 stand_dict: Optional[Dict[str, str]] = None):
        super().__init__()
        assert mixed_method in ['Unweighted', 'Weighted', 'stop_early']
        self.cuts = ifnone(cuts, [])
        self.mixed_method = mixed_method
        self.mixed_weights = mixed_weights
        self.dataset = None
        self.read_keys = read_keys
        self.sorted = False
        self.stand_dict = stand_dict
        self.carry_filename = False

    def merge_dataset(self):
        cutset = self.cuts #CutSet.from_cuts(self.cuts)
        if self.mixed_method == 'stop_early':
            cuts = CutSet.mux(*cutset, stop_early=True)
        elif self.mixed_method == 'Weighted':
            if self.mixed_weights is None:
                mixed_weights = [len(cut) for cut in cutset]
            cuts = CutSet.mux(*cutset, weights = mixed_weights)
        else:
            cuts = CutSet.mux(*cutset)
        self.dataset = cuts

    def _get_supplement_data_(self, values_dict, indice: List[int]):
        values_dict = dict()
        if self.carry_filename:
            file_names = self._get_data_(indice, [self.file_path_key])[self.file_path_key]
            begin_time = self._get_data_(indice, ["begin_time"])["begin_time"]
            values_dict['file_names'] = [f"{filename}_{begin_time}" for filename, begin_time in zip(file_names,begin_time)]
        return values_dict

    def get_supplement_data(self, values_dict, indice: Union[int, List[int]]):
        if isinstance(indice, (list, tuple)):
            if self.sorted:
                indice = sorted(indice)
            origin_supplement = self._get_supplement_data_(values_dict, indice)
        else:
            supplement = self._get_supplement_data_(values_dict, [indice])
            origin_supplement = dict()
            for k, v in supplement.items():
                origin_supplement[k] = supplement[k][0]

        return origin_supplement

    def postprocess_items(self, values_dict: Dict[str, List[Any]], **kwargs) -> Dict[str, Any]:
        return values_dict

    def complete_data(self, values_dict: Dict[str, Any], indice: Union[int, List[int]]):
        values_dict.update(self.get_supplement_data(values_dict, indice))
        whole_values_dict = self.postprocess_items(values_dict)
        return whole_values_dict

    def __getitem__(self, indice: Union[int, Iterable]):
        values_dict = self._get_data_(indice, self.read_keys)
        if self.stand_dict is not None:
            stand_value_dict = dict()
            for k, v in values_dict.items():
                if k in self.stand_dict:
                    stand_value_dict[self.stand_dict[k]] = v
                else:
                    stand_value_dict[k] = v
            values_dict = stand_value_dict
        if is_scalar(indice) or is_scalar(indice[0]):
            whole_values_dict = self.complete_data(values_dict, indice)
        else:
            whole_values_dict = []
            for values_dict_, indice_ in zip(values_dict, indice):
                whole_values_dict.append(self.complete_data(values_dict_, indice_))
        return whole_values_dict

    def _get_data_(self, indice, read_keys) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """ In order to improve IO efficiency of HuggingfaceAudioDataset on HDFS, the kvreader
        fetches multiple batches (batch group) onces.
        Return is list of batches. IterableDataloader should pop back only one.
        There are two kinds of implemenets. If IterableDataloader is not modified, 
        In order to avoid modifying dataloader code of pytorch, the kv Sampler pops
        multiple batches indices at once to help reader read data of multiple batches.
        but the trainer or fetcher of dataloader only fetches data of single batch at once.
        So the kv Sampler provides None to disable reader read consequential data in the
        following code.
        Args:
            index ([type]): [description]

        Returns:
            [type]: [description]
        """
        if is_scalar(indice) or is_scalar(indice[0]):
            if self.sorted:
                indice = sorted(indice)
            all_values_dict = self.get_item(indice)
            values_dict = {key: all_values_dict[key] for key in read_keys}
            return values_dict
        else:
            num_parts = [len(sub_indice) for sub_indice in indice]
            all_indice = concat_list(indice)
            if self.sorted:
                all_indice = sorted(all_indice)

            all_values_dict = self.get_item(all_indice)
            values_dict = {key: all_values_dict[key] for key in read_keys}
            values_dict_list = list()
            accum_num = 0
            for _, num_part in enumerate(num_parts):
                sub_value_dict = dict()
                for k, v in values_dict.items():
                    sub_value_dict[k] = v[accum_num: accum_num + num_part]
                values_dict_list.append(sub_value_dict)
                accum_num += num_part
            return values_dict_list

    def get_feat(self, indice):
        batch_dict = {}
        cuts = []
        feats = []
        for idx in indice:
            cut: Cut = self.dataset[idx]
            if 'feat' in self.read_keys:
                feat = cut.load_features()
                feats.append(feat)
            cuts.append(cut)
        if len(feats) > 0:
            batch_dict['feat'] = feats
        cutset = CutSet.from_cuts(cuts)
        if 'audio' in self.read_keys:
            audios = cutset.load_audio(collate=False)
            batch_dict['audio'] = audios

        return batch_dict

    def get_item(self, item):
        return self.get_feat(item)
    
    def __len__(self) -> int:
        return len(self.dataset)


class Collator(object):
    def __init__(self, opt):
        data_cfg = opt_get(opt, ['data_cfg'], None)
        batch_size = opt_get(opt, ['batch_size'], 16)
        buffer_batch_group = opt_get(opt, ['buffer_batch_group'], 8)
        apply_half = opt_get(opt, ['apply_half'], False)
        half_type = opt_get(opt, ['half_type'], 'fp16')
        half_list = opt_get(opt, ['half_list'], None)
        self.batch_size = batch_size
        self.buffer_batch_group = buffer_batch_group
        self.data_cfg = data_cfg
        self.apply_half = apply_half
        self.half_type = half_type
        self.half_list = half_list
        self.inter_post = True
        self.ignored_keys = []

    def _postprocess_items_(self, batch_dict):
        return batch_dict

    def _postprocess_tensor_(self, batch_dict, force_inter=False):
        return batch_dict

    def _extra_info_(self, batch_dict):
        return batch_dict

    def _decode_(self, key, raw_data):
        if key in self.data_cfg:
            return raw_data
        else:
            if key not in self.ignored_keys:
                print(f"ignore the key {key}")
                self.ignored_keys.append(key)
            return None

    def _fp_convert_sample(self, sample):
        def apply_half(t):
            if t.dtype is torch.float32:
                return t.to(dtype=torch.half)
            return t

        def apply_bfloat16(t):
            if t.dtype is torch.float32:
                return t.to(dtype=torch.bfloat16)
            return t

        if self.half_type == 'fp16':
            sample = apply_to_sample(apply_half, sample)
        elif self.half_type == 'bf16':
            sample = apply_to_sample(apply_bfloat16, sample)

        return sample

    def _prepare_sample_(self, batch_dict):
        new_batch_dict = dict()
        if self.apply_half:
            if self.half_list is None:
                for k, v in batch_dict.items():
                    sample = self._fp_convert_sample(v)
                    new_batch_dict[k] = sample
            else:
                for k, v in batch_dict.items():
                    if k in self.half_list:
                        sample = self._fp_convert_sample(v)
                    else:
                        sample = v
                    new_batch_dict[k] = sample
        else:
            new_batch_dict = batch_dict
        return new_batch_dict

    def _collate_tensors_(self, data, key):
        result = []
        num_data = len(result)
        num_dim = data[0].dim()

        largest_dims = [0 for _ in range(num_dim)]
        for elem in data:
            result.append(elem)
            largest_dims = [max(current_largest, new_consideration) \
                for current_largest, new_consideration in zip(largest_dims, elem.shape)]

        # Now pad each tensor by the largest dimension.
        for i in range(num_data):
            padding_tuple = ()
            for d in range(num_dim):
                padding_needed = largest_dims[d] - result[i].shape[d]
                if padding_needed < 0:
                    raise ValueError(f"Padding needed is negative: {padding_needed}")
                padding_tuple = (0, padding_needed) + padding_tuple

            try:
                constant_val = self.data_cfg[key]['padding_val']
            except:
                constant_val = 0

            result[i] = F.pad(result[i], padding_tuple, value=constant_val)
        return torch.stack(result, dim=0)

    def _collate_one_batch_(self, batch):
        collated = {}
        for key, data in batch.items():
            if data is None:
                continue
            if isinstance(data[0], int):
                collated[key] = torch.LongTensor(data)
            elif isinstance(data[0], float):
                collated[key] = torch.FloatTensor(data)
            elif isinstance(data[0], str):
                collated[key] = data
            else:
                if isinstance(data[0], (np.ndarray)):
                    tensors = [torch.from_numpy(d.copy()) for d in data]
                else:
                    tensors = data
                if len(tensors[0].shape) > 0:
                    collated[key] = self._collate_tensors_(tensors, key)
                else:
                    try:
                        collated[key] = torch.stack(tensors)
                    except:
                        raise RuntimeError(f"{key} type is not compatitable.")
        return collated

    def __call__(self, compound_data):
        if isinstance(compound_data, (list,)):
            # It's a legacy that you should not pay much attention on it.
            # it handles with the old kv dataset which partitions a group of batches into a list of dicts.
            for data_dict in compound_data:
                for d_k, d_v in data_dict.items():
                    if d_v is None:
                        continue
                    if isinstance(d_v, (list, tuple)):
                        batch_group = True
                    else:
                        batch_group = False
                    break
                break
            if batch_group:
                value_batch_group = []
                for data_dict in compound_data:
                    value_batch = dict()
                    for d_k, d_v in data_dict.items():
                        batch_val = self._decode_(d_k, d_v)
                        if batch_val is not None:
                            value_batch[d_k] = batch_val
                    value_batch = self._postprocess_items_(value_batch)
                    collated_batch = self._collate_one_batch_(value_batch)
                    collated_batch = self._postprocess_tensor_(collated_batch)
                    collated_batch = self._prepare_sample_(collated_batch)
                    value_batch_group.append(collated_batch)
                return value_batch_group
            else:
                value_batch = dict()
                for data_dict in compound_data: 
                    for d_k, d_v in data_dict.items():
                        val = self._decode_(d_k, [d_v])
                        if val is not None:
                            if d_k in value_batch:
                                value_batch[d_k] += val
                            else:
                                value_batch[d_k] = val
                value_batch = self._postprocess_items_(value_batch)
                collated_batch = self._collate_one_batch_(value_batch)
                collated_batch = self._postprocess_tensor_(collated_batch)
                collated_batch = self._prepare_sample_(collated_batch)
                return collated_batch                           
        else:
            # The current practice, the dataset does not need to partition a group batches.
            # The input is a dict with the key which represents the feature name and the value which is a ordered
            # list of raw value.
            assert isinstance(
                compound_data, dict), "data from dataset to collate should be a dict or a list of dict"
            # get the length of a group of batches.
            data_size = 1
            for _, val in compound_data.items():
                data_size = len(val)
                break

            # patition the group
            enc_value_batch_group = []
            for i in range(self.buffer_batch_group):
                if (i + 1) * self.batch_size > data_size:
                    if i == 0:
                        print("batch size is too large to the dataset")
                if i * self.batch_size >= data_size:
                    if i == 0:
                        raise RuntimeError("could not get any samplers in the dataset")
                    break

                value_batch = dict()
                # partition the value of the dict into batches respected to each feature
                for d_k, d_v in compound_data.items():
                    data = d_v[i * self.batch_size: (i + 1) * self.batch_size]
                    value_batch[d_k] = data
                enc_value_batch_group.append(value_batch)

            value_batch_group = []
            for enc_value_batch in enc_value_batch_group:
                value_batch = dict()
                for d_k, d_v in enc_value_batch.items():
                    batch_val = self._decode_(d_k, d_v)
                    if batch_val is not None:
                        value_batch[d_k] = batch_val
                    value_batch[d_k] = batch_val
                value_batch = self._postprocess_items_(value_batch)
                collated_batch = self._collate_one_batch_(value_batch)
                collated_batch = self._postprocess_tensor_(collated_batch)
                collated_batch = self._prepare_sample_(collated_batch)
                value_batch_group.append(collated_batch)
            return value_batch_group


if __name__ == '__main__':
    jsonl_f = '/home/wumenglin/workspace/data/tokenized/hifitts/hifitts_cuts_92_clean_dev.jsonl'
    jsonl_f1 = '/home/wumenglin/workspace/data/tokenized/hifitts/hifitts_cuts_92_clean_dev.jsonl'
    cuts = CutSet.from_jsonl(jsonl_f)
    cuts_f1 = CutSet.from_jsonl(jsonl_f)
    dataset = CutSetPtDataset([cuts, cuts_f1], stand_dict={'feat': 'codec'})
    dataset.merge_dataset()

    data = dataset[[[0, 5, 7, 9]]]

    batch_size = 4
    padding_idx = 65535

    collate_cfg = {
        'batch_size': batch_size,
        'codec':  {
            "padding_val": padding_idx
        },
        'audio': {
            'padding_val': -0.0
        }
    }
    collator = Collator(collate_cfg)

    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collator, sampler=None)

    for i, batch in enumerate(dataloader):
        print(batch)
    
    a = 1
    
import sys
import os
import yaml
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import pandas as pd
import torchaudio
import torchaudio.transforms as T
import torch.nn.functional as F
from collections import defaultdict


class handwritten(object):
    def __init__(self, num_of_sel_Tx, num_of_sel_Rx) -> None:
        self.num_of_Tx = 3
        self.num_of_Mx = 3
        self.num_of_Rx = 3

        self.num_of_sel_Tx = num_of_sel_Tx
        self.num_of_sel_Rx = num_of_sel_Rx

        if self.num_of_sel_Rx >= self.num_of_Tx:
            self.num_of_sel_Rx = self.num_of_Tx
        if self.num_of_sel_Tx >= self.num_of_Mx*self.num_of_Rx:
            self.num_of_sel_Tx = int(self.num_of_Mx*self.num_of_Rx)
        print(f"The num_of_sel_Rx {self.num_of_sel_Rx} corresponds to E_t.")
        print(f"The num_of_sel_Tx {self.num_of_sel_Tx} corresponds to E_k.")

        self.init_data()

    def init_data(self):
        trans = transforms.Compose([
            transforms.ToTensor(), 
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        d_train_img = datasets.MNIST(
            './data/mnist/', train=True, 
            download=True, transform=trans
        )
        d_test_img = datasets.MNIST(
            './data/mnist/', train=False, 
            download=True, transform=trans
        )

        trans_fsdd_train = FSDDTransform(target_sample_rate=8000, target_length=8000, augment=True)
        trans_fsdd_test = FSDDTransform(target_sample_rate=8000, target_length=8000, augment=False)
        
        d_train_audio = FSDDDataset(
            csv_file="./data/FSDD/fsdd_dataset_train.csv", 
            transform=trans_fsdd_train,
            augment=True
        )
        d_test_audio = FSDDDataset(
            csv_file="./data/FSDD/fsdd_dataset_test.csv", 
            transform=trans_fsdd_test,
            augment=False
        )

        self.d_train = AlignedDataset(d_train_img, d_train_audio, is_train=True)
        self.d_test = AlignedDataset(d_test_img, d_test_audio, is_train=False)

    def load(self, train= True, bs= 20, shuffle= True, k= 0):
        return DataLoader(self.d_train, bs, shuffle, num_workers= k) \
            if train else DataLoader(self.d_test, bs, shuffle, num_workers= k)

    def min_max_norm(self, tensor):
        min_val = tensor.min()
        max_val = tensor.max()
        return (tensor - min_val) / (max_val - min_val + 1e-8)

    def custom(self, d, no_conv= True):
        (x_img, x_audio), y = d
        bs = len(y)

        x_img_rand = torch.randn_like(x_img[:,:,14:,14:])
        x_img_rand_2 = torch.randn_like(x_img[:,:,14:,14:])
        x_img_crop = x_img[:,:,7:21,7:21]
        
        x_F = torch.fft.fft2(x_img_crop, dim=(-2, -1))
        x_F_shift_mag = self.min_max_norm(torch.abs(torch.fft.fftshift(x_F, dim=(-2, -1))))
        
        if no_conv:
            _x_1 = x_img_rand.reshape(bs, -1)
            _x_2 = x_img_rand_2.reshape(bs, -1)
            _x_3 = x_F_shift_mag.reshape(bs, -1)
            _x_4 = x_audio.reshape(bs, -1)
        else:
            _x_1 = x_img_rand
            _x_2 = x_img_rand_2
            _x_3 = x_F_shift_mag
            _x_4 = x_audio

        _x = [
            [_x_1, _x_2, _x_3], 
            [_x_2, _x_3, _x_4], 
            [_x_3, _x_4, _x_1],
        ]

        _y = torch.stack([self.y_divide(y_) for y_ in y])
        _y = torch.stack(torch.split(_y, 1, dim=1), dim=0)

        return _x, _y 
    
    def y_divide(self, y):
        y_odd_or_even = 0 if y % 2 == 0 else 1
        y_code = [0, 4, 6, 8, 9]
        y_04689 = y_code.index(y) if y in y_code else 5

        return torch.tensor(
            [y_odd_or_even, y_04689, y]
        )

    def info(self):
        info = {
            'input_dim': [
                [196, 196, 196],
                [196, 196, 512],
                [196, 512, 196]
            ],
            'output_dim': [2, 6, 10],
            'mi': [0.6, 1.3, 2.1],
            'num_of_Tx': self.num_of_Tx,
            'num_of_Mx': self.num_of_Mx,
            'num_of_Rx': self.num_of_Rx,
            'num_of_sel_Tx': self.num_of_sel_Tx,
            'num_of_sel_Rx': self.num_of_sel_Rx,
        }
        print(f"The info of settings under handwritten: {info}")
        return info


class FSDDTransform:
    def __init__(self, target_sample_rate=8000, target_length=8000, augment=False):
        self.resample = torchaudio.transforms.Resample(orig_freq=8000, new_freq=target_sample_rate)
        self.target_length = target_length
        self.augment = augment
    
    def __call__(self, wvfm, sample_rate):
        wvfm = self.resample(wvfm)
        
        if self.augment:
            noise = torch.randn_like(wvfm) * 1e-5
            wvfm = wvfm + noise
            
            gain = 0.95 + 0.1 * torch.rand(1).item()
            wvfm = wvfm * gain
        
        if wvfm.size(1) < self.target_length:
            wvfm = F.pad(wvfm, (0, self.target_length - wvfm.size(1)))
        else:
            wvfm = wvfm[:, :self.target_length]
        
        return wvfm


class FSDDDataset(Dataset):
    def __init__(self, csv_file, transform=None, augment=False):
        self.df = pd.read_csv(csv_file)
        self.transform = transform
        self.to_mel = T.MelSpectrogram(
            sample_rate=8000, 
            n_fft=2048, 
            hop_length=256, 
            n_mels=16
        )
        self.augment = augment
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx): 
            idx = idx.tolist()

        row = self.df.iloc[idx]
        audio_path = row['path']
        label = row['label']

        wvfm, sample_rate = torchaudio.load(audio_path)

        if self.transform: 
            wvfm = self.transform(wvfm, sample_rate)
        
        mel_spec = self.to_mel(wvfm)
        
        if mel_spec.dim() == 2:
            mel_spec = mel_spec.unsqueeze(0)
        
        if self.augment:
            freq_mask_param = 10
            if freq_mask_param > 0:
                freq_mask = T.FrequencyMasking(freq_mask_param)
                mel_spec = freq_mask(mel_spec)
            
            time_mask_param = 20
            if time_mask_param > 0:
                time_mask = T.TimeMasking(time_mask_param)
                mel_spec = time_mask(mel_spec)
            
            scale = 0.95 + 0.1 * torch.rand(1).item()
            mel_spec = mel_spec * scale
        
        raw_data = (mel_spec - mel_spec.mean()) / (mel_spec.std() + 1e-8)
        
        return raw_data, label


class AlignedDataset(Dataset):
    def __init__(self, mnist_dataset, fsdd_dataset, is_train=True):
        self.mnist_data = []
        self.fsdd_data = []
        self.labels = []
        self.is_train = is_train
        
        mnist_dict = defaultdict(list)
        for i in range(len(mnist_dataset)):
            img, label = mnist_dataset[i]
            mnist_dict[label].append(img)
        
        fsdd_dict = defaultdict(list)
        for i in range(len(fsdd_dataset)):
            audio, label = fsdd_dataset[i]
            if audio.dim() == 2:
                audio = audio.unsqueeze(0)
            fsdd_dict[label].append(audio)
        
        for label in range(10):
            mnist_samples = mnist_dict[label]
            fsdd_samples = fsdd_dict[label]
            
            num_mnist = len(mnist_samples)
            num_fsdd = len(fsdd_samples)
            
            if num_fsdd == 0:
                print(f"Warning: No FSDD samples for label {label}")
                continue
                
            multiplier = (num_mnist + num_fsdd - 1) // num_fsdd
            print(f"Label {label}: MNIST samples={num_mnist}, FSDD samples={num_fsdd}, Multiplier={multiplier}")
            
            augmented_fsdd = []
            for audio in fsdd_samples:
                augmented_fsdd.append(audio)
                
                if is_train:
                    for i in range(1, multiplier):
                        augmented_audio = self.augment_audio(audio.clone())
                        augmented_fsdd.append(augmented_audio)
            
            if len(augmented_fsdd) > num_mnist:
                augmented_fsdd = augmented_fsdd[:num_mnist]
            else:
                while len(augmented_fsdd) < num_mnist:
                    augmented_fsdd.append(augmented_fsdd[-1])
            
            self.mnist_data.extend(mnist_samples)
            self.fsdd_data.extend(augmented_fsdd)
            self.labels.extend([label] * num_mnist)
        
        assert len(self.mnist_data) == len(self.fsdd_data) == len(self.labels)
        print(f"Created aligned dataset with {len(self.mnist_data)} samples")
    
    def __len__(self):
        return len(self.mnist_data)
    
    def __getitem__(self, idx):
        return (self.mnist_data[idx], self.fsdd_data[idx]), self.labels[idx]
    
    def augment_audio(self, audio):
        if audio.dim() == 4:
            audio = audio.squeeze(0)
        if audio.dim() == 2:
            audio = audio.unsqueeze(0)
            
        noise = torch.randn_like(audio) * 1e-5
        augmented = audio + noise
        
        volume = 0.95 + 0.1 * torch.rand(1).item()
        augmented = augmented * volume
        
        shift = torch.randint(-10, 10, (1,)).item()
        if shift != 0:
            if shift > 0:
                augmented = torch.cat([
                    augmented[:, :, shift:], 
                    torch.zeros_like(augmented[:, :, :shift])
                ], dim=2)
            else:
                shift = abs(shift)
                augmented = torch.cat([
                    torch.zeros_like(augmented[:, :, :shift]),
                    augmented[:, :, :-shift]
                ], dim=2)
        
        return augmented
    

class mmfi(object):
    def __init__(self, num_of_sel_Tx, num_of_sel_Rx):
        self.num_of_Tx = 4
        self.num_of_Mx = 4
        self.num_of_Rx = 3

        self.num_of_sel_Tx = num_of_sel_Tx
        self.num_of_sel_Rx = num_of_sel_Rx

        if self.num_of_sel_Rx >= self.num_of_Tx:
            self.num_of_sel_Rx = self.num_of_Tx
        if self.num_of_sel_Tx >= self.num_of_Mx*self.num_of_Rx:
            self.num_of_sel_Tx = int(self.num_of_Mx*self.num_of_Rx)
        print(f"The num_of_sel_Rx {self.num_of_sel_Rx} corresponds to E_t.")
        print(f"The num_of_sel_Tx {self.num_of_sel_Tx} corresponds to E_k.")
            
        dataset_root='./data/MMFI/MMFi_Dataset/MMFiData'
        config_file='./data/MMFI/MMFi_Dataset/MMFi_dataset-main/config.yaml'

        mmfi_lib_path = os.path.join(os.path.dirname(__file__), '../data/MMFI/MMFi_Dataset/MMFi_dataset-main/mmfi_lib')
        mmfi_lib_path = os.path.abspath(mmfi_lib_path)
        if mmfi_lib_path not in sys.path:
            sys.path.append(mmfi_lib_path)
        try:
            from mmfi import make_dataset, make_dataloader
        except ImportError as e:
            print(f"Could not import mmfi: {e}. Make sure mmfi.py is in the correct path and accessible.")
            raise

        self.dataset_root = dataset_root
        self.config_file = config_file

        with open(config_file, 'r') as fd:
            self.config = yaml.load(fd, Loader=yaml.FullLoader)

        self.train_dataset, self.val_dataset = make_dataset(dataset_root, self.config)
        self.rng_generator = torch.manual_seed(self.config['init_rand_seed'])
        self.make_dataloader = make_dataloader

        self.data_shape()
        self.init_y_table()

    def load(self, bs= 20, train=True):
        """
        Returns the train or validation DataLoader.
        """
        return self.make_dataloader(
            self.train_dataset, is_training=train, generator=self.rng_generator, batch_size=bs
        ) if train else self.make_dataloader(
            self.val_dataset, is_training=train, generator=self.rng_generator, batch_size=bs
        )

    def info(self):
        info = {
            'input_dim': [
                [34, 34, 34, 250],
                [16384, 3600, 250, 250],
                [2700, 3600, 250, 250],
                [34, 34, 250, 250],
            ],
            'output_dim': [51, 27, 4],
            'mi': [0, 3, 1.3],
            'num_of_Tx': self.num_of_Tx,
            'num_of_Mx': self.num_of_Mx,
            'num_of_Rx': self.num_of_Rx,
            'num_of_sel_Tx': self.num_of_sel_Tx,
            'num_of_sel_Rx': self.num_of_sel_Rx,
            'train_dataset_len': len(self.train_dataset),
            'val_dataset_len': len(self.val_dataset),
            'init_rand_seed': self.config.get('init_rand_seed', None),
        }
        print(f"The info of settings under mmfi: {info}")
        return info

    def data_shape(self):
        self.target_shapes = {
            'input_rgb':      (17, 2),
            'input_infra1':   (17, 2),
            'input_infra2':   (17, 2),
            'input_depth':    (128, 128),
            'input_lidar':    (1200, 3),
            'input_mmwave':   (50, 5),
            'input_wifi-csi': (3, 90, 10)
        }
        return self.target_shapes
    
    def init_y_table(self):
        self.action_to_idx = {action: idx for idx, action in enumerate([
            'A01', 'A02', 'A03', 'A04', 'A05', 'A06', 'A07', 'A08', 'A09', 'A10', 'A11', 'A12', 'A13', 'A14',
            'A15', 'A16', 'A17', 'A18', 'A19', 'A20', 'A21', 'A22', 'A23', 'A24', 'A25', 'A26', 'A27'
        ])}
        self.scene_to_idx = {scene: idx for idx, scene in enumerate(['E01', 'E02', 'E03', 'E04'])}

    def custom(self, batch, no_conv=True):
        x = []
        for key, shape in self.target_shapes.items():
            tensor = batch[key]
            current_shape = tensor.shape[1:]
            # Step 1: Pad if needed
            pad_width = []
            for cur, tgt in zip(reversed(current_shape), reversed(shape)):
                if cur < tgt:
                    pad_width.extend([0, tgt - cur])
                else:
                    pad_width.extend([0, 0])
            if any(cur < tgt for cur, tgt in zip(current_shape, shape)):
                tensor = F.pad(tensor, pad_width)
            # Step 2: Random crop if needed
            slices = [slice(0, tensor.shape[0])]  # batch dimension
            for cur, tgt in zip(tensor.shape[1:], shape):
                if cur > tgt:
                    start = int(torch.randint(0, cur - tgt + 1, (1,)).item())
                    slices.append(slice(start, start + tgt))
                else:
                    slices.append(slice(0, tgt))
            tensor = tensor[tuple(slices)]
            if no_conv:
                tensor = tensor.reshape(tensor.size(0), -1)
            x.append(tensor)
        
        _x = [
            [x[0], x[1], x[2], torch.randn_like(x[5])], 
            [x[3], x[4], x[5], torch.randn_like(x[5])], 
            [x[6], x[4], x[5], torch.randn_like(x[5])],
            [x[1], x[2], x[5], torch.randn_like(x[5])],
        ]
        _y = [
            batch['output'].reshape(batch['output'].size(0), -1),
            torch.tensor([self.action_to_idx[code] for code in batch['action']]),
            torch.tensor([self.scene_to_idx[code] for code in batch['scene']]),
        ]
        return _x, _y



if __name__ == "__main__":
    d = mmfi(2, 2)
    print(d.info())

    for _ in range(100):
        for batch in d.load(train=True):
            print(batch['output'].shape)
            break

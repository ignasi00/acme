import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from utility.cutout import Cutout

import random


class MiSampler:
    def __init__(self, data_source):
        self.data_source = data_source

        self.tipos_set = []
        self.dicc_set = {}
        for ind, tipo in enumerate(self.data_source):
            if tipo[1] in self.dicc_set:
                self.dicc_set[tipo[1]].append(ind)
            else:
                self.tipos_set.append(tipo[1])
                self.dicc_set[tipo[1]] = [ind]

    def __iter__(self):
        for tipo in self.tipos_set:
            random.shuffle(self.dicc_set[tipo])

        #print([self.dicc_set[tipo][ind] for ind in range(len(self.dicc_set[0])) for tipo in self.tipos_set])
        return iter(self.dicc_set[tipo][ind] for ind in range(len(self.dicc_set[0])) for tipo in sorted(self.tipos_set, key=lambda _: random.random()))
        #return(iter(ind for ind in self.dicc_set[tipo] for tipo in self.tipos_set))
        return iter(range(len(self.data_source)))
    
    def __len__(self):
        return len(self.data_source)


class MiCifar:
    def __init__(self, batch_size, threads, root=None):
        root, download = (root, False) if root else ('./data', True)

        mean, std = self._get_statistics(root, download)

        train_transform = transforms.Compose([
            torchvision.transforms.RandomCrop(size=(32, 32), padding=4),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.AutoAugment(policy=torchvision.transforms.AutoAugmentPolicy.CIFAR10),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
            Cutout(),
        ])

        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

        train_set = torchvision.datasets.CIFAR100(root=root, train=True, download=download, transform=train_transform)
        test_set = torchvision.datasets.CIFAR100(root=root, train=False, download=download, transform=test_transform)

        self.train = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=False, num_workers=threads, drop_last=True,
                                                 sampler=MiSampler(train_set))
        self.test = torch.utils.data.DataLoader(test_set, batch_size=64, shuffle=False, num_workers=threads)

        self.classes = ('beaver', 'dolphin', 'otter', 'seal', 'whale',
                        'aquarium fish', 'flatfish', 'ray', 'shark', 'trout',
                        'orchids', 'poppies', 'roses', 'sunflowers', 'tulips',
                        'bottles', 'bowls', 'cans', 'cups', 'plates',
                        'apples', 'mushrooms', 'oranges', 'pears', 'sweet peppers',
                        'clock', 'computer keyboard', 'lamp', 'telephone', 'television',
                        'bed', 'chair', 'couch', 'table', 'wardrobe',
                        'bee', 'beetle', 'butterfly', 'caterpillar', 'cockroach',
                        'bear', 'leopard', 'lion', 'tiger', 'wolf',
                        'bridge', 'castle', 'house', 'road', 'skyscraper',
                        'cloud', 'forest', 'mountain', 'plain', 'sea',
                        'camel', 'cattle', 'chimpanzee', 'elephant', 'kangaroo',
                        'fox', 'porcupine', 'possum', 'raccoon', 'skunk',
                        'crab', 'lobster', 'snail', 'spider', 'worm',
                        'baby', 'boy', 'girl', 'man', 'woman',
                        'crocodile', 'dinosaur', 'lizard', 'snake', 'turtle',
                        'hamster', 'mouse', 'rabbit', 'shrew', 'squirrel',
                        'maple', 'oak', 'palm', 'pine', 'willow',
                        'bicycle', 'bus', 'motorcycle', 'pickup truck', 'train',
                        'lawn-mower', 'rocket', 'streetcar', 'tank', 'tractor',
                       )


    def _get_statistics(self, root, download):
        train_set = torchvision.datasets.CIFAR100(root=root, train=True, download=download, transform=transforms.ToTensor())

        data = torch.cat([d[0] for d in DataLoader(train_set)])
        return data.mean(dim=[0, 2, 3]), data.std(dim=[0, 2, 3])

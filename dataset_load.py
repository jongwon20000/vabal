import torch
import torchvision
import torchvision.transforms as transforms

# when sample_idx is given: (Selectively use only the indexed samples)
# The remaining samples are returned by valloader
def dataset_loader(dataset_name, datapool_idx=[], sample_idx=[], batch_size=128, verbose=False):
        
    trainloader = None
    valloader = None  # With empty sample_idx, return None
    testloader = None
    classes = ()
        
    if dataset_name is 'CIFAR10':
        
        (trainloader, valloader, testloader, classes) = CIFAR10_loader(datapool_idx, sample_idx, batch_size, verbose)
        
    else:
        
        raise NameError('UNKNOWN DATASET NAME')
        
    return (trainloader, valloader, testloader, classes)


##############################################
## CIFAR10 dataset loader
def CIFAR10_loader(datapool_idx=[], sample_idx=[], batch_size=128, verbose=False):

    # transformer loading...
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=transform_train)
    # when datapool_idx is given: (Selectively use only the datapooled samples)
    if datapool_idx:
        trainset = torch.utils.data.Subset(trainset, datapool_idx)
    # when sample_idx is given: (Selectively use only the indexed samples)
    if sample_idx:
        valset = torch.utils.data.Subset(trainset, [x for x in range(trainset.__len__()) if x not in sample_idx])
        valloader = torch.utils.data.DataLoader(valset, batch_size=batch_size, shuffle=False, num_workers=2)
        trainset = torch.utils.data.Subset(trainset, sample_idx)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
        
    # testset loading...
    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=False, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=100, shuffle=False, num_workers=2)

    # classes...
    classes = ('plane', 'car', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck')
    
    # histogram of classes...
    if verbose:
        num_classes = len(classes)
        cls_counter = torch.zeros(num_classes, dtype=torch.float64)
        for (_, targets) in trainloader:            
            cls_counter += torch.histc(targets.float(), bins=num_classes, min=0, max=num_classes)
        print('class counter: ') 
        print(cls_counter)
    
    return (trainloader, valloader, testloader, classes)
    

def dataset_length(dataset_name):
    
    if dataset_name is 'CIFAR10':
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True)
        testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True)
                
    else:
        raise NameError('UNKNOWN DATASET NAME')
        
    return (trainset.__len__(), testset.__len__())


def datapool_sampling(dataset_name, sampling_classes=[], sampling_ratio=1.0):
    
    import random
    import numpy as np
    
    if dataset_name is 'CIFAR10':
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transforms.ToTensor())
        testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True)
    else:
        raise NameError('UNKNOWN DATASET NAME')
        
    if sampling_classes:
        
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=1000, shuffle=False, num_workers=2)
        labels_list = []
        for (_, labels) in trainloader:
            labels_list += labels
        
        datapool_idx = []
        for curr_sampling_class in range(max(labels_list)+1):
            if curr_sampling_class in sampling_classes:                
                curr_sampling_idx_list = np.where(np.array(labels_list) == curr_sampling_class)[0].tolist()
                random.shuffle(curr_sampling_idx_list)
                datapool_idx += curr_sampling_idx_list[:int(len(curr_sampling_idx_list)*sampling_ratio)]
            else:
                datapool_idx += np.where(np.array(labels_list) == curr_sampling_class)[0].tolist()
                            
        return (datapool_idx, len(datapool_idx), testset.__len__())
    
    else:
        
        return ([], trainset.__len__(), testset.__len__())
    
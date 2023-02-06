from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

CATEGORIES = {
    'dog': 0,
    'elephant': 1,
    'giraffe': 2,
    'guitar': 3,
    'horse': 4,
    'house': 5,
    'person': 6,
}


class PACSDatasetBaseline(Dataset):
    def __init__(self, examples, transform):
        self.examples = examples
        self.transform = transform

    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, index):
        img_path, y = self.examples[index]
        x = self.transform(Image.open(img_path).convert('RGB'))
        return x, y


class PACSDatasetDomainDisentangle(Dataset):
    def __init__(self, examples, transform):
        self.examples = examples
        self.transform = transform

    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, index):
        img_path, y = self.examples[index]
        x = self.transform(Image.open(img_path).convert('RGB'))
        return x, y


class PACSDatasetClipDisentangle(Dataset):
    def __init__(self, examples, transform):
        self.examples = examples
        self.transform = transform

    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, index):
        img_path, y, descriptions = self.examples[index]
        x = self.transform(Image.open(img_path).convert('RGB'))
        return x, y, descriptions


class PACSDatasetDescriptions(Dataset):
    def __init__(self, examples, transform):
        self.examples = examples
        self.transform = transform

    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, index):
        img_path, descriptions = self.examples[index]
        x = self.transform(Image.open(img_path).convert('RGB'))
        return x, descriptions


def read_lines(data_path, domain_name):
    examples = {}
    with open(f'{data_path}/{domain_name}.txt') as f:
        lines = f.readlines()

    for line in lines: 
        line = line.strip().split()[0].split('/')
        category_name = line[3]
        category_idx = CATEGORIES[category_name]
        image_name = line[4]
        image_path = f'{data_path}/kfold/{domain_name}/{category_name}/{image_name}'
        if category_idx not in examples.keys():
            examples[category_idx] = [image_path]
        else:
            examples[category_idx].append(image_path)
    return examples


def read_clip_labels(data_path, domain_name):
    # returns a dictionary of images with their CLIP labels
    text_files = ['groupe1AML.txt', 'groupe1DAAI.txt', 'groupe2AML.txt', 'groupe2DAAI.txt', 'groupe3AML.txt',
            'groupe3DAAI.txt', 'groupe5AML.txt', 'groupe6AML.txt', 'group_299837_300451_300708.txt']
    examples = {}
    
    for file_name in text_files:
        with open(f'{data_path}/clip_labels/{file_name}') as f:
            line = f.read()

        line = line[1:-1:1]
        items = line.split('},')
        for i in range(len(items)-1):
            items[i] = items[i] + '}'

        for i in items: 
            item_dict = eval(i)
            item_domain_name = item_dict['image_name'].split('/')[0]
            if item_domain_name == domain_name:
                examples[f'{data_path}/kfold/' + item_dict[f'image_name']] = ', '.join(item_dict['descriptions'])

    return examples

   
def build_splits_baseline(opt):
    source_domain = 'art_painting'
    target_domain = opt['target_domain']

    source_examples = read_lines(opt['data_path'], source_domain)
    target_examples = read_lines(opt['data_path'], target_domain)

    # Compute ratios of examples for each category
    source_category_ratios = {category_idx: len(examples_list) for category_idx, examples_list in source_examples.items()}
    source_total_examples = sum(source_category_ratios.values())
    source_category_ratios = {category_idx: c / source_total_examples for category_idx, c in source_category_ratios.items()}

    # Build splits - we train only on the source domain (Art Painting)
    val_split_length = source_total_examples * 0.2 # 20% of the training split used for validation

    train_examples = []
    val_examples = []
    test_examples = []

    for category_idx, examples_list in source_examples.items():
        split_idx = round(source_category_ratios[category_idx] * val_split_length)
        for i, example in enumerate(examples_list):
            if i > split_idx:
                train_examples.append([example, category_idx]) # each pair is [path_to_img, class_label]
            else:
                val_examples.append([example, category_idx]) # each pair is [path_to_img, class_label]
    
    for category_idx, examples_list in target_examples.items():
        for example in examples_list:
            test_examples.append([example, category_idx]) # each pair is [path_to_img, class_label]
    
    # Transforms
    normalize = T.Normalize([0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # ResNet18 - ImageNet Normalization

    train_transform = T.Compose([
        T.Resize(256),
        T.RandAugment(3, 15),
        T.CenterCrop(224),
        T.ToTensor(),
        normalize
    ])

    eval_transform = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        normalize
    ])

    # Dataloaders
    train_loader = DataLoader(PACSDatasetBaseline(train_examples, train_transform), batch_size=opt['batch_size'], num_workers=opt['num_workers'], shuffle=True)
    val_loader = DataLoader(PACSDatasetBaseline(val_examples, eval_transform), batch_size=opt['batch_size'], num_workers=opt['num_workers'], shuffle=False)
    test_loader = DataLoader(PACSDatasetBaseline(test_examples, eval_transform), batch_size=opt['batch_size'], num_workers=opt['num_workers'], shuffle=False)

    return train_loader, val_loader, test_loader


def build_splits_domain_disentangle(opt):
    source_domain = 'art_painting'
    target_domain = opt['target_domain']

    source_examples = read_lines(opt['data_path'], source_domain)
    target_examples = read_lines(opt['data_path'], target_domain)

    # Compute ratios of examples for each category
    source_category_ratios = {category_idx: len(examples_list) for category_idx, examples_list in source_examples.items()}
    source_total_examples = sum(source_category_ratios.values())
    source_category_ratios = {category_idx: c / source_total_examples for category_idx, c in source_category_ratios.items()}
    
    target_category_ratios = {category_idx: len(examples_list) for category_idx, examples_list in target_examples.items()}
    target_total_examples = sum(target_category_ratios.values())
    source_domain_ratio = source_total_examples / (source_total_examples + target_total_examples)

    # Build splits - we train on both source and target domain (for domain classification)
    source_val_split_length = source_total_examples * 0.2 
    target_val_split_length = target_total_examples * 0.2 

    train_examples = []
    val_examples = []
    test_examples = []

    for category_idx, examples_list in source_examples.items():
        split_idx = round(source_category_ratios[category_idx] * source_val_split_length * source_domain_ratio)
        for i, example in enumerate(examples_list):
            if i > split_idx:
                train_examples.append([example, category_idx]) # each pair is [path_to_img, class_label]
            else:
                val_examples.append([example, category_idx]) # each pair is [path_to_img, class_label]
    
    split_idx = round(target_val_split_length * (1 - source_domain_ratio))
    loop_index = 0
    for category_idx, examples_list in target_examples.items():
        for example in examples_list:
            loop_index += 1
            test_examples.append([example, category_idx]) # each pair is [path_to_img, class_label]       
            # including target examples in the train_examples for domain classifier
            # we specify them with label=7 to be able to determine them       
            if loop_index > split_idx:
                train_examples.append([example, 7]) # each pair is [path_to_img, 7]
            else:
                val_examples.append([example, 7]) # each pair is [path_to_img, 7]


    # Transforms
    normalize = T.Normalize([0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # ResNet18 - ImageNet Normalization

    train_transform = T.Compose([
        T.Resize(256),
        T.RandAugment(3, 15),
        T.CenterCrop(224),
        T.ToTensor(),
        normalize
    ])

    eval_transform = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        normalize
    ])

    # Dataloaders
    train_loader = DataLoader(PACSDatasetDomainDisentangle(train_examples, train_transform), batch_size=opt['batch_size'], num_workers=opt['num_workers'], shuffle=True)
    val_loader = DataLoader(PACSDatasetDomainDisentangle(val_examples, eval_transform), batch_size=opt['batch_size'], num_workers=opt['num_workers'], shuffle=False)
    test_loader = DataLoader(PACSDatasetDomainDisentangle(test_examples, eval_transform), batch_size=opt['batch_size'], num_workers=opt['num_workers'], shuffle=False)

    return train_loader, val_loader, test_loader


def build_splits_clip_disentangle(opt):
    source_domain = 'art_painting'
    target_domain = opt['target_domain']

    source_examples = read_lines(opt['data_path'], source_domain)
    target_examples = read_lines(opt['data_path'], target_domain)
    source_clip_examples = read_clip_labels(opt['data_path'], source_domain)
    target_clip_examples = read_clip_labels(opt['data_path'], target_domain)

    # Compute ratios of examples for each category
    source_category_ratios = {category_idx: len(examples_list) for category_idx, examples_list in source_examples.items()}
    source_total_examples = sum(source_category_ratios.values())
    source_category_ratios = {category_idx: c / source_total_examples for category_idx, c in source_category_ratios.items()}
    
    target_category_ratios = {category_idx: len(examples_list) for category_idx, examples_list in target_examples.items()}
    target_total_examples = sum(target_category_ratios.values())
    source_domain_ratio = source_total_examples / (source_total_examples + target_total_examples)

    # Build splits - we train on both source and target domain (for domain classification)
    source_val_split_length = source_total_examples * 0.2 
    target_val_split_length = target_total_examples * 0.2 

    train_examples = []
    val_examples = []
    test_examples = []

    for category_idx, examples_list in source_examples.items():
        split_idx = round(source_category_ratios[category_idx] * source_val_split_length * source_domain_ratio)
        for i, example in enumerate(examples_list):
            if i > split_idx:
                if example in source_clip_examples:
                    train_examples.append([example, category_idx, source_clip_examples[example]]) # each item is is [path_to_img, class_label, clip labels]
                else:
                    train_examples.append([example, category_idx, ''])
            else:
                val_examples.append([example, category_idx, '']) # each pair is [path_to_img, class_label]
    
    split_idx = round(target_val_split_length * (1 - source_domain_ratio))
    loop_index = 0
    for category_idx, examples_list in target_examples.items():
        for example in examples_list:
            loop_index += 1
            test_examples.append([example, category_idx, '']) # each pair is [path_to_img, class_label, '']       
            # including target examples in the train_examples for domain classifier
            # we specify them with label=7 to be able to determine them       
            if loop_index > split_idx:
                if example in target_clip_examples:
                    train_examples.append([example, 7, target_clip_examples[example]]) # each item is [path_to_img, 7, clip labels]
                else:
                    train_examples.append([example, 7, ''])
            else:
                val_examples.append([example, 7, '']) # each pair is [path_to_img, 7]
    
    # Transforms
    normalize = T.Normalize([0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # ResNet18 - ImageNet Normalization

    train_transform = T.Compose([
        T.Resize(256),
        T.RandAugment(3, 15),
        T.CenterCrop(224),
        T.ToTensor(),
        normalize
    ])

    eval_transform = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        normalize
    ])

    # Dataloaders
    train_loader = DataLoader(PACSDatasetClipDisentangle(train_examples, train_transform), batch_size=opt['batch_size'], num_workers=opt['num_workers'], shuffle=True)
    val_loader = DataLoader(PACSDatasetClipDisentangle(val_examples, eval_transform), batch_size=opt['batch_size'], num_workers=opt['num_workers'], shuffle=False)
    test_loader = DataLoader(PACSDatasetClipDisentangle(test_examples, eval_transform), batch_size=opt['batch_size'], num_workers=opt['num_workers'], shuffle=False)

    return train_loader, val_loader, test_loader


def build_splits_clip_finetuner(batch_size, num_workers):
    text_files = ['groupe1AML.txt', 'groupe1DAAI.txt', 'groupe2AML.txt', 'groupe2DAAI.txt', 'groupe3AML.txt',
            'groupe3DAAI.txt', 'groupe5AML.txt', 'groupe6AML.txt', 'group_299837_300451_300708.txt']
    
    clip_examples = [] # it will be a list of images with their descriptions
    
    for file_name in text_files:
        with open(f'data/PACS/clip_labels/{file_name}') as f:
            line = f.read()

        line = line[1:-1:1]
        items = line.split('},')
        for i in range(len(items)-1):
            items[i] = items[i] + '}'

        for i in items: 
            item_dict = eval(i)
            address = 'data/PACS/kfold/' + item_dict[f'image_name']
            descriptions = ', '.join(item_dict['descriptions'])
            clip_examples.append([address, descriptions])
    
    # Transforms
    normalize = T.Normalize([0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # ResNet18 - ImageNet Normalization

    train_transform = T.Compose([
        T.Resize(256),
        T.RandAugment(3, 15),
        T.CenterCrop(224),
        T.ToTensor(),
        normalize
    ])

    # Dataloaders
    train_loader = DataLoader(PACSDatasetDescriptions(clip_examples, train_transform), batch_size=batch_size, num_workers=num_workers, shuffle=True)
    
    return train_loader
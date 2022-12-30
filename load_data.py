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

DOMAINS = {
    'art_painting': 0,
    'cartoon': 1,
    'photo': 2,
    'sketch': 3,
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
    def __init__(self, src_examples, trg_examples, transform):
        self.src_examples = src_examples
        self.trg_examples = trg_examples
        self.transform = transform

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, src_index, trg_index):
        src_image_path, src_label = self.src_examples[src_index]
        trg_image_path, trg_label = self.trg_examples[trg_index]
        xsrc = self.transform(Image.open(src_image_path).convert('RGB'))
        xtrg = self.transform(Image.open(trg_image_path).convert('RGB'))
        return xsrc, src_label, xtrg


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


# def read_lines2(data_path, domain_name):
#     examples = {}
#     with open(f'{data_path}/{domain_name}.txt') as f:
#         lines = f.readlines()
#     for line in lines:
#         line = line.strip().split()[0].split('/')
#         category_name = line[3]
#         domain_idx = DOMAINS[domain_name]
#         category_idx = CATEGORIES[category_name]
#         image_name = line[4]
#         image_path = f'{data_path}/kfold/{domain_name}/{category_name}/{image_name}'
#         if (domain_idx, category_idx) not in examples.keys():
#             examples[(domain_idx, category_idx)] = [image_path]
#         else:
#             examples[(domain_idx, category_idx)].append(image_path)
#     return examples
    



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
    # print("vaaaaaaaaaaaaaaaaaaaaaaaal", val_examples)
    
    for category_idx, examples_list in target_examples.items():
        for example in examples_list:
            test_examples.append([example, category_idx]) # each pair is [path_to_img, class_label]
    # print("teeeeeeeeeeeeeeeeeeeeeest", test_examples)
    
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
    raise NotImplementedError('[TODO] Implement build_splits_domain_disentangle') #TODO




def build_splits_clip_disentangle(opt):
    raise NotImplementedError('[TODO] Implement build_splits_clip_disentangle') #TODO







################### print/test area ####################


# rltest = read_lines("./data/PACS", "cartoon")
# print(rltest)

# rl2test = read_lines2("./data/PACS", "cartoon")
# print(rl2test)

opt = {
    'data_path': "./data/PACS",
    'target_domain': "cartoon"
}

# bsddtest = build_splits_domain_disentangle(opt)
# print(bsddtest)

# build_splits_baseline(opt)
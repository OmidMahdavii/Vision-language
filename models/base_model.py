import torch
import torch.nn as nn
from torchvision.models import resnet18


class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        self.resnet18 = resnet18(pretrained=True)
    
    def forward(self, x):
        x = self.resnet18.conv1(x)
        x = self.resnet18.bn1(x)
        x = self.resnet18.relu(x)
        x = self.resnet18.maxpool(x)
        x = self.resnet18.layer1(x)
        x = self.resnet18.layer2(x)
        x = self.resnet18.layer3(x)
        x = self.resnet18.layer4(x)
        x = self.resnet18.avgpool(x)
        # return x.squeeze()
        temp = x.squeeze()
        return temp.unsqueeze(0) if len(temp.size()) < 2 else temp


class BaselineModel(nn.Module):
    def __init__(self):
        super(BaselineModel, self).__init__()
        self.feature_extractor = FeatureExtractor()
        self.category_encoder = nn.Sequential(
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU()
        )
        self.classifier = nn.Linear(512, 7)
    
    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.category_encoder(x)
        x = self.classifier(x)
        return x


class DomainDisentangleModel(nn.Module):
    def __init__(self):
        super(DomainDisentangleModel, self).__init__()
        self.feature_extractor = FeatureExtractor()

        self.domain_encoder = nn.Sequential(
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU()
        )

        self.category_encoder = nn.Sequential(
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU()
        )

        self.category_classifier = nn.Linear(512, 7)
        self.domain_classifier = nn.Linear(512, 2)

        self.reconstructor = nn.Linear(1024, 512)


    def forward(self, x, status='cc', minimizing=True):
        # status: cc: category classifier, dc: domain classifier, rc: reconstructor
        x = self.feature_extractor(x)
        category_features = self.category_encoder(x)
        domain_features = self.domain_encoder(x)

        if status == 'cc':
            y = self.category_classifier(category_features) if minimizing else self.category_classifier(domain_features)
        
        elif status == 'dc':
            y = self.domain_classifier(domain_features) if minimizing else self.domain_classifier(category_features)

        elif status == 'rc':
            # we should return the features too to be able to compute the loss
            y = (self.reconstructor(torch.cat((category_features, domain_features), dim=1)), x)

        return y


class CLIPDisentangleModel(nn.Module):
    def __init__(self):
        super(CLIPDisentangleModel, self).__init__()
        self.feature_extractor = FeatureExtractor()

        self.domain_encoder = nn.Sequential(
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU()
        )

        self.category_encoder = nn.Sequential(
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU()
        )

        self.category_classifier = nn.Linear(512, 7)
        self.domain_classifier = nn.Linear(512, 2)

        self.reconstructor = nn.Linear(1024, 512)


    def forward(self, x, status='cc', minimizing=True):
        # status: cc: category classifier, dc: domain classifier, rc: reconstructor, df: domain features
        x = self.feature_extractor(x)
        category_features = self.category_encoder(x)
        domain_features = self.domain_encoder(x)

        if status == 'cc':
            y = self.category_classifier(category_features) if minimizing else self.category_classifier(domain_features)
        
        elif status == 'dc':
            y = self.domain_classifier(domain_features) if minimizing else self.domain_classifier(category_features)

        elif status == 'rc':
            # we should return the features too to be able to compute the loss
            y = (self.reconstructor(torch.cat((category_features, domain_features), dim=1)), x)
        elif status == 'df':
            y = domain_features

        return y

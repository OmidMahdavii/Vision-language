import torch
import torch.nn.functional as F
from models.base_model import DomainDisentangleModel

class HLoss(torch.nn.Module):
    def __init__(self):
        super(HLoss, self).__init__()

    def forward(self, x):
        b = F.softmax(x, dim=1) * F.log_softmax(x, dim=1)
        b = -1.0 * b.sum()
        return b


class DomainDisentangleExperiment: # See point 2. of the project
    
    def __init__(self, opt):
        # Utils
        self.opt = opt
        self.device = torch.device('cpu' if opt['cpu'] else 'cuda:0')

        # Setup model
        self.model = DomainDisentangleModel()
        self.model.train()
        self.model.to(self.device)
        for param in self.model.parameters():
            param.requires_grad = True

        # Setup optimization procedure
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=opt['lr'])
        self.criterion1 = torch.nn.CrossEntropyLoss()
        self.criterion2 = HLoss()

    def save_checkpoint(self, path, iteration, best_accuracy, total_train_loss):
        checkpoint = {}

        checkpoint['iteration'] = iteration
        checkpoint['best_accuracy'] = best_accuracy
        checkpoint['total_train_loss'] = total_train_loss

        checkpoint['model'] = self.model.state_dict()
        checkpoint['optimizer'] = self.optimizer.state_dict()

        torch.save(checkpoint, path)

    def load_checkpoint(self, path):
        checkpoint = torch.load(path)

        iteration = checkpoint['iteration']
        best_accuracy = checkpoint['best_accuracy']
        total_train_loss = checkpoint['total_train_loss']

        self.model.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])

        return iteration, best_accuracy, total_train_loss

    def train_iteration(self, data):
        # source_images, category_labels, target_images = data
        # source_images = source_images.to(self.device)
        # category_labels = category_labels.to(self.device)
        # target_images = target_images.to(self.device)

        # self.optimizer.zero_grad()

        # logits = self.model(source_images, status='cc', maximizing=True)
        # loss = self.criterion1(logits, category_labels)
        # loss.backward()

        # logits = self.model(source_images, status='cc', maximizing=False)
        # loss = self.criterion2(logits, category_labels)
        # loss.backward()

        # logits = self.model(source_images, status='dc', maximizing=True)
        # loss = self.criterion1(logits, 0)
        # loss.backward()

        # logits = self.model(source_images, status='dc', maximizing=False)
        # loss = self.criterion2(logits, 0)
        # loss.backward()

        # logits = self.model(source_images, status='rc')
        # loss = self.criterion1(*logits)
        # loss.backward()

        x, y = data
        x = x.to(self.device)
        y = y.to(self.device)

        self.optimizer.zero_grad()

        if y != 8:
            # image is related to the source domain
            logits = self.model(x, status='cc', maximizing=True)
            loss = self.criterion1(logits, y)
            loss.backward()

            logits = self.model(x, status='cc', maximizing=False)
            loss = self.criterion2(logits, y)
            loss.backward()

            logits = self.model(x, status='dc', maximizing=True)
            loss = self.criterion1(logits, 0)
            loss.backward()

            logits = self.model(x, status='dc', maximizing=False)
            loss = self.criterion2(logits, 0)
            loss.backward()

            logits = self.model(x, status='rc')
            loss = self.criterion1(*logits)
            loss.backward()

        else:
            # image is related to the target domain
            logits = self.model(x, status='dc', maximizing=True)
            loss = self.criterion1(logits, 1)
            loss.backward()

            logits = self.model(x, status='dc', maximizing=False)
            loss = self.criterion2(logits, 1)
            loss.backward()

            logits = self.model(x, status='rc')
            loss = self.criterion1(*logits)
            loss.backward()

        self.optimizer.step()
        
        return loss.item()

    def validate(self, loader):
        self.model.eval()
        accuracy = 0
        count = 0
        loss = 0
        with torch.no_grad():
            for x, y in loader:
                if y != 8:
                    # validating only source domain
                    x = x.to(self.device)
                    y = y.to(self.device)

                    logits = self.model(x)
                    loss += self.criterion1(logits, y)
                    pred = torch.argmax(logits, dim=-1)

                    accuracy += (pred == y).sum().item()
                    count += x.size(0)

        mean_accuracy = accuracy / count
        mean_loss = loss / count
        self.model.train()
        return mean_accuracy, mean_loss
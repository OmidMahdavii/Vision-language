import torch
import torch.nn.functional as F
from models.base_model import DomainDisentangleModel

class HLoss(torch.nn.Module):
    def __init__(self):
        super(HLoss, self).__init__()

    def forward(self, x):
        b = F.softmax(x, dim=1) * F.log_softmax(x, dim=1)
        return b.sum()


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
        self.cross_entropy_criterion = torch.nn.CrossEntropyLoss()
        self.entropy_criterion = HLoss()
        # self.reconstruction_criterion = torch.nn.MSELoss()
        self.reconstruction_criterion = torch.nn.L1Loss()

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
        x, y = data
        # source_indices = []
        # target_indices = []

        # separating source and target images
        # for i in range(len(y)):
        #     source_indices.append(i) if y[i] != 7 else target_indices.append(i)

        source_indices = [i for i in range(len(y)) if y[i] != 7]
        
        domain_labels = torch.ones(y.size(), dtype=torch.long)
        domain_labels[source_indices] = 0.0

        x = x.to(self.device)
        domain_labels = domain_labels.to(self.device)
        
        # only source samples are going through category classifier
        source_samples = x[source_indices].to(self.device)
        category_labels = y[source_indices].to(self.device)

        self.optimizer.zero_grad()

        logits = self.model(x, status='dc', maximizing=True)
        loss1 = self.cross_entropy_criterion(logits, domain_labels)
        loss1.backward()

        logits = self.model(x, status='dc', maximizing=False)
        loss2 = self.entropy_criterion(logits)
        loss2.backward()

        logits = self.model(x, status='rc')
        loss3 = self.reconstruction_criterion(*logits)
        loss3.backward()

        # the batch should have more than 1 source sample to be able to classify the category
        if len(source_indices) > 1:
            ### source samples
            logits = self.model(source_samples, status='cc', maximizing=True)
            loss4 = self.cross_entropy_criterion(logits, category_labels)
            loss4.backward()

            logits = self.model(source_samples, status='cc', maximizing=False)
            loss5 = self.entropy_criterion(logits)
            loss5.backward()

            loss = loss1 + loss2 + loss3 + loss4 + loss5
        else:
            loss = loss1 + loss2 + loss3
        
        self.optimizer.step()
    
        return loss.item()

        


















            
        # source_samples = x[source_indices].to(self.device)
        # target_samples = x[target_indices].to(self.device)
        
        # source_labels = y[source_indices].to(self.device)
        # target_labels_size = y[target_indices].size()
        
        # # source samples are labeled 0 and target samples are labeled 1
        # source_domain_labels = torch.zeros(source_labels.size(), dtype=torch.long).to(self.device)
        # target_domain_labels = torch.ones(target_labels_size, dtype=torch.long).to(self.device)

        # self.optimizer.zero_grad()

        # ### source samples
        # logits = self.model(source_samples, status='cc', maximizing=True)
        # loss = self.cross_entropy_criterion(logits, source_labels)
        # loss.backward()

        # logits = self.model(source_samples, status='cc', maximizing=False)
        # loss = self.entropy_criterion(logits)
        # loss.backward()

        # logits = self.model(source_samples, status='dc', maximizing=True)
        # loss = self.cross_entropy_criterion(logits, source_domain_labels)
        # loss.backward()

        # logits = self.model(source_samples, status='dc', maximizing=False)
        # loss = self.entropy_criterion(logits)
        # loss.backward()

        # logits = self.model(source_samples, status='rc')
        # loss = self.reconstruction_criterion(*logits)
        # loss.backward()


        # # target samples
        # logits = self.model(target_samples, status='dc', maximizing=True)
        # loss = self.cross_entropy_criterion(logits, target_domain_labels)
        # loss.backward()

        # logits = self.model(target_samples, status='dc', maximizing=False)
        # loss = self.entropy_criterion(logits)
        # loss.backward()

        # logits = self.model(target_samples, status='rc')
        # loss = self.reconstruction_criterion(*logits)
        # loss.backward()

        # self.optimizer.step()
        
        # return loss.item()

    def validate(self, loader):
        self.model.eval()
        accuracy = 0
        count = 0
        loss = 0
        with torch.no_grad():
            for x, y in loader:
                x = x.to(self.device)
                y = y.to(self.device)

                logits = self.model(x)
                loss += self.cross_entropy_criterion(logits, y)
                pred = torch.argmax(logits, dim=-1)

                accuracy += (pred == y).sum().item()
                count += x.size(0)

        mean_accuracy = accuracy / count
        mean_loss = loss / count
        self.model.train()
        return mean_accuracy, mean_loss
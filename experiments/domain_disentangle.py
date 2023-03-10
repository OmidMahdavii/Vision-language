import torch
import torch.nn.functional as F
from models.base_model import DomainDisentangleModel

class HLoss(torch.nn.Module):
    def __init__(self):
        super(HLoss, self).__init__()

    def forward(self, x):
        b = F.softmax(x, dim=1) * F.log_softmax(x, dim=1)
        return b.sum() / x.size(0)


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
        self.L2_criterion = torch.nn.MSELoss()
        # self.KLDiv_criterion = torch.nn.KLDivLoss()

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

        source_indices = [i for i in range(len(y)) if y[i] != 7]
        
        domain_labels = torch.ones(y.size(), dtype=torch.long)
        domain_labels[source_indices] = 0.0

        x = x.to(self.device)
        domain_labels = domain_labels.to(self.device)
        
        # only source samples are going through category classifier
        source_samples = x[source_indices].to(self.device)
        category_labels = y[source_indices].to(self.device)

        self.optimizer.zero_grad()
        loss = 0

        # the batch should have more than 1 source sample to be able to classify the category
        if len(source_indices) > 1:
            ### source samples
            logits = self.model(source_samples, status='cc', minimizing=True)
            temp_loss = 4 * self.cross_entropy_criterion(logits, category_labels)
            temp_loss.backward()
            loss += temp_loss

            logits = self.model(source_samples, status='cc', minimizing=False)
            temp_loss = self.entropy_criterion(logits)
            temp_loss.backward()
            loss += temp_loss

        logits = self.model(x, status='dc', minimizing=True)
        temp_loss = self.cross_entropy_criterion(logits, domain_labels)
        temp_loss.backward()
        loss += temp_loss

        logits = self.model(x, status='dc', minimizing=False)
        temp_loss = self.entropy_criterion(logits)
        temp_loss.backward()
        loss += temp_loss

        logits = self.model(x, status='rc')
        temp_loss = self.L2_criterion(*logits)
        # temp_loss2 = self.KLDiv_criterion(*logits) / 4
        # temp_loss = temp_loss1 + temp_loss2
        temp_loss.backward()
        loss += temp_loss

        self.optimizer.step()
    
        return loss.item()

    def validate(self, loader):
        self.model.eval()
        accuracy = 0
        count = 0
        loss = 0
        with torch.no_grad():
            for x, y in loader:
                source_indices = [i for i in range(len(y)) if y[i] != 7]
                domain_labels = torch.ones(y.size(), dtype=torch.long)
                domain_labels[source_indices] = 0.0

                x = x.to(self.device)
                domain_labels = domain_labels.to(self.device)
                
                # if we are in the test stage, the loss corresponding to the domain classifier should not be considered
                if self.opt["test"]:
                    y = y.to(self.device)

                    logits = self.model(x, status='cc', minimizing=True)
                    loss += self.cross_entropy_criterion(logits, y)
                    pred = torch.argmax(logits, dim=-1)

                    accuracy += (pred == y).sum().item()
                    count += x.size(0)

                else:                            
                    # only source samples are going through category classifier
                    source_samples = x[source_indices].to(self.device)
                    category_labels = y[source_indices].to(self.device)
                                        
                    # the batch should have more than 1 source sample to be able to classify the category
                    if len(source_indices) > 1:
                        logits = self.model(source_samples, status='cc', minimizing=True)
                        loss += self.cross_entropy_criterion(logits, category_labels)
                        pred = torch.argmax(logits, dim=-1)
                        accuracy += (pred == category_labels).sum().item()
                        count += source_samples.size(0)
                    
                    logits = self.model(x, status='dc', minimizing=True)
                    loss += self.cross_entropy_criterion(logits, domain_labels)
                    pred = torch.argmax(logits, dim=-1)
                    accuracy += (pred == domain_labels).sum().item()
                    count += x.size(0)
                

        mean_accuracy = accuracy / count
        mean_loss = loss / count
        self.model.train()
        return mean_accuracy, mean_loss
import torch
import clip
from load_data import build_splits_clip_finetuner

device = "cuda:0" if torch.cuda.is_available() else "cpu" # If using GPU then use mixed precision training.

# Load CLIP model and freeze it
model, _ = clip.load('ViT-B/32', device='cpu') # load it first to CPU to ensure you're using fp32 precision.
model = model.to(device)

# unfreeze all the network
# for param in model.parameters():
#     param.requires_grad = True

# unfreeze just the last layer
for name, param in model.named_parameters():
    param.requiresGrad = False if 'ln_final' not in name else True

train_loader = build_splits_clip_finetuner(batch_size=32, num_workers=2)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=5e-5,betas=(0.9,0.98),eps=1e-6,weight_decay=0.2) #Params used from paper, the lr is smaller, more safe for fine tuning to new dataset

best_loss = None

for iteration in range(200):
  total_loss = 0
  for data in train_loader :
    optimizer.zero_grad()

    images, descriptions = data 

    images = images.to(device)
    tokenized_text = clip.tokenize(descriptions).to(device)
    
    logits_per_image, logits_per_text = model(images, tokenized_text)

    ground_truth = torch.arange(len(images), dtype=torch.long, device=device)

    loss = (criterion(logits_per_image, ground_truth) + criterion(logits_per_text, ground_truth)) / 2
    loss.backward()
    optimizer.step()

    total_loss += loss
  
  if best_loss is None or total_loss / len(train_loader) <= best_loss:
    best_loss = total_loss / len(train_loader)
    print(f'[TRAIN - {iteration}] Loss: {best_loss}')
    final_model = {}
    final_model['iteration'] = iteration
    final_model['model'] = model.state_dict()
    final_model['optimizer'] = optimizer.state_dict()
    torch.save(final_model, 'finetuned_clip/last_layer.pt')


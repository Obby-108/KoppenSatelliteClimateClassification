import torch

# Function for checkpointing model during training
def save_checkpoint(state, filename="models/koppen_checkpoint.pth"):
    print(f"=> Saving best model to {filename}")
    torch.save(state, filename)

# Function for loading model checkpoint
def load_checkpoint(checkpoint_path, model, optimizer, scheduler):
    print(f"=> Loading checkpoint '{checkpoint_path}'")
    checkpoint = torch.load(checkpoint_path)

    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    scheduler.load_state_dict(checkpoint['scheduler'])
    start_epoch = checkpoint['epoch']
    best_accuracy = checkpoint['best_acc']

    return model, optimizer, scheduler, start_epoch, best_accuracy

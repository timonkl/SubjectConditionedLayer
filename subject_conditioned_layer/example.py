import torch
from torch import nn
from subject_conditioned_layer import SubjectModelWrapper


# Example model
class BaseModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Linear(32, 10),
            nn.BatchNorm1d(10),
            nn.ReLU(),
            nn.Flatten(),
        )
        self.fc = nn.Linear(10, 10)

    def forward(self, x):
        out = self.features(x)
        out = self.fc(out)
        return out


# Instantiate original model
base_model = BaseModel()
num_subjects = 5

# Wrap it
model = SubjectModelWrapper(base_model=base_model, num_subjects=num_subjects)

# Print to confirm that linear layers have been replaced
print(model)

# batch_size x input_dim
x = torch.randn(4, 32)

# batch_size
subject_id = torch.randint(0, num_subjects, (4,))

# forward pass
out = model(x, subject_id)
print("Output shape:", out.shape)

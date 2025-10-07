import torch
import torch.nn as nn
from torch.nn import functional as F


class SubjectConditionedLayer(nn.Module):
    def __init__(self, in_features, out_features, rank=4, alpha=1.0, num_adapters=10):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.rank = rank
        self.alpha = alpha
        self.num_adapters = num_adapters

        # LoRA adapters
        self.lora_A = nn.ModuleList(
            [nn.Linear(in_features, rank, bias=False) for _ in range(num_adapters)]
        )
        self.lora_B = nn.ModuleList(
            [nn.Linear(rank, out_features, bias=False) for _ in range(num_adapters)]
        )

        # Init B with zeros
        for a in self.lora_A:
            torch.nn.init.normal_(a.weight, mean=0.0, std=0.02)
        for b in self.lora_B:
            nn.init.zeros_(b.weight)
            # torch.nn.init.normal_(b.weight, mean=0.0, std=0.02)

    def forward(self, x, subject_id=None):
        """
        x: batch size, sequence length, token dim
        subject_id: (batch_size,) ints in [0, ..., num_subjects]
        """

        # W_general
        out = self.linear(x)

        # W_subject
        if subject_id is not None:
            # Apply correct adapter to each Subject
            lora_out = torch.zeros_like(out)
            for i in range(self.num_adapters):
                mask = subject_id == i
                if mask.any():
                    lora_A_i = self.lora_A[i](x[mask])
                    lora_B_i = self.lora_B[i](lora_A_i)
                    lora_out[mask] = self.alpha / self.rank * lora_B_i
        
            out = out + lora_out

        else:
            raise Warning("Subject ID is not set. Subject-conditioned layer will not be applied.")

        return out


class SubjectModelWrapper(nn.Module):
    def __init__(self, base_model, num_subjects):
        super().__init__()
        self.base_model = base_model
        self.replace_linear_with_subjectlinear(
            model=self.base_model, num_subjects=num_subjects
        )

    def forward(self, x, subject_id):
        return self._forward_module(self.base_model, x, subject_id)

    def _forward_module(self, module, x, subject_id):
        # If module is a SubjectConditionedLayer -> pass x and subject_id
        if isinstance(module, SubjectConditionedLayer):
            return module(x, subject_id)

        # if the module has children recurse
        elif len(list(module.children())) > 0:
            for child in module.children():
                x = self._forward_module(child, x, subject_id)
            return x

        # leaf node (e.g., Norm, activation, etc.)
        else:
            return module(x)

    @staticmethod
    def replace_linear_with_subjectlinear(model, num_subjects, rank=4, alpha=1.0):
        """
        Recursively replaces all nn.Linear layers in the model with SubjectConditionedLayer
        """
        for name, module in model.named_children():
            if isinstance(module, nn.Linear):
                setattr(
                    model,
                    name,
                    SubjectConditionedLayer(
                        in_features=module.in_features,
                        out_features=module.out_features,
                        num_adapters=num_subjects,
                        rank=rank,
                        alpha=alpha,
                    ),
                )
            else:
                SubjectModelWrapper.replace_linear_with_subjectlinear(
                    module, num_subjects
                )



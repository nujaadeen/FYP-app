import torch
from torch import nn

# transformers
from transformers import (
    AutoModel
)

# Model Acceleration and Compression
from accelerate import Accelerator
from peft import(
    get_peft_config,
    get_peft_model
)

LORA_RANK = 32

class ParallelBERT(nn.Module):
    def __init__(
        self,
        dropout_rate,
        protein_model_name,
        compress_model: bool,
        dims: list[int],
        act_fn: str = "relu",
    ):
        super(ParallelBERT, self).__init__()
        self.dropout_rate = dropout_rate
        
        if act_fn == "relu":
            self.act_fn = nn.ReLU()
        elif act_fn == "tanh":
            self.act_fn = nn.Tanh()
        else:
            raise ValueError("Activation function not supported")

        # ESM Encoder for ab and ag
        if compress_model:
            self.ab_ESM = AutoModel.from_pretrained(
                protein_model_name
            )

            self.ag_ESM = AutoModel.from_pretrained(
                protein_model_name
            )


            # Convert the model into a PeftModel
            config = {
                "peft_type": "LORA",
                "task_type": "FEATURE_EXTRACTION",
                "inference_mode": False,
                "r": LORA_RANK,
                "target_modules": [
                    "query",
                    "key",
                    "value",
                    "output.dense",
                    "pooler.dense"
                ],
                "lora_alpha": 32,
                "lora_dropout": 0.05,
                "fan_in_fan_out": False,
                "bias": "none",
            }

            peft_config = get_peft_config(config)
            self.ab_ESM = get_peft_model(self.ab_ESM, peft_config)
            self.ag_ESM = get_peft_model(self.ag_ESM, peft_config)
            
            # Use the accelerator
            self.ab_ESM = Accelerator().prepare(self.ab_ESM)
            self.ag_ESM = Accelerator().prepare(self.ag_ESM)
        else:
            self.ab_ESM = AutoModel.from_pretrained(protein_model_name)
            self.ag_ESM = AutoModel.from_pretrained(protein_model_name)


        # Output Regressors
        self.bn_feature = nn.BatchNorm1d(dims[0])
        self.mlp = self.create_mlp(dims)
        self.out_reg = nn.Linear(dims[-1], 1)
        self.out_cls = nn.Linear(dims[-1], 1)

    def create_layer(self, in_dim: int, out_dim: int):
        return nn.Sequential(
            nn.Dropout(p=self.dropout_rate),
            nn.Linear(in_dim, out_dim),
            self.act_fn,
            nn.BatchNorm1d(out_dim),
        )

    def create_mlp(self, dims: list[int]):
        return nn.Sequential(
            *[self.create_layer(dims[i], dims[i + 1]) for i in range(len(dims) - 1)]
        )

    def forward(self, ab_input, ag_input):
        # Get ProtBERT embeddings
        ab_feature = self.ab_ESM(**ab_input)["pooler_output"]
        ag_feature = self.ag_ESM(**ag_input)["pooler_output"]

        # Final Regressor
        x = torch.cat((ab_feature, ag_feature), dim=1)
        x = self.bn_feature(x)
        x = self.mlp(x)
        x_reg = self.out_reg(x)
        x_cls = torch.sigmoid(self.out_cls(x))

        return x_reg, x_cls

    def freeze_layers(self, all: bool = True, n_layers: int = 1):
        if all:
            ab_modules = self.ab_ESM.modules()
            ag_modules = self.ag_ESM.modules()
        elif n_layers == 0:
            # Freeze none
            pass
        elif n_layers == 1:
            # Freeze only the embedding layer
            ab_modules = self.ab_ESM.encoder.layer[:n_layers]
            ag_modules = self.ag_ESM.encoder.layer[:n_layers]
            print("Freezing only the embedding layer")
        else:
            ab_modules = [
                self.ab_ESM.embeddings,
                *self.ab_ESM.encoder.layer[: (n_layers - 1)],
            ]
            ag_modules = [
                self.ag_ESM.embeddings,
                *self.ag_ESM.encoder.layer[: (n_layers - 1)],
            ]

        for module in ab_modules:
            for param in module.parameters():
                param.requires_grad = False

        for module in ag_modules:
            for param in module.parameters():
                param.requires_grad = False
    
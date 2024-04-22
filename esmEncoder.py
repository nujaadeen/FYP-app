import torch
from torch import nn

# transformers
from transformers import (
    AutoModel,
    AutoTokenizer,
)

device = torch.device('cpu')
PROTEIN_LM_NAME = "facebook/esm2_t6_8M_UR50D"

class ESMEncoder(nn.Module):
    def __init__(self, protein_model_name: str = PROTEIN_LM_NAME):
        super(ESMEncoder, self).__init__()

        self.emb_model = AutoModel.from_pretrained(protein_model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(PROTEIN_LM_NAME, do_lower_case=False)

    def forward(self, seq):
        # Tokenize
        token = self.tokenizer(
            seq,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
            max_length=1500,
        )

        # Put tensors to device
        for key, tensor in token.items():
            token[key] = tensor.to(device)

        # Get the embeddings
        embedding = self.emb_model(**token)['last_hidden_state'][0]

        # Detach embedding
        embedding = embedding.detach()

        # Delete intermediate tensors to release GPU memory
        del token

        return embedding
"""DeepDTA branches with a standalone protein representation for Phase 3."""

import torch
from torch import nn

from .config import Config


class ProteinEncoder(nn.Module):
    """Encode protein tokens into a pooled vector for regression or DANN."""

    def __init__(self, vocab_size: int, config: Config):
        """Build the protein embedding and three valid convolutions.

        Args:
            vocab_size: Number of tokens including zero padding.
            config: Embedding width, filter widths and protein kernel size.
        """
        super().__init__()
        self.out_dim = config.filters[-1]
        self.embedding = nn.Embedding(vocab_size, config.embedding_dim, padding_idx=0)
        channels = (config.embedding_dim, *config.filters)
        self.convs = nn.Sequential(*[
            layer for incoming, outgoing in zip(channels, channels[1:])
            for layer in (nn.Conv1d(incoming, outgoing, config.protein_kernel), nn.ReLU())
        ])

    def forward(self, tokens):
        """Return pooled protein features.

        Args:
            tokens: Long tensor with shape (batch, protein length).

        Returns:
            Tensor of shape (batch, final filter count).
        """
        return self.convs(self.embedding(tokens).transpose(1, 2)).amax(dim=2)


class EsmProteinEncoder(nn.Module):
    """Project a frozen ESM-2 sequence embedding to the CNN branch's output width.

    The protein language model is not part of this module: its embeddings are
    precomputed once by src/esm_embed.py and never updated. Projecting to
    config.filters[-1], the default, keeps the fusion width and the whole
    regression head identical to the CNN model, so the protein representation is
    the only thing that differs between the two arms. That default is also a 13x
    compression of a 1280-wide embedding, so config.esm_proj_dim can widen it;
    the regression head's input then grows with it.
    """

    def __init__(self, config: Config):
        """Build the dropout and linear projection over the frozen embedding.

        Args:
            config: ESM width, projection width (0 means config.filters[-1]) and
                dropout probability.
        """
        super().__init__()
        self.out_dim = config.esm_proj_dim or config.filters[-1]
        self.layers = nn.Sequential(
            nn.Dropout(config.dropout),
            nn.Linear(config.esm_dim, self.out_dim),
            nn.ReLU(),
        )

    def forward(self, embeddings):
        """Return projected protein features.

        Args:
            embeddings: Float tensor of shape (batch, esm_dim).

        Returns:
            Tensor of shape (batch, self.out_dim).
        """
        return self.layers(embeddings)


class DrugEncoder(nn.Module):
    """Encode SMILES tokens into a pooled drug representation."""

    def __init__(self, vocab_size: int, config: Config):
        """Build the SMILES embedding and three valid convolutions.

        Args:
            vocab_size: Number of tokens including zero padding.
            config: Embedding width, filter widths and SMILES kernel size.
        """
        super().__init__()
        self.out_dim = config.filters[-1]
        self.embedding = nn.Embedding(vocab_size, config.embedding_dim, padding_idx=0)
        channels = (config.embedding_dim, *config.filters)
        self.convs = nn.Sequential(*[
            layer for incoming, outgoing in zip(channels, channels[1:])
            for layer in (nn.Conv1d(incoming, outgoing, config.drug_kernel), nn.ReLU())
        ])

    def forward(self, tokens):
        """Return pooled drug features.

        Args:
            tokens: Long tensor with shape (batch, SMILES length).

        Returns:
            Tensor of shape (batch, final filter count).
        """
        return self.convs(self.embedding(tokens).transpose(1, 2)).amax(dim=2)


class DeepDTA(nn.Module):
    """Two CNN branches followed by the DeepDTA regression head."""

    def __init__(self, drug_vocab_size: int, protein_vocab_size: int, config: Config):
        """Build independently accessible encoders and the regression head.

        Args:
            drug_vocab_size: SMILES vocabulary size including padding.
            protein_vocab_size: Protein vocabulary size including padding; ignored
                when config.protein_encoder is "esm".
            config: CNN, protein encoder and fully connected hyperparameters.
        """
        super().__init__()
        self.drug_encoder = DrugEncoder(drug_vocab_size, config)
        self.protein_encoder = (EsmProteinEncoder(config) if config.protein_encoder == "esm"
                                else ProteinEncoder(protein_vocab_size, config))
        layers = []
        incoming = self.drug_encoder.out_dim + self.protein_encoder.out_dim
        for index, outgoing in enumerate(config.fc_dims):
            layers.extend((nn.Linear(incoming, outgoing), nn.ReLU()))
            if index < 2:
                layers.append(nn.Dropout(config.dropout))
            incoming = outgoing
        self.regressor = nn.Sequential(*layers, nn.Linear(incoming, 1))

    def forward(self, drug_tokens, prot_tokens):
        """Predict affinity on the original processed score scale.

        Args:
            drug_tokens: Long tensor of shape (batch, SMILES length).
            prot_tokens: Long tensor of shape (batch, protein length) for the CNN
                encoder, or float ESM-2 embeddings of shape (batch, esm_dim).

        Returns:
            Score tensor of shape (batch,).
        """
        features = torch.cat((self.drug_encoder(drug_tokens), self.protein_encoder(prot_tokens)), dim=1)
        return self.regressor(features).squeeze(-1)

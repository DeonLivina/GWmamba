import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "common"))

import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import WandbLogger


# =====================================================================
# 1. OFFICIAL LeJEPA SIGReg IMPLEMENTATION
# =====================================================================
def official_sigreg(x, global_step, num_slices=256):
    """
    Official Sketched Isotropic Gaussian Regularization (SIGReg).
    Forces the latent space to form an isotropic Gaussian sphere using the 
    Epps-Pulley statistic on 1D random projections.
    """
    # slice sampling -- synced across devices --
    dev = dict(device=x.device)
    g = torch.Generator(**dev)
    # manual_seed ensures all GPUs in DDP generate the same random projection matrix A
    g.manual_seed(int(global_step)) 
    
    proj_shape = (x.size(1), num_slices)
    A = torch.randn(proj_shape, generator=g, **dev)
    A /= A.norm(p=2, dim=0)
    
    # -- Epps-Pulley stat. --
    # integration points
    t = torch.linspace(-5, 5, 17, **dev)
    
    # theoretical CF for N(0,1) and Gauss. window
    exp_f = torch.exp(-0.5 * t**2)
    
    # empirical CF -- gathered across devices --
    x_t = (x @ A).unsqueeze(2) * t # (N, M, T)
    ecf = (1j * x_t).exp().mean(0)
    
    # Safe handling for both single-GPU and distributed (DDP) setups
    if torch.distributed.is_initialized():
        torch.distributed.all_reduce(ecf, op=torch.distributed.ReduceOp.AVG)
        world_size = torch.distributed.get_world_size()
    else:
        world_size = 1
        
    # weighted L2 distance
    err = (ecf - exp_f).abs().square().mul(exp_f)
    N = x.size(0) * world_size
    
    # Integrate over the evaluation points t
    T_stat = torch.trapz(err, t, dim=1) * N
    
    # Return the mean statistic across all slices as the scalar loss
    return T_stat.mean()


# =====================================================================
# 2. CONV1D FUNNEL ENCODER
# =====================================================================
class ConvDownBlock(nn.Module):
    def __init__(self, d_model, kernel_size=4, stride=2):
        super().__init__()
        padding = (kernel_size - stride) // 2
        self.conv = nn.Conv1d(d_model, d_model, kernel_size=kernel_size, stride=stride, padding=padding)
        self.norm = nn.GroupNorm(num_groups=min(8, d_model), num_channels=d_model)
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))

class ConvResBlock(nn.Module):
    def __init__(self, d_model, kernel_size=5, dilation=1):
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2
        self.conv = nn.Conv1d(d_model, d_model, kernel_size=kernel_size, padding=padding, dilation=dilation)
        self.norm = nn.GroupNorm(num_groups=min(8, d_model), num_channels=d_model)
        self.act = nn.GELU()

    def forward(self, x):
        return x + self.act(self.norm(self.conv(x)))

class ConvFunnelEncoder(nn.Module):
    def __init__(self, d_model=32, n_down_layers=2, n_res_blocks=2, kernel_size=5):
        super().__init__()
        self.in_proj = nn.Conv1d(1, d_model, kernel_size=1)
        self.down_blocks = nn.ModuleList([ConvDownBlock(d_model) for _ in range(n_down_layers)])
        self.res_blocks = nn.ModuleList([
            ConvResBlock(d_model, kernel_size=kernel_size, dilation=2 ** i) for i in range(n_res_blocks)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):  
        x = x.permute(0, 2, 1)
        x = self.in_proj(x)
        for block in self.down_blocks:
            x = block(x)
        for block in self.res_blocks:
            x = block(x)
        x = x.permute(0, 2, 1)  
        return self.norm(x)


# =====================================================================
# 3. LeJEPA PRE-TRAINING MODEL
# =====================================================================
class LeJEPAPretrainModel(nn.Module):
    def __init__(self, d_model=32, mamba_layers=4, proj_dim=32, n_wit=4):
        super().__init__()
        self.d_model = d_model
        self.n_wit = n_wit

        # 1. Shared Encoders (Used for BOTH Context and Target generation)
        self.strain_enc = ConvFunnelEncoder(d_model=d_model)
        self.witness_enc = ConvFunnelEncoder(d_model=d_model) 

        # 2. Projection head
        self.strain_proj = nn.Linear(d_model, proj_dim)
        
        # 3. Sequence Fusion (Compress 1 Strain + N Witness into one dimension)
        fusion_dim = d_model + (n_wit * d_model)
        self.context_fusion = nn.Linear(fusion_dim, d_model)

        # 4. Mamba Causal Predictor
        self.mamba_blocks = nn.ModuleList([
            Mamba(d_model=d_model, d_state=16, d_conv=4, expand=2)
            for _ in range(mamba_layers)
        ])
        self.mamba_norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(mamba_layers)])
        
        # Predictor MLP maps Mamba's causal output to the Latent Target Space
        self.predictor = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, proj_dim)
        )

        # 5. Auxiliary Instrumental Head (For L1 vs H1 separation)
        self.instrumental_pool = nn.AdaptiveAvgPool1d(1)
        self.instrumental_classifier = nn.Linear(proj_dim, 2) 

    def forward(self, strain, witness):
        B, T_raw = strain.shape[0], strain.shape[1]
        
        # Encode Context (Past)
        h_strain_context = self.strain_enc(strain) 
        B, T_down, _ = h_strain_context.shape
        
        wit_flat = witness.permute(0, 2, 1).reshape(B * self.n_wit, T_raw, 1)
        h_wit_flat = self.witness_enc(wit_flat) 
        h_wit_context = h_wit_flat.view(B, self.n_wit, T_down, self.d_model)
        
        # Encode Target (Future) using the exact same shared strain encoder
        with torch.no_grad(): 
            h_strain_target = self.strain_enc(strain)
            target_tokens = self.strain_proj(h_strain_target) 

        # Shift the sequences for auto-regressive task (Predict t from t-1)
        context_strain = h_strain_context[:, :-1, :] 
        context_witness = h_wit_context[:, :, :-1, :]
        target_tokens = target_tokens[:, 1:, :]      

        # Fuse Contexts
        context_witness_flat = context_witness.permute(0, 2, 1, 3).reshape(B, T_down-1, self.n_wit * self.d_model)
        fused_context = torch.cat([context_strain, context_witness_flat], dim=-1)
        mamba_input = self.context_fusion(fused_context)

        # Predict Next Token
        for norm, block in zip(self.mamba_norms, self.mamba_blocks):
            mamba_input = mamba_input + block(norm(mamba_input))
            
        predicted_tokens = self.predictor(mamba_input) 

        # Instrumental Separation Head
        projected_context = self.strain_proj(context_strain)
        pooled_context = self.instrumental_pool(projected_context.permute(0, 2, 1)).squeeze(-1)
        detector_logits = self.instrumental_classifier(pooled_context)

        return predicted_tokens, target_tokens, detector_logits


# =====================================================================
# 4. LIGHTNING MODULE FOR PRE-TRAINING
# =====================================================================
class LeJEPALitModule(L.LightningModule):
    def __init__(self, model, lr=1e-3, lambda_reg=0.1, aux_weight=0.5):
        super().__init__()
        self.model = model
        self.lr = lr
        self.lambda_reg = lambda_reg
        self.aux_weight = aux_weight

    def _step(self, batch, stage):
        x, _, y_det = batch 
        strain, witness = x[:, :, 0:1], x[:, :, 1:]

        predicted_tokens, target_tokens, detector_logits = self.model(strain, witness)

        # 1. Prediction Loss 
        pred_loss = F.mse_loss(predicted_tokens, target_tokens)

        # 2. SIGReg Loss (Using official LeJEPA implementation)
        flat_predictions = predicted_tokens.reshape(-1, predicted_tokens.size(-1))
        # Pass the global_step to keep the random slice sampling synced
        reg_loss = official_sigreg(flat_predictions, self.global_step, num_slices=256)

        # 3. Auxiliary Instrumental Loss 
        aux_loss = F.cross_entropy(detector_logits, y_det)

        # Total Objective
        loss = pred_loss + (self.lambda_reg * reg_loss) + (self.aux_weight * aux_loss)

        self.log(f"{stage}_loss", loss, on_step=(stage == "train"), on_epoch=True, prog_bar=True)
        self.log(f"{stage}_pred_mse", pred_loss, on_epoch=True)
        self.log(f"{stage}_reg_loss", reg_loss, on_epoch=True)
        self.log(f"{stage}_aux_loss", aux_loss, on_epoch=True)

        return loss

    def training_step(self, batch, _):
        return self._step(batch, "train")

    def validation_step(self, batch, _):
        return self._step(batch, "val")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"}
        }


# =====================================================================
# 5. PRE-TRAINING MAIN SCRIPT
# =====================================================================
def main():
    from compact_loader import get_dataloaders
    train_loader, val_loader, _, meta = get_dataloaders(batch_size=128)

    sample_batch = next(iter(train_loader))
    x_sample, _, _ = sample_batch
    n_wit = x_sample.shape[-1] - 1

    model = LeJEPAPretrainModel(
        d_model=32,
        mamba_layers=4,
        proj_dim=32, 
        n_wit=n_wit,
    )
    
    lit_model = LeJEPALitModule(model, lr=1e-3, lambda_reg=0.1, aux_weight=0.5)

    ckpt_cb = ModelCheckpoint(
        dirpath="checkpoints_lejepa", monitor="val_loss", mode="min",
        save_top_k=1, filename="lejepa-{epoch:02d}-{val_loss:.4f}",
    )
    callbacks = [ckpt_cb, EarlyStopping(monitor="val_loss", mode="min", patience=12)]

    trainer = L.Trainer(
        max_epochs=100, 
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        precision="16-mixed", 
        accumulate_grad_batches=2, 
        callbacks=callbacks,
        logger=WandbLogger(project="ligo-lejepa-pretrain", name="lejepa_run_1"),
    )

    trainer.fit(lit_model, train_loader, val_loader)

if __name__ == "__main__":
    main()
import torch
import torch.nn as nn
import torch.nn.functional as F


def weighted_circular_phase(phase_a, phase_b, weight_a, weight_b, eps=1e-8):
    sine = weight_a * torch.sin(phase_a) + weight_b * torch.sin(phase_b)
    cosine = weight_a * torch.cos(phase_a) + weight_b * torch.cos(phase_b)
    near_zero = (sine.square() + cosine.square()) < eps
    fused = torch.atan2(sine, cosine)
    return torch.where(near_zero, torch.zeros_like(fused), fused)


class HFCEBoostBlock(nn.Module):
    def __init__(self, channels, dilation=2, reduction=8, residual_scale=0.7):
        super().__init__()
        self.residual_scale = residual_scale  
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.dw = nn.Conv2d(channels, channels, kernel_size=3, padding=dilation,
                            dilation=dilation, groups=channels, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False),
            nn.Sigmoid()
        )
        self.act = nn.ReLU(inplace=True)
    
    def forward(self, x, w_hf):
        identity = x
        out = self.conv1(x); out = self.bn1(out); out = self.act(out)
        out = self.dw(out); out = self.bn2(out); out = self.act(out)
        out = out * self.se(out) * (1.0 + w_hf)
        return identity + self.residual_scale * out

class HFCEBoost(nn.Module):
    def __init__(self, channels, n_blocks=2, dilation=2, reduction=8, residual_scale=0.7):
        super().__init__()
        self.w_gen = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=1),
            nn.Sigmoid()
        )
        self.blocks = nn.ModuleList([
            HFCEBoostBlock(channels, dilation=dilation, reduction=reduction, residual_scale=residual_scale)
            for _ in range(n_blocks)
        ])
    
    @torch.no_grad()
    def _stopgrad(self, x):
        return x.detach()
    
    def forward(self, F_h_star, A_comp, C):
        w_hf = A_comp.mean(dim=1, keepdim=True)
        w_hf = self.w_gen(w_hf)
        x = F_h_star - 0.3 * self._stopgrad(C)
        for blk in self.blocks:
            x = blk(x, w_hf)
        return x


class RegularizationLoss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def decorrelation_loss(self, A_comp, C):
        pass
        
        A_flat = A_comp.flatten(2)  
        C_flat = C.flatten(2)       
        
        
        A_norm = F.normalize(A_flat, p=2, dim=2)
        C_norm = F.normalize(C_flat, p=2, dim=2)
        
        
        correlation = torch.bmm(A_norm.transpose(1, 2), C_norm)
        
        
        mask = torch.eye(correlation.size(-1), device=correlation.device).unsqueeze(0)
        correlation = correlation * (1 - mask)
        
        
        loss = torch.mean(torch.abs(correlation))
        return loss
    
    def sparsity_loss(self, A_comp):
        pass
        
        l1_loss = torch.mean(torch.abs(A_comp))
        
        
        threshold = 0.1
        sparsity_loss = torch.mean(torch.relu(torch.abs(A_comp) - threshold))
        
        return l1_loss + 0.1 * sparsity_loss



class FCDM_Module(nn.Module):
    pass
    def __init__(self, channels, disable_b1=False, tau2=0.9, compute_decorr_mi_loss=False):
        super().__init__()
        self.channels = channels
        self.disable_b1 = disable_b1  
        self.compute_decorr_mi_loss = compute_decorr_mi_loss  
        
        
        
        
        
        
        self.hsi_semantic_mlp = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size=1)
        )
        
        self.lidar_semantic_mlp = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size=1)
        )
        
        
        self.Wq = nn.Conv2d(channels, channels, kernel_size=1)
        self.Wk = nn.Conv2d(channels, channels, kernel_size=1)
        self.Wv = nn.Conv2d(channels, channels, kernel_size=1)
        
        
        
        self.complementary_gate = nn.Sequential(
            nn.Conv2d(channels * 3, channels, kernel_size=1),  
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(channels, 2, kernel_size=1),  
            nn.Softmax(dim=1)
        )
        
        
        self.complementary_enhancement = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU()
        )
        
        
        self.residual_refinement = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels),
            nn.BatchNorm2d(channels),
            nn.ReLU()
        )
        
        
        
        if not disable_b1:
            self.tau1 = nn.Parameter(torch.tensor(0.8, dtype=torch.float32))
            self.beta_B1 = nn.Parameter(torch.tensor(0.05, dtype=torch.float32))
        
        
        
        self.tau2 = nn.Parameter(torch.tensor(tau2, dtype=torch.float32))  
        self.beta_B2 = nn.Parameter(torch.tensor(0.05, dtype=torch.float32))
        
        
        self.beta_agree = nn.Parameter(torch.tensor(0.10, dtype=torch.float32))
        
        self.eps = 1e-6  
        
    def forward(self, hsi_high_amp, lidar_high_amp, PH, PL, AH_low, AL_low):
        pass
        eps = 1e-8
        
        
        AH_processed = hsi_high_amp + eps
        AL_processed = lidar_high_amp + eps
        
        
        AH_semantic = self.hsi_semantic_mlp(AH_processed)
        AL_semantic = self.lidar_semantic_mlp(AL_processed)
        
        
        QH = self.Wq(AH_semantic)
        KL = self.Wk(AL_semantic)
        VL = self.Wv(AL_semantic)
        
        
        SH_L = F.softmax(QH * KL, dim=1)
        CH_L = SH_L * VL
        
        
        QL = self.Wq(AL_semantic)
        KH = self.Wk(AH_semantic)
        VH = self.Wv(AH_semantic)
        
        SL_H = F.softmax(QL * KH, dim=1)
        CL_H = SL_H * VH
        
        
        C = 0.5 * (CH_L + CL_H)
        
        
        RH = AH_processed - C
        RL = AL_processed - C
        
        
        gate_input = torch.cat([RH, RL, C], dim=1)
        alpha_weights = self.complementary_gate(gate_input)
        alpha_H = alpha_weights[:, 0:1, :, :]
        alpha_L = alpha_weights[:, 1:2, :, :]
        
        
        A_comp_original = alpha_H * RH + alpha_L * RL
        
        
        RH_enhanced = self.complementary_enhancement(RH)
        RL_enhanced = self.complementary_enhancement(RL)
        
        
        R_agree = F.softmax(RH_enhanced * RL_enhanced, dim=1) * (RH_enhanced + RL_enhanced)
        
        
        
        if not self.disable_b1:
            
            Delta_mag = torch.abs(RH_enhanced) - torch.abs(RL_enhanced)
            W_B1 = torch.sigmoid(Delta_mag / self.tau1)
            A_B1 = W_B1 * RH_enhanced + (1 - W_B1) * RL_enhanced
        else:
            A_B1 = torch.zeros_like(RH_enhanced)  
        
        
        RHn = RH_enhanced / (torch.sqrt((RH_enhanced ** 2).sum(dim=1, keepdim=True)) + self.eps)
        RLn = RL_enhanced / (torch.sqrt((RL_enhanced ** 2).sum(dim=1, keepdim=True)) + self.eps)
        
        
        S_anti = -(RHn * RLn).sum(dim=1, keepdim=True)
        
        
        W_B2 = torch.sigmoid(S_anti / self.tau2)
        
        
        A_B2 = W_B2 * (RH_enhanced - RL_enhanced)
        
        
        if not self.disable_b1:
            A_comp = A_comp_original + self.beta_agree * R_agree + self.beta_B1 * A_B1 + self.beta_B2 * A_B2
        else:
            
            A_comp = A_comp_original + self.beta_agree * R_agree + self.beta_B2 * A_B2
        
        
        P_comp = weighted_circular_phase(PH, PL, alpha_H, alpha_L)
        
        
        A_comp_clamped = torch.clamp(A_comp, min=eps, max=1e3)
        complex_comp = A_comp_clamped * torch.exp(1j * P_comp)
        F_h = torch.fft.ifft2(complex_comp, dim=(-2, -1)).real
        
        
        residual = self.residual_refinement(F_h)
        residual = torch.clamp(residual, min=-1.0, max=1.0)
        F_h_star = F_h + F.relu(residual)
        
        
        
        if self.compute_decorr_mi_loss:
            decorr_loss = self.compute_decorrelation_loss(A_comp, C)
            mi_loss = self.compute_mutual_info_loss(C, AH_processed, AL_processed)
        else:
            
            decorr_loss = torch.tensor(0.0, device=A_comp.device, dtype=A_comp.dtype)
            mi_loss = torch.tensor(0.0, device=A_comp.device, dtype=A_comp.dtype)
        
        
        
        return F_h_star, C, A_comp, decorr_loss, mi_loss, PH, PL, AH_low, AL_low
    
    def compute_decorrelation_loss(self, A_comp, C):
        pass
        decorr_H = torch.abs(A_comp * C)
        decorr_L = torch.abs(A_comp * C)
        loss_decorr = torch.norm(decorr_H, p=2) + torch.norm(decorr_L, p=2)
        return loss_decorr
    
    def compute_mutual_info_loss(self, C, AH_processed, AL_processed):
        pass
        C_flat = C.flatten(2)
        AH_flat = AH_processed.flatten(2)
        AL_flat = AL_processed.flatten(2)
        
        C_norm = F.normalize(C_flat, p=2, dim=1)
        AH_norm = F.normalize(AH_flat, p=2, dim=1)
        AL_norm = F.normalize(AL_flat, p=2, dim=1)
        
        mi_H = -torch.mean(torch.sum(C_norm * AH_norm, dim=1))
        mi_L = -torch.mean(torch.sum(C_norm * AL_norm, dim=1))
        
        return mi_H + mi_L


class FrequencyDecompose(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x, return_complex=False):
        pass
        
        fft = torch.fft.fft2(x, dim=(-2, -1))
        
        
        fft_shifted = torch.fft.fftshift(fft, dim=(-2, -1))
        
        
        B, C, H, W = fft_shifted.shape
        center_h, center_w = H // 2, W // 2
        radius = min(H, W) // 8
        
        
        y_coords = torch.arange(H, device=x.device, dtype=torch.float32)
        x_coords = torch.arange(W, device=x.device, dtype=torch.float32)
        Y, X = torch.meshgrid(y_coords, x_coords, indexing='ij')
        
        
        dist_sq = (Y - center_h) ** 2 + (X - center_w) ** 2
        mask = (dist_sq <= radius ** 2).float()
        mask = mask[None, None, :, :]  
        
        
        low_freq_shifted = fft_shifted * mask
        high_freq_shifted = fft_shifted * (1 - mask)
        
        
        low_freq = torch.fft.ifftshift(low_freq_shifted, dim=(-2, -1))
        high_freq = torch.fft.ifftshift(high_freq_shifted, dim=(-2, -1))
        
        
        if return_complex:
            return high_freq, low_freq
        else:
            return torch.abs(high_freq), torch.abs(low_freq)


class DualInputBackbone9_V4_4_22(nn.Module):
    pass
    def __init__(self, in_channels_hsi, in_channels_lidar, hidden_dim):
        super().__init__()
        self.in_channels_hsi = in_channels_hsi
        
        self.hsi_main = nn.Sequential(
            nn.Conv3d(1, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm3d(hidden_dim),
            nn.ReLU(),
            nn.Conv3d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm3d(hidden_dim),
            nn.ReLU(),
            nn.Conv3d(hidden_dim, hidden_dim, kernel_size=1),
        )
        
        
        self.lidar_main = nn.Sequential(
            nn.Conv2d(in_channels_lidar, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1),
        )
        
        
        self.hsi_aux = nn.Sequential(
            nn.Conv3d(1, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm3d(hidden_dim),
            nn.ReLU()
        )
        
        
        self.lidar_aux = nn.Sequential(
            nn.Conv2d(in_channels_lidar, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU()
        )
    
    def forward(self, x1, x2):
        pass
        
        
        
        if x1.dim() == 4:
            if x1.shape[1] != self.in_channels_hsi:
                raise ValueError(f"Unexpected HSI shape: {tuple(x1.shape)}")
            x1 = x1.unsqueeze(1)
        elif x1.dim() == 5:
            if x1.shape[1] == self.in_channels_hsi and x1.shape[2] == 1:
                x1 = x1.permute(0, 2, 1, 3, 4).contiguous()
            elif not (x1.shape[1] == 1 and x1.shape[2] == self.in_channels_hsi):
                raise ValueError(f"Unexpected HSI shape: {tuple(x1.shape)}")
        else:
            raise ValueError(f"Unexpected HSI dimension: {x1.dim()}")

        
        hsi_main = self.hsi_main(x1).mean(dim=2)  
        hsi_aux = self.hsi_aux(x1).mean(dim=2)   
        
        
        
        if x2.dim() == 5:
            
            if x2.shape[1] == 1:
                
                lidar_2d = x2.squeeze(1)
            else:
                
                lidar_2d = x2.squeeze(2)
        elif x2.dim() == 4:
            
            lidar_2d = x2
        else:
            raise ValueError(f"Unexpected LiDAR input dimension: {x2.dim()}")
        
        
        lidar_main = self.lidar_main(lidar_2d)  
        lidar_aux = self.lidar_aux(lidar_2d)    
        
        return hsi_main, lidar_main, hsi_aux, lidar_aux


class PGSE_SpatialBlock(nn.Module):
    def __init__(self, in_channels, reduction=8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.channel_gate = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1),
            nn.Sigmoid()
        )
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )
        
        self.conv_multi_scale = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels, in_channels, kernel_size=5, padding=2, groups=in_channels),
            nn.ReLU()
        )
        
    def forward(self, x):
        
        chn_attn = self.channel_gate(self.avg_pool(x))
        x_pgse = x * chn_attn + x
        max_out, _ = torch.max(x_pgse, dim=1, keepdim=True)
        avg_out = torch.mean(x_pgse, dim=1, keepdim=True)
        spatial_attn = self.spatial_gate(torch.cat([avg_out, max_out], dim=1))
        x_pgse = x_pgse * spatial_attn + x_pgse
        
        
        x_fused = self.conv_multi_scale(x_pgse)
        
        return x_fused


class DualINNAuxClassifier(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_channels, num_classes)
        )
    def forward(self, z):
        return self.classifier(z)


class SEBlock(nn.Module):
    pass
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class DualRevNetBlock(nn.Module):
    pass
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        
        
        self.F = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels)
        )
        
        
        self.G = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels)
        )
    
    def forward(self, hsi_feat, lidar_feat):
        pass
        
        
        
        
        hsi_out = hsi_feat + self.F(lidar_feat)
        
        
        
        lidar_out = lidar_feat + self.G(hsi_out)
        
        
        
        
        
        
        return hsi_out, lidar_out


class DualFeatureDisentangler(nn.Module):
    def __init__(self, channels, num_blocks=3):
        super().__init__()
        self.channels = channels
        assert channels % 2 == 0, "Channels must be even for disentanglement"
        
        
        self.blocks = nn.ModuleList([
            DualRevNetBlock(channels) 
            for _ in range(num_blocks)
        ])
        
        
        self.freq_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels, 1),
            nn.Sigmoid()
        )
        self.spat_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels, 1),
            nn.Sigmoid()
        )
        
        
        self.cross_modal_attention = nn.Sequential(
            nn.Conv2d(channels * 4, channels, 1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, 2, 1),
            nn.Softmax(dim=1)
        )
        
    def forward(self, freq_feat, spat_feat):
        
        for block in self.blocks:
            freq_feat, spat_feat = block(freq_feat, spat_feat)
        
        
        freq_w = self.freq_gate(freq_feat)
        spat_w = self.spat_gate(spat_feat)
        
        
        freq_a = freq_feat * freq_w
        freq_b = freq_feat * (1 - freq_w)
        spat_a = spat_feat * spat_w
        spat_b = spat_feat * (1 - spat_w)
        
        
        cross_modal_input = torch.cat([freq_a, spat_a, freq_b, spat_b], dim=1)
        att = self.cross_modal_attention(cross_modal_input)
        alpha_f = att[:, 0:1, :, :]
        alpha_s = att[:, 1:2, :, :]
        
        
        z1 = alpha_f * freq_a + alpha_s * spat_a
        z2 = alpha_f * freq_b + alpha_s * spat_b
        
        return z1, z2


def js_divergence_improved(hsi_high, lidar_high, temperature=0.5, eps=1e-6):
    pass
    
    p = F.softmax(hsi_high.flatten(2) / temperature, dim=1)  
    q = F.softmax(lidar_high.flatten(2) / temperature, dim=1)  
    
    
    m = 0.5 * (p + q)
    
    js_per_pixel = 0.5 * (
        (p * ((p + eps) / (m + eps)).log()).sum(dim=1) +  
        (q * ((q + eps) / (m + eps)).log()).sum(dim=1)    
    )
    
    
    loss_js = js_per_pixel.mean()
    return loss_js


def get_triplet_samples_batch_hard(feat, label):
    B, C, H, W = feat.shape
    
    
    center_idx = (H // 2) * W + (W // 2)
    x = feat.permute(0, 2, 3, 1).reshape(B, H*W, C)[:, center_idx, :]  
    
    
    with torch.no_grad():
        
        x_norm = F.normalize(x, p=2, dim=1)
        dist = torch.cdist(x_norm, x_norm, p=2)
    
    anchors, positives, negatives = [], [], []
    for i in range(B):
        same = (label == label[i]).nonzero(as_tuple=True)[0]
        diff = (label != label[i]).nonzero(as_tuple=True)[0]
        
        
        same = same[same != i]
        
        if len(same) > 0:
            
            p_idx = same[torch.argmax(dist[i, same]).item()]
        else:
            
            p_idx = i
        
        if len(diff) > 0:
            
            n_idx = diff[torch.argmin(dist[i, diff]).item()]
        else:
            n_idx = i  
        
        anchors.append(x[i])
        positives.append(x[p_idx])
        negatives.append(x[n_idx])
    
    return torch.stack(anchors), torch.stack(positives), torch.stack(negatives)


class TripletLoss(nn.Module):
    def __init__(self, margin=2.0):
        super().__init__()
        self.margin = margin
    
    def forward(self, anchors, positives, negatives):
        pass
        
        anchors = F.normalize(anchors, p=2, dim=1)
        positives = F.normalize(positives, p=2, dim=1)
        negatives = F.normalize(negatives, p=2, dim=1)
        
        
        pos_dist = torch.sum((anchors - positives) ** 2, dim=1)
        neg_dist = torch.sum((anchors - negatives) ** 2, dim=1)
        
        
        loss = torch.clamp(pos_dist - neg_dist + self.margin, min=0.0)
        return loss.mean()

class FCIFNet(nn.Module):
    pass
    def __init__(self, in_ch1=63, hidden_dim=64, in_ch2=2, num_classes=6, 
                 js_temperature=0.5, triplet_margin=2.0, enable_triplet_after_epochs=15,
                 hfce_residual_scale=0.7, disable_b1_gate=True, tau2=0.9,
                 use_triplet_loss=False, compute_decorr_mi_loss=False):
        super().__init__()
        self.in_ch1 = in_ch1
        self.in_ch2 = in_ch2
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.js_temperature = js_temperature
        self.triplet_margin = triplet_margin
        
        self.enable_triplet_after_epochs = enable_triplet_after_epochs
        self.use_triplet_loss = use_triplet_loss  
        self.compute_decorr_mi_loss = compute_decorr_mi_loss  
        self.current_epoch = 0
        
        
        self.backbone = DualInputBackbone9_V4_4_22(self.in_ch1, self.in_ch2, self.hidden_dim)
        
        
        
        self.fcdm_module = FCDM_Module(self.hidden_dim, disable_b1=disable_b1_gate, tau2=tau2, 
                                       compute_decorr_mi_loss=False)
        
        
        self.freq_decompose = FrequencyDecompose()
        
        
        self.pgse_spatial = PGSE_SpatialBlock(self.hidden_dim * 2)  
        
        
        self.c_fusion = nn.Sequential(
            nn.Conv2d(self.hidden_dim, self.hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.hidden_dim),
            nn.ReLU(),
            nn.Conv2d(self.hidden_dim, self.hidden_dim, kernel_size=1)
        )
        
        
        
        self.low_freq_norm = nn.LayerNorm([self.hidden_dim], elementwise_affine=True)
        self.low_freq_gate = nn.Sequential(
            nn.Conv2d(self.hidden_dim, self.hidden_dim, kernel_size=1),
            nn.Sigmoid()
        )
        
        
        
        self.low_freq_proj = nn.Sequential(
            nn.Conv2d(self.hidden_dim, self.hidden_dim, kernel_size=1),
            nn.BatchNorm2d(self.hidden_dim),
            nn.ReLU(inplace=True)
        )
        
        
        
        self.compress_freq = nn.Conv2d(self.hidden_dim, self.hidden_dim // 2, 1)
        
        self.compress_spatial = nn.Conv2d(self.hidden_dim, self.hidden_dim // 2, 1)
        
        
        
        self.dual_inn = DualFeatureDisentangler(
            self.hidden_dim // 2, 
            num_blocks=3
        )
        self.dual_inn_aux = DualINNAuxClassifier(self.hidden_dim // 2, self.num_classes)
        
        
        classifier_input_channels = self.hidden_dim
        self.classifier = nn.Sequential(
            nn.Conv2d(classifier_input_channels, self.hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout2d(p=0.3),
            nn.ReLU(),
            nn.Conv2d(self.hidden_dim, self.num_classes, kernel_size=1)
        )
        
        
        self.triplet_loss = TripletLoss(margin=self.triplet_margin)
        
        
        self.hfce_boost = HFCEBoost(self.hidden_dim, n_blocks=2, dilation=2, reduction=8, 
                                    residual_scale=hfce_residual_scale)
        self.reg_loss = RegularizationLoss()
        
    def set_epoch(self, epoch):
        pass
        self.current_epoch = epoch
        
    def forward(self, x1, x2, labels=None):
        
        if self.training:
            if torch.rand(1).item() < 0.3:
                noise_scale = 0.0005 + torch.rand(1).item() * 0.001
                noise = torch.randn_like(x1) * noise_scale
                x1 = x1 + noise
            
            if torch.rand(1).item() < 0.2:
                scale = 0.99 + torch.rand(1).item() * 0.02
                x1 = x1 * scale
        
        
        
        if x1.dim() == 5:
            if x1.shape[1] == self.in_ch1 and x1.shape[2] == 1:
                x1 = x1.permute(0, 2, 1, 3, 4).contiguous()
            elif not (x1.shape[1] == 1 and x1.shape[2] == self.in_ch1):
                raise ValueError(f"Unexpected HSI input shape: {tuple(x1.shape)}")
        elif x1.dim() == 4:
            if x1.shape[1] != self.in_ch1:
                raise ValueError(f"Unexpected HSI input shape: {tuple(x1.shape)}")
            x1 = x1.unsqueeze(1)
        else:
            raise ValueError(f"Unexpected HSI input dimension: {x1.dim()}")
        
        if x2.dim() == 5:
            if x2.shape[1] == 1 and x2.shape[2] == self.in_ch2:
                
                x2 = x2.permute(0, 2, 1, 3, 4)
        elif x2.dim() == 4:
            x2 = x2.unsqueeze(2)
            
        
        hsi_main, lidar_main, hsi_aux, lidar_aux = self.backbone(x1, x2)
        
        
        
        hsi_high_complex, hsi_low_complex = self.freq_decompose(hsi_main, return_complex=True)
        lidar_high_complex, lidar_low_complex = self.freq_decompose(lidar_main, return_complex=True)
        
        
        
        PH = torch.angle(hsi_high_complex)  
        PL = torch.angle(lidar_high_complex)  
        
        
        hsi_high_amp = torch.abs(hsi_high_complex)
        hsi_low_amp = torch.abs(hsi_low_complex)
        lidar_high_amp = torch.abs(lidar_high_complex)
        lidar_low_amp = torch.abs(lidar_low_complex)
        
        
        hsi_low_spatial = torch.fft.ifft2(hsi_low_complex, dim=(-2, -1)).real
        lidar_low_spatial = torch.fft.ifft2(lidar_low_complex, dim=(-2, -1)).real
        
        
        
        
        F_h_star, C, A_comp, decorr_loss, mi_loss, PH_out, PL_out, AH_low, AL_low = self.fcdm_module(
            hsi_high_amp, lidar_high_amp, PH, PL, hsi_low_amp, lidar_low_amp
        )
        
        
        
        F_h_enh = self.hfce_boost(F_h_star, A_comp, C)
        
        
        hsi_high = hsi_high_amp
        lidar_high = lidar_high_amp
        
        
        aux_combined = torch.cat([hsi_aux, lidar_aux], dim=1)  
        aux_enhanced = self.pgse_spatial(aux_combined)  
        
        hsi_aux_enhanced = aux_enhanced[:, :self.hidden_dim, :, :]
        lidar_aux_enhanced = aux_enhanced[:, self.hidden_dim:, :, :]
        common_feat = (hsi_aux_enhanced + lidar_aux_enhanced) / 2
        
        
        
        low_freq_common = (hsi_low_spatial + lidar_low_spatial) / 2
        
        
        
        B, num_channels, H, W = low_freq_common.shape  
        low_freq_normed = self.low_freq_norm(low_freq_common.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        
        low_freq_gate = self.low_freq_gate(low_freq_normed)
        
        low_freq_gated = low_freq_normed * low_freq_gate
        
        
        
        low_freq_projected = self.low_freq_proj(low_freq_gated)
        
        
        
        
        phase_weight = torch.full_like(PH, 0.5)
        P_C = weighted_circular_phase(PH, PL, phase_weight, phase_weight)
        C_clamped = torch.clamp(C, min=1e-8, max=1e3)  
        C_complex = C_clamped * torch.exp(1j * P_C)
        C_spatial = torch.fft.ifft2(C_complex, dim=(-2, -1)).real  
        
        
        C_spatial_enhanced = self.c_fusion(C_spatial)
        
        
        common_feat = common_feat + low_freq_gated + 0.5 * C_spatial_enhanced
        
        
        
        freq_compressed = self.compress_freq(F_h_enh)  
        
        spatial_compressed = self.compress_spatial(common_feat)  
        
        
        z_a, z_b = self.dual_inn(freq_compressed, spatial_compressed)
        
        spectral_frequency_feat = z_a
        
        spatial_context_feat = z_b
        aux_out = self.dual_inn_aux(spectral_frequency_feat)
        
        
        fusion = torch.cat([spectral_frequency_feat, spatial_context_feat], dim=1)
        out = self.classifier(fusion)
        out = F.adaptive_avg_pool2d(out, 1).squeeze(-1).squeeze(-1)
        
        
        
        js_diff = js_divergence_improved(hsi_high, lidar_high, self.js_temperature)
        
        
        
        
        
        triplet_loss_value = None
        if self.use_triplet_loss and self.training and labels is not None and self.current_epoch >= self.enable_triplet_after_epochs:
            
            anchors, positives, negatives = get_triplet_samples_batch_hard(spatial_context_feat, labels)
            triplet_loss_value = self.triplet_loss(anchors, positives, negatives)
        
        
        
        
        if self.compute_decorr_mi_loss:
            loss_decorr = self.reg_loss.decorrelation_loss(A_comp, C)
            loss_sparse = self.reg_loss.sparsity_loss(A_comp)
        else:
            
            loss_decorr = torch.tensor(0.0, device=A_comp.device, dtype=A_comp.dtype)
            loss_sparse = torch.tensor(0.0, device=A_comp.device, dtype=A_comp.dtype)
        
        return {
            'main_out': out,
            'aux_out': aux_out,
            'spectral_frequency_feat': spectral_frequency_feat,
            'spatial_context_feat': spatial_context_feat,
            'js_diff': js_diff,  
            'triplet_loss': triplet_loss_value,  
            'F_h_star': F_h_star,
            'C': C,
            'A_comp': A_comp,
            'F_h_enh': F_h_enh,
            'C_spatial': C_spatial,  
            'low_freq_projected': low_freq_projected,  
            'loss_decorr': loss_decorr,  
            'loss_sparse': loss_sparse,
            'decorr_loss': decorr_loss,  
            'mi_loss': mi_loss  
        }



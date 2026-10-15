import torch
import torch.nn as nn
import torch.nn.functional as F

                           
class RegularizationLoss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def sparsity_loss(self, A_comp):
           
               
        l1_loss = torch.mean(torch.abs(A_comp))
        
                  
        threshold = 0.1
        sparsity_loss = torch.mean(torch.relu(torch.abs(A_comp) - threshold))
        
        return l1_loss + 0.1 * sparsity_loss

                                        
                          
class FCDM_Module(nn.Module):
       
    def __init__(self, channels, tau2=0.9):
        super().__init__()
        self.channels = channels
        
                                     
                           
        
                            
                           
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
        
                                                    
                                     
        self.frequency_refinement = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels)
        )
        
                         
                        
        
                  
                                                 
        self.tau2 = nn.Parameter(torch.tensor(tau2, dtype=torch.float32))         
        self.beta_B2 = nn.Parameter(torch.tensor(0.05, dtype=torch.float32))
        
                
        self.beta_agree = nn.Parameter(torch.tensor(0.10, dtype=torch.float32))
        
        self.eps = 1e-6         
        
    def forward(self, hsi_high_amp, lidar_high_amp, PH, PL):
           
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
        
                       
                                               
        
                                                     
        RHn = RH_enhanced / (torch.sqrt((RH_enhanced ** 2).sum(dim=1, keepdim=True)) + self.eps)
        RLn = RL_enhanced / (torch.sqrt((RL_enhanced ** 2).sum(dim=1, keepdim=True)) + self.eps)
        
                        
        S_anti = -(RHn * RLn).sum(dim=1, keepdim=True)
        
                                
        W_B2 = torch.sigmoid(S_anti / self.tau2)
        
                 
        A_B2 = W_B2 * (RH_enhanced - RL_enhanced)
        
                  
        A_comp = A_comp_original + self.beta_agree * R_agree + self.beta_B2 * A_B2
        
                                                 
        sin_H, cos_H = torch.sin(PH), torch.cos(PH)
        sin_L, cos_L = torch.sin(PL), torch.cos(PL)
        sin_comp = alpha_H * sin_H + alpha_L * sin_L
        cos_comp = alpha_H * cos_H + alpha_L * cos_L

        phase_eps = 1e-6
        phase_r2 = sin_comp.square() + cos_comp.square()
        phase_norm = torch.sqrt(phase_r2 + phase_eps ** 2)
        sin_unit = sin_comp / phase_norm
        cos_unit = cos_comp / phase_norm

                                            
        ambiguous = phase_r2 <= phase_eps ** 2
        sin_unit = torch.where(ambiguous, sin_H, sin_unit)
        cos_unit = torch.where(ambiguous, cos_H, cos_unit)
        
                                       
        delta_A = self.frequency_refinement(A_comp)
        A_freq = A_comp + delta_A
        A_freq_clamped = torch.clamp(A_freq, min=eps, max=1e3)
        complex_comp = torch.complex(
            A_freq_clamped * cos_unit,
            A_freq_clamped * sin_unit
        )
        F_h = torch.fft.ifft2(complex_comp, dim=(-2, -1)).real
        return F_h, A_comp

        
class FrequencyDecompose(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
           
                   
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
        
                                        
        return high_freq, low_freq

                          
class DualInputBackbone9_V4_4_22(nn.Module):
       
    def __init__(self, in_channels_hsi, in_channels_lidar, hidden_dim):
        super().__init__()
        self.in_channels_hsi = in_channels_hsi
                              
        self.hsi_main = nn.Sequential(
            nn.Conv2d(in_channels_hsi, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1),
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
            nn.Conv2d(in_channels_hsi, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU()
        )
        
                        
        self.lidar_aux = nn.Sequential(
            nn.Conv2d(in_channels_lidar, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU()
        )
    
    def forward(self, x1, x2):
           
        if x1.dim() != 4 or x1.shape[1] != self.in_channels_hsi:
            raise ValueError(
                f"HSI Conv2d expects [B, {self.in_channels_hsi}, H, W], "
                f"got {tuple(x1.shape)}"
            )

        hsi_main = self.hsi_main(x1)                         
        hsi_aux = self.hsi_aux(x1)                           
        
                               
                   
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

             
                                    
class DualRevNetBlock(nn.Module):
       
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
       
                  
    p = F.softmax(hsi_high.flatten(2) / temperature, dim=1)              
    q = F.softmax(lidar_high.flatten(2) / temperature, dim=1)              
    
                   
    m = 0.5 * (p + q)
    
    js_per_pixel = 0.5 * (
        (p * ((p + eps) / (m + eps)).log()).sum(dim=1) +           
        (q * ((q + eps) / (m + eps)).log()).sum(dim=1)             
    )
    
                
    loss_js = js_per_pixel.mean()
    return loss_js

                              
class FCIFNet(nn.Module):
       
    def __init__(self, in_ch1=63, hidden_dim=64, in_ch2=2, num_classes=6,
                 js_temperature=0.5, tau2=0.9):
        super().__init__()
        self.in_ch1 = in_ch1
        self.in_ch2 = in_ch2
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.js_temperature = js_temperature
        
                               
        self.backbone = DualInputBackbone9_V4_4_22(self.in_ch1, self.in_ch2, self.hidden_dim)
        
                                          
                                         
        self.fcdm_module = FCDM_Module(self.hidden_dim, tau2=tau2)
        
                            
        self.freq_decompose = FrequencyDecompose()
        
                                               
        self.pgse_spatial = PGSE_SpatialBlock(self.hidden_dim * 2)                           
        
                                        
        
                               
                                                        
        self.low_freq_norm = nn.LayerNorm([self.hidden_dim], elementwise_affine=True)
        self.low_freq_gate = nn.Sequential(
            nn.Conv2d(self.hidden_dim, self.hidden_dim, kernel_size=1),
            nn.Sigmoid()
        )
        
                                                     
                                 
        
                                       
                
        self.compress_freq = nn.Conv2d(self.hidden_dim, self.hidden_dim // 2, 1)
                
        self.compress_spatial = nn.Conv2d(self.hidden_dim, self.hidden_dim // 2, 1)
        
                                                 
                                  
        self.dual_inn = DualFeatureDisentangler(
            self.hidden_dim // 2, 
            num_blocks=3
        )
        
                                                                                 
        classifier_input_channels = self.hidden_dim
        self.classifier = nn.Sequential(
            nn.Conv2d(classifier_input_channels, self.hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout2d(p=0.3),
            nn.ReLU(),
            nn.Conv2d(self.hidden_dim, self.num_classes, kernel_size=1)
        )
        
               
        self.reg_loss = RegularizationLoss()

    def forward(self, x1, x2):
                   
        if self.training:
            if torch.rand(1).item() < 0.3:
                noise_scale = 0.0005 + torch.rand(1).item() * 0.001
                noise = torch.randn_like(x1) * noise_scale
                x1 = x1 + noise
            
            if torch.rand(1).item() < 0.2:
                scale = 0.99 + torch.rand(1).item() * 0.02
                x1 = x1 * scale
        
                                                      
        if x1.dim() == 5:
            if x1.shape[1] == 1 and x1.shape[2] == self.in_ch1:
                x1 = x1.squeeze(1)
            elif x1.shape[1] == self.in_ch1 and x1.shape[2] == 1:
                x1 = x1.squeeze(2)
            else:
                raise ValueError(f"Unexpected HSI input shape: {tuple(x1.shape)}")
        elif x1.dim() == 4:
            if x1.shape[1] != self.in_ch1:
                raise ValueError(f"Unexpected HSI input shape: {tuple(x1.shape)}")
        else:
            raise ValueError(f"Unexpected HSI input dimension: {x1.dim()}")
        
        if x2.dim() == 5:
            if x2.shape[1] == 1 and x2.shape[2] == self.in_ch2:
                                                                  
                x2 = x2.permute(0, 2, 1, 3, 4)
        elif x2.dim() == 4:
            x2 = x2.unsqueeze(2)
            
                  
        hsi_main, lidar_main, hsi_aux, lidar_aux = self.backbone(x1, x2)
        
                                     
                               
        hsi_high_complex, hsi_low_complex = self.freq_decompose(hsi_main)
        lidar_high_complex, lidar_low_complex = self.freq_decompose(lidar_main)
        
                                       
                                           
        PH = torch.angle(hsi_high_complex)        
        PL = torch.angle(lidar_high_complex)        
        
              
        hsi_high_amp = torch.abs(hsi_high_complex)
        lidar_high_amp = torch.abs(lidar_high_complex)
        
                                  
        hsi_low_spatial = torch.fft.ifft2(hsi_low_complex, dim=(-2, -1)).real
        lidar_low_spatial = torch.fft.ifft2(lidar_low_complex, dim=(-2, -1)).real
        
                                         
                                
                                             
        F_h_star, A_comp = self.fcdm_module(
            hsi_high_amp, lidar_high_amp, PH, PL
        )
        
                                             
        F_h_enh = F_h_star
        
                                     
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
        
                                           
                                           
                               
        common_feat = common_feat + low_freq_gated
        
                                         
                
        freq_compressed = self.compress_freq(F_h_enh)                            
                
        spatial_compressed = self.compress_spatial(common_feat)                            
        
                                
        z_a, z_b = self.dual_inn(freq_compressed, spatial_compressed)
                     
        spectral_frequency_feat = z_a
                     
        spatial_context_feat = z_b
        
                          
        fusion = torch.cat([spectral_frequency_feat, spatial_context_feat], dim=1)
        out = self.classifier(fusion)
        out = F.adaptive_avg_pool2d(out, 1).squeeze(-1).squeeze(-1)
        
                                                     
                                                     
        js_diff = js_divergence_improved(hsi_high, lidar_high, self.js_temperature)
        
                                        
                                                  
                            
                                                  
        loss_sparse = self.reg_loss.sparsity_loss(A_comp)
        
        return {
            'main_out': out,
            'js_diff': js_diff,                                   
            'loss_sparse': loss_sparse,
        }



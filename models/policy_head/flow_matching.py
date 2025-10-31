import torch
from torch.distributions import Beta, Uniform

class FlowMatching:
    def __init__(self, model, sample_method="beta"):
        
        assert sample_method in ["uniform", "beta"], "Invalid sampling method"
        self.model = model
        self.sample_method = sample_method
        self.s = 1.0  # The threshold for timesteps

        if self.sample_method == "beta":
            # Beta(1.5, 1.0) distribution
            self.distribution = Beta(torch.tensor([1.5]), torch.tensor([1.0]))
        elif self.sample_method == "uniform":
            # Uniform distribution from [0, s]
            self.distribution = Uniform(torch.tensor([0.0]), torch.tensor([1.0]))
        
    def sample_t(self, num_samples):
        """
        Sample timesteps using the specified distribution.

        Args:
            num_samples (int): Number of samples to generate.

        Returns:
            torch.Tensor: Timesteps sampled and scaled to [0, s].
        """
        if self.sample_method == "beta":
            # Sample from the Beta distribution and scale to [0, s]
            samples = self.distribution.sample((num_samples,))
            timesteps = self.s * (1 - samples)  # Scale to [0, s]
        else:
            # Sample uniformly in [0, s]
            timesteps = self.distribution.sample((num_samples,)) * self.s

        return timesteps.squeeze(1)

    def forward(self, x, cond):
        b = x.size(0)
        
        # 采样时间步t，采用指定的分布
        t = self.sample_t(b).to(x.device)

        # 计算时间步t对应的扩展形式，用于广播
        texp = t.view(b, *([1] * (x.dim() - 1)))
        
        # 生成与x相同形状的噪声张量z1
        z1 = torch.randn_like(x)
        
        # 计算带噪声的输入
        zt = (1 - texp) * x + texp * z1
        
        # 使用模型对带噪声的输入进行预测
        vtheta = self.model(zt, t, cond)
        
        # 计算均方误差（MSE）
        batchwise_mse = ((z1 - x - vtheta) ** 2).mean(dim=tuple(range(1, x.dim())))
        
        # 将MSE结果转化为列表
        ttloss = list(zip(t.cpu().tolist(), batchwise_mse.detach().cpu().tolist()))
        
        return batchwise_mse.mean(), ttloss

    @torch.no_grad()
    def sample(self, z, cond, null_cond=None, sample_steps=50, cfg=2.0):
        b = z.size(0)
        dt = 1.0 / sample_steps
        dt = torch.tensor([dt] * b).to(z.device).view([b, *([1] * len(z.shape[1:]))])
        images = [z]
        for i in range(sample_steps, 0, -1):
            t = i / sample_steps
            t = torch.tensor([t] * b).to(z.device)

            vc = self.model(z, t, cond)
            if null_cond is not None:
                vu = self.model(z, t, null_cond)
                vc = vu + cfg * (vc - vu)

            z = z - dt * vc
            images.append(z)
        return images
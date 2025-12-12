import torch

__all__ = ['PSCoder', 'UCResolver']


def angle_coder_scheme(i):
    return {
        0: """
bbox_pred────(tblr)───┐
                      ▼
angle_pred          decode──►rbox_pred──(xywha)─►loss_bbox
    │                 ▲
    ├────►decode──(a)─┘
    │
    └───────────────────────────────────────────►loss_angle
""",
        1: """
bbox_pred────(tblr)───┐
                      ▼
angle_pred          decode──►rbox_pred──(xywha)─►loss_bbox
    │                 ▲
    └────►decode──(a)─┘
"""
    }.get(i, 0)


class PSCoder:
    """Phase-Shifting Coder.

    `Phase-Shifting Coder (PSC)
    <https://arxiv.org/abs/2211.06368>`.

    Args:
        ns (int, optional): Number of phase steps. Also denoted as N_step, Default: 3.
        df (bool, optional): Whether to use dual frequency. Default: True.
        tm (float): Threshold of modulation. Default: 0.47.
        ang_ver (str): Angle definition version. Choose in ['le90', 'le135'].
        dec_mod (str): decoder mode. Choose in ['cosine', 'diff'].
    """
    # The size of the last of dimension of the encoded tensor.
    encode_size = 3

    def __init__(self, ns: int = 3, df: bool = True, tm: float = 0.47, ang_ver: str = 'le90', dec_mod: str = 'cosine'):
        assert ang_ver.lower() in ['le90', 'le135']
        assert dec_mod.lower() in ['cosine', 'diff']
        self.df = df
        self.ns = ns
        self.tm = tm
        self.encode_size = 2 * self.ns if self.df else self.ns
        self.ang_ver = ang_ver.lower()
        self.dec_mod = dec_mod.lower()
        self.ang_ofs = 0.0 if self.ang_ver == 'le90' else torch.pi / 4  # angle offset

        # Note: In the paper, n starts from 1, while in this code, it starts from 0.
        self.coef_sin = torch.tensor(tuple(
            torch.sin(torch.tensor(2 * n * torch.pi / self.ns))
            for n in range(self.ns)
        ))  # sin(2 * n * π / N_step)
        self.coef_cos = torch.tensor(tuple(
            torch.cos(torch.tensor(2 * n * torch.pi / self.ns))
            for n in range(self.ns)
        ))  # cos(2 * n * π / N_step)

    def encode(self, angle: torch.Tensor) -> torch.Tensor:
        """Phase-Shifting Encoder.

        Args:
            angle (torch.Tensor): Angle offset for each scale level.
                Has shape (num_anchors * H * W, 1)
                Also see 'ultralytics/utils/ops.py': poly2rbox
        Returns:
            list[torch.Tensor]: The psc coded data (phase-shifting patterns)
                for each scale level.
                Has shape (num_anchors * H * W, encode_size)
        """
        # input: θ ∈ [-π/2, π/2) if ang_ver == 'le90' else θ ∈ [-π/4, 3π/4)
        angle = angle - self.ang_ofs  # θ = angle - ang_ofs, θ ∈ [-π/2, π/2)

        # φ1 = 2 * θ, θ ∈ [-π, π)
        phase_angle = angle * 2
        # x_n = cos(φ + 2 * n * π / N_step)
        phase_shift_x = tuple(
            torch.cos(phase_angle + 2 * torch.pi * n / self.ns)
            for n in range(self.ns)
        )  # X1

        # Dual-freq PSC for square-like problem
        if self.df:
            # φ2 = 4 * θ, θ ∈ [-2π, 2π)
            phase_angle = angle * 4
            # x_n = cos(φ + 2 * n * π / N_step)
            phase_shift_x += tuple(
                torch.cos(phase_angle + 2 * torch.pi * n / self.ns)
                for n in range(self.ns)
            )  # X2

        return torch.cat(phase_shift_x, dim=-1)  # {X1, X2}

    def decode(self, x: torch.Tensor, keepdim: bool = False, ne: int = 1) -> torch.Tensor:
        """Phase-Shifting Decoder.

        Args:
            x (torch.Tensor): The psc coded data (phase-shifting patterns), angle of prediction.
                for each scale level.
                Has shape (bs, encode_size, L), x in [-1, 1]
            keepdim (bool): Whether the output tensor has dim retained or not.
            ne (int): number of extra parameters, see OBB head

        Returns:
            list[Tensor]: Angle offset for each scale level.
                Has shape (batch size, 1, L) when keepdim is true,
                (num_anchors * H * W) otherwise
        """

        # Adjust the input dimension: (bs, encode_size, L) -> (bs, L, encode_size)
        x = x.permute(0, 2, 1)  # Rearrange the dimensions: Place the feature dimensions at the end
        batch_size, L = x.shape[0], x.shape[1]  # Save the original batch size and the length of the space dimension
        # Merge batch and spatial dimensions to adapt to the original decoding logic
        x = x.reshape(-1, x.shape[-1])  # new shape: (bs * L, encode_size)

        self.coef_sin = self.coef_sin.to(x)
        self.coef_cos = self.coef_cos.to(x)

        # decode φ1 | sum(): Generally, the dimensions should remain unchanged, so keepdim=True
        phase_sin = torch.sum(x[:, 0:self.ns] * self.coef_sin, dim=-1, keepdim=keepdim)
        phase_cos = torch.sum(x[:, 0:self.ns] * self.coef_cos, dim=-1, keepdim=keepdim)
        phase_mod = phase_cos ** 2 + phase_sin ** 2
        phase1 = -torch.atan2(phase_sin, phase_cos)  # φ1 ∈ [-π, π)

        if self.df:
            # decode φ2
            phase_sin = torch.sum(x[:, self.ns:(2 * self.ns)] * self.coef_sin, dim=-1, keepdim=keepdim)
            phase_cos = torch.sum(x[:, self.ns:(2 * self.ns)] * self.coef_cos, dim=-1, keepdim=keepdim)
            phase_mod = phase_cos ** 2 + phase_sin ** 2
            phase2 = -torch.atan2(phase_sin, phase_cos) / 2

            # Phase unwrapping, dual freq mixing, mix them to obtain the final phase
            if self.dec_mod == 'cosine':
                # Use cosine similarity for judgment
                # Angle between phase1 and phase2 is obtuse angle, δ < 0
                # δ = cos(φ1) * cos(φ2) + sin(φ1) * sin(φ2)
                idx = (torch.cos(phase1) * torch.cos(phase2) + torch.sin(phase1) * torch.sin(phase2)) < 0
                # Subtract π to phase2 and keep it in range [-π, π)
                phase2[idx] = phase2[idx] % (2 * torch.pi) - torch.pi
            elif self.dec_mod == 'diff':
                # Method for minimizing Angle difference
                # Two candidates: phase2 and phase2 + π
                phase3 = phase2 + torch.pi
                # Calculate the Angle difference from phase1 (considering circular symmetry)
                dif2 = torch.abs(phase2 - phase1)
                dif3 = torch.abs(phase3 - phase1)
                # Handle the situation where the Angle difference exceeds π (select a smaller arc length)
                dif2 = torch.where(dif2 > torch.pi, 2 * torch.pi - dif2, dif2)
                dif3 = torch.where(dif3 > torch.pi, 2 * torch.pi - dif3, dif3)
                # Choose the one with a smaller Angle difference
                phase2 = torch.where(dif2 > dif3, phase3, phase2)

            phase1 = phase2

        # Set the angle of isotropic objects to zero
        phase1[phase_mod < self.tm] = 0.  # Force it to be set in the horizontal direction
        _angle = phase1 / 2  # θ = φ / 2, θ ∈ [-π/2, π/2)
        _angle += self.ang_ofs  # θ ∈ [-π/2, π/2) if ang_ver == 'le90' else θ ∈ [-π/4, 3π/4)

        if keepdim:
            angle = _angle.view(batch_size, ne, L)  # shape: (bs, 1, L)
        else:
            angle = _angle.view(batch_size, L)  # shape: (bs, L)

        return angle  # angle of prediction


class UCResolver:
    """Unit Cycle Resolver.

    Args:
        ns (int, optional): Number of mapping Dimension.
        invalid_thr (float, optional): Threshold of invalid angle code. Defaults to 0.0.
        ang_ver (str): Angle definition. Only 'le90' is supported at present.
    """
    encode_size = 3

    def __init__(self, ns: int = 3, invalid_thr: float = 0.0, ang_ver: str = 'le90'):
        assert ang_ver.lower() in ['le90']
        assert ns >= 2
        self.ang_ver = ang_ver
        self.ns = ns
        self.invalid_thr = invalid_thr
        self.encode_size = ns

        self.coef_sin = torch.tensor(tuple(
            torch.sin(torch.tensor(2 * k * torch.pi / self.ns))
            for k in range(self.ns)
        ))
        self.coef_cos = torch.tensor(tuple(
            torch.cos(torch.tensor(2 * k * torch.pi / self.ns))
            for k in range(self.ns)
        ))

    def encode(self, angle: torch.Tensor) -> torch.Tensor:
        """Unit Cycle Resolver.

        Args:
            angle (Tensor): Angle offset for each scale level.
                Has shape (..., num_anchors * H * W, 1)

        Returns:
            (Tensor): The psc coded data (phase-shifting patterns)
                for each scale level.
                Has shape (..., num_anchors * H * W, encode_size)
        """
        angle = angle * 2

        if self.ns > 2:
            ang_enc = torch.cat(
                [torch.cos(angle + 2 * torch.pi * x / self.ns) for x in range(self.ns)],
                dim=-1
            )
        else:
            ang_enc = torch.cat(
                [torch.cos(angle), torch.sin(angle)],
                dim=-1
            )

        return ang_enc

    def decode(self, x: torch.Tensor, keepdim: bool = False) -> torch.Tensor:
        """Unit Cycle Resolver.

        Args:
            x (Tensor): The encoding state of angle.
                Has shape (..., num_anchors * H * W, encode_size)
            keepdim (bool): Whether the output tensor has dim retained or not.

        Returns:
            (Tensor): Angle offset for each scale level.
                Has shape (num_anchors * H * W, 1) when keepdim is true,
                (num_anchors * H * W) otherwise
        """

        self.coef_sin = self.coef_sin.to(x)
        self.coef_cos = self.coef_cos.to(x)

        if self.ns > 2:
            pred_cos = torch.sum(x * self.coef_cos, dim=-1, keepdim=keepdim)
            pred_sin = - torch.sum(x * self.coef_sin, dim=-1, keepdim=keepdim)
        else:
            pred_cos = x[..., 0, None]
            pred_sin = x[..., 1, None]

        theta = torch.atan2(pred_sin, pred_cos)

        if self.invalid_thr > 0:
            theta[pred_sin ** 2 + pred_cos ** 2 < (self.ns / 2) ** 2 * self.invalid_thr] *= 0

        return theta / 2

    def get_restrict_loss(self, x: torch.Tensor, weight, avg_factor) -> torch.Tensor:
        """Unit Cycle Resolver.

        Args:
            x (Tensor): The encoding state of angle.
                Has shape (..., num_anchors * H * W, encode_size)
            weight : Weight of loss.
            avg_factor : Average factor.

        Returns:
            (Tensor): Angle restrict loss.
                Has shape (1)
        """
        assert self.ns <= 3

        d_angle_restrict = torch.sum(torch.pow(x, 2), dim=-1)
        d_angle_target = torch.ones_like(d_angle_restrict) * torch.tensor(self.ns / 2)
        loss_angle_restrict = (torch.abs(d_angle_restrict - d_angle_target) * weight).sum() / avg_factor

        if self.ns == 3:
            d_angle_restrict = torch.sum(x, dim=-1)
            d_angle_target = torch.zeros_like(d_angle_restrict)
            loss_angle_restrict += (torch.abs(d_angle_restrict - d_angle_target) * weight).sum() / avg_factor

        return loss_angle_restrict

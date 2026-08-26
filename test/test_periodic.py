''' Check that myUnet(periodic=True) is shift-equivariant along longitude (width),
    and that periodic=False reproduces the original zero-padded behavior. '''

import torch
from wavediffusion.model_unet import myUnet


def build_model(periodic):
    meanx, stdx = torch.zeros(3), torch.ones(3)
    meanf, stdf = torch.zeros(14), torch.ones(14)
    model = myUnet(in_dim=32, in_ch=3, out_ch=3, ch=32, precond_ch=14,
                    scale=(meanx, stdx, meanf, stdf),
                    ch_mult=(1, 2), attn_resolutions=(), periodic=periodic)
    model.eval()
    return model


def shift_equivariance_error(model, shift):
    # shift must be a multiple of the total downsample stride so the
    # comparison isn't confounded by the strided conv/upsample's own
    # phase-dependence (see model_unet.py's Downsample/Upsample).
    x = torch.randn(2, 3, 32, 32)
    cond = torch.randn(2, 14, 32, 32)
    sigma = torch.tensor([1.0, 1.0])

    with torch.no_grad():
        out = model(x, sigma, cond=cond)

        x_shift = torch.roll(x, shifts=shift, dims=3)
        cond_shift = torch.roll(cond, shifts=shift, dims=3)
        out_shift = model(x_shift, sigma, cond=cond_shift)

    out_shift_back = torch.roll(out_shift, shifts=-shift, dims=3)
    return (out - out_shift_back).abs().max().item()


def test_periodic_model_is_shift_equivariant():
    torch.manual_seed(0)
    model = build_model(periodic=True)
    err = shift_equivariance_error(model, shift=4)
    assert err < 1e-4, f"periodic=True should be shift-equivariant in longitude, got max diff {err}"


def test_nonperiodic_model_is_not_shift_equivariant():
    torch.manual_seed(0)
    model = build_model(periodic=False)
    err = shift_equivariance_error(model, shift=4)
    assert err > 1e-2, f"periodic=False (zero-padded) should NOT be shift-equivariant, got max diff {err}"


if __name__ == '__main__':
    torch.manual_seed(0)
    for periodic in [False, True]:
        model = build_model(periodic)
        err = shift_equivariance_error(model, shift=4)
        print(f"periodic={periodic}: max diff after shift-roll-unroll = {err:.6g}")

    test_periodic_model_is_shift_equivariant()
    test_nonperiodic_model_is_not_shift_equivariant()
    print("OK")

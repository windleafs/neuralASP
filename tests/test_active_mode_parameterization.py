import torch

from physics.active_mode_parameterization import (
    build_active_mode_templates,
    load_shared_active_basis,
)


def synthetic_population_artifact(path):
    # 2 depths x 3 lateral modes = 6 candidate parameters.
    Gp = torch.diag(torch.tensor([6., 5., 4., 3., 2., 1.]))
    Ga = torch.diag(torch.tensor([3., 2.5, 2., 1.5, 1., 0.5]))
    phase_vh = torch.eye(6)
    amp_vh = torch.eye(6)
    torch.save({
        "Gp": Gp,
        "Ga": Ga,
        "phase_Vh": phase_vh,
        "amplitude_Vh": amp_vh,
        "depths_mm": [2.0, 6.0],
        "labels_phase": [
            "P:z2.0:k0", "P:z2.0:k1", "P:z2.0:k2",
            "P:z6.0:k0", "P:z6.0:k1", "P:z6.0:k2",
        ],
    }, path)


def test_load_active_basis_preserves_candidate_layout(tmp_path):
    path = tmp_path / "population.pt"
    synthetic_population_artifact(path)
    basis = load_shared_active_basis(path, rank=2, source="phase")
    assert basis["Vh"].shape == (2, 6)
    assert basis["n_lateral_modes"] == 3
    assert basis["depths_mm"] == [2.0, 6.0]
    torch.testing.assert_close(basis["Vh"], torch.eye(6)[:2])


def test_active_template_unit_integrals_match_first_candidates(tmp_path):
    path = tmp_path / "population.pt"
    synthetic_population_artifact(path)
    basis = load_shared_active_basis(path, rank=2, source="phase")
    dz = 1e-3
    tpl = build_active_mode_templates(
        basis, nz=10, nx=12, dz_m=dz, z0_m=0.0, pad=2)

    phase = tpl["phase_unit_ds"]
    amp = tpl["amplitude_unit_rate"]
    assert phase.shape == (2, 10, 12)
    assert amp.shape == (2, 10, 12)

    # Eigenvectors 0/1 pick the first two DCT modes at the 2-mm screen.
    # Their depth integral is one us / one Np times the unit-RMS lateral mode.
    phase_integrated_us = phase.sum(dim=1) * dz * 1e6
    amp_integrated_np = amp.sum(dim=1) * dz
    phase_rms = phase_integrated_us[:, 2:-2].square().mean(dim=-1).sqrt()
    amp_rms = amp_integrated_np[:, 2:-2].square().mean(dim=-1).sqrt()
    torch.testing.assert_close(phase_rms, torch.ones(2), atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(amp_rms, torch.ones(2), atol=1e-5, rtol=1e-5)
    assert tpl["depth_indices"][0] == 2


def test_balanced_basis_ignores_phase_amplitude_unit_scaling(tmp_path):
    path1 = tmp_path / "a.pt"
    path2 = tmp_path / "b.pt"
    synthetic_population_artifact(path1)
    p = torch.load(path1, weights_only=False)
    p["Gp"] = p["Gp"] * 1000.0
    p["Ga"] = p["Ga"] * 0.001
    torch.save(p, path2)
    a = load_shared_active_basis(path1, rank=3, source="balanced")["Vh"]
    b = load_shared_active_basis(path2, rank=3, source="balanced")["Vh"]
    # Eigenvector signs are arbitrary; compare projectors.
    torch.testing.assert_close(a.T @ a, b.T @ b, atol=1e-5, rtol=1e-5)

import numpy as np
import pytest

from common import build_meta, load_config
from data.l11_fullwave import validate_l11_sample
from scripts.generate_l11_ultrawave_raw import choose_anatomy, geometry_case, absorption_model


def test_eleven_angle_config():
    cfg=load_config("configs/l11_ultrawave_500_11angle.yaml"); meta=build_meta(cfg)
    assert len(meta.angles_deg)==11
    assert np.allclose(meta.angles_deg,np.linspace(-8,8,11))
    assert len(meta.freqs)==53
    assert len(meta.train_idx)==8 and meta.hold_idx.tolist()==[2,5,8]
    assert (cfg.train.n_train,cfg.train.n_val,cfg.train.n_test)==(400,50,50)


def test_anatomical_partition_no_leakage():
    candidates=[dict(z_index=2*i+20,score=.15+.03*np.sin(i/13)) for i in range(252)]
    train,val=choose_anatomy(candidates)
    assert len(train)==200 and len(val)==25
    assert len({x["z_index"] for x in train+val})==225
    assert min(abs(a["z_index"]-b["z_index"]) for a in train for b in val)>=10


def test_absorption_clock_and_geometry():
    case=geometry_case(); fit=absorption_model(case)
    assert fit["max_relative_error"]<.10
    assert case["dt"]<=min(fit["taus_s"])/8*1.001
    assert case["nt"]==24001
    assert np.isclose(case["end"],60e-6)


def test_required_backend_rejects_old_kwave():
    cfg=load_config("configs/l11_ultrawave_500_11angle.yaml"); meta=build_meta(cfg)
    with pytest.raises(ValueError,match="backend"):
        validate_l11_sample({"metadata":{"backend":"kwave"}},cfg,meta)

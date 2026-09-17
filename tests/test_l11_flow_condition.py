import importlib.util
from pathlib import Path
import torch
from common import _to_C

path=Path(__file__).resolve().parents[1]/'runs/l11_flow_branch_compare_20260916/code/prepare_l11_flow.py'
spec=importlib.util.spec_from_file_location('l11_condition',path)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def test_heldout_angle_is_not_used_and_point_focus_preserves_timing():
    fs=40e6;fc=7.5e6;tref=[2.259441e-6,.533333e-6,2.259441e-6]
    meta=_to_C({'train_idx':[0,1],'fs':fs,'xe_coords':[-.0003,-.0001,.0001,.0003],
               'angles_deg':[-8.,0.,8.],'t_ref_s':tref})
    t=torch.arange(2401)/fs
    rf=torch.zeros(3,4,2401)
    for a in range(3):
        th=torch.tensor(meta.angles_deg[a])*torch.pi/180
        for e,xe in enumerate(meta.xe_coords):
            delay=tref[a]+.02*th.cos()/1500+(.02**2+xe**2)**.5/1500
            rf[a,e]=torch.exp(-((t-delay)/.15e-6)**2)*torch.cos(2*torch.pi*fc*(t-delay))
    cx=torch.tensor([-.001,0.,.001]);cz=torch.tensor([.01,.02,.03])
    cond=module.make_condition(rf,meta,cx,cz,fc)
    assert cond.shape==(18,3,3)
    mag=(cond[2].square()+cond[3].square()).sqrt()
    assert mag[1,1]>10*torch.cat([mag[:,0],mag[:,2]]).max()
    changed=rf.clone();changed[2]=torch.randn_like(changed[2])*1e9
    assert torch.equal(cond,module.make_condition(changed,meta,cx,cz,fc))

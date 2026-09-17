import pytest
import torch
from common import _to_C
from models.external_sos import resample_sos,sos_to_delta_s


def test_physical_axes_and_slowness_reference():
    meta=_to_C({'x0':-.002,'z0':.001})
    grid=_to_C({'nx':5,'nz':7,'dx':.001,'dz':.0005})
    cx=torch.linspace(-.002,.002,3);cz=torch.linspace(.001,.004,4)
    c=1500+1000*cx[:,None]+2000*cz[None,:]
    actual=resample_sos(c,cx,cz,meta,grid)
    x=meta.x0+torch.arange(5)*grid.dx;z=meta.z0+torch.arange(7)*grid.dz
    expected=1500+1000*x[None,:]+2000*z[:,None]
    assert actual.shape==(7,5)
    assert torch.allclose(actual,expected,atol=2e-4)
    assert torch.allclose(sos_to_delta_s(c,cx,cz,meta,grid,1540),expected.reciprocal()-1/1540,atol=1e-10)


def test_no_silent_transpose_extrapolation_or_log_variable():
    meta=_to_C({'x0':0.,'z0':0.})
    grid=_to_C({'nx':3,'nz':4,'dx':.001,'dz':.001})
    cx=torch.arange(3)*.001;cz=torch.arange(4)*.001
    with pytest.raises(ValueError,match='shape'):
        resample_sos(torch.ones(4,3)*1500,cx,cz,meta,grid)
    with pytest.raises(ValueError,match='positive'):
        resample_sos(torch.zeros(3,4),cx,cz,meta,grid)
    with pytest.raises(ValueError,match='cover'):
        resample_sos(torch.ones(3,4)*1500,cx,cz,meta,_to_C({**grid,'nz':5}))

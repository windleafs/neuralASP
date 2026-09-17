"""Error-aware completion: pack500, validate, formal smoke, report then READY."""
import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path

PROJECT=Path(__file__).resolve().parents[1]
CONFIG=PROJECT/"configs/l11_ultrawave_500_11angle.yaml"
ROOT=Path("/data/zhuangyang/NumerialBreastPhantoms/l11_ultrawave_500_11angle")


def idle_smoke_gpu():
    deadline=time.monotonic()+1800
    while time.monotonic()<deadline:
        output=subprocess.check_output(["nvidia-smi","--query-gpu=index,memory.used,utilization.gpu","--format=csv,noheader,nounits"],text=True)
        for row in csv.reader(output.splitlines()):
            index,mem,util=map(int,row)
            if index in (1,2) and mem<512 and util<5:
                return index
        time.sleep(30)
    raise RuntimeError("GPU1/2 not idle for smoke; do not interfere with other work")


def command(*args,env=None):
    subprocess.run([sys.executable,*map(str,args)],cwd=PROJECT,env=env,check=True)


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--root",type=Path,default=ROOT)
    ap.add_argument("--config",type=Path,default=CONFIG); ap.add_argument("--timeout-hours",type=float,default=24.)
    args=ap.parse_args(); root=args.root
    manifest=json.loads((root/"index.json").read_text()); expected=len(manifest["samples"])
    if expected!=500: raise RuntimeError("expected500")
    deadline=time.monotonic()+args.timeout_hours*3600; previous=-1
    while time.monotonic()<deadline:
        errors=list((root/"raw").glob("*.error.txt"))
        if errors: raise RuntimeError("raw generation errors: "+"; ".join(p.name+": "+p.read_text().strip() for p in errors))
        n=sum((root/r["raw_path"]).exists() for r in manifest["samples"])
        if n!=previous: print(f"raw complete {n}/{expected}",flush=True); previous=n
        if n==expected: break
        time.sleep(30)
    else: raise TimeoutError(f"raw incomplete {previous}/500")
    command(PROJECT/"scripts/pack_l11_kwave_dataset.py","--mode","pack","--config",args.config,"--root",root)
    command(PROJECT/"scripts/pack_l11_kwave_dataset.py","--mode","validate","--config",args.config,"--root",root)
    gpu=idle_smoke_gpu(); env=os.environ.copy(); env["CUDA_VISIBLE_DEVICES"]=str(gpu)
    smoke_name="l11_ultrawave_500_11angle_smoke"; run=PROJECT/"runs"/smoke_name
    command(PROJECT/"train.py","--config",args.config,"--stage","all","--steps",1,"--out",smoke_name,env=env)
    command(PROJECT/"eval.py","--config",args.config,"--ckpt",run/"joint.pt","--out",run/"eval",env=env)
    command(PROJECT/"scripts/report_l11_ultrawave_dataset.py","--config",args.config,"--smoke-run",run)
    result=dict(ok=True,raw_samples=500,packed_samples=500,formal_smoke_ok=True,eval_smoke_samples=50,
                completed=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
    tmp=root/"READY.tmp"; tmp.write_text(json.dumps(result,indent=2)+"\n"); os.replace(tmp,root/"READY")
    print(json.dumps(result,indent=2),flush=True)


if __name__=="__main__":
    try: main()
    except Exception as exc:
        ROOT.mkdir(parents=True,exist_ok=True)
        (ROOT/"FINALIZATION_FAILED.txt").write_text(f"{type(exc).__name__}: {exc}\n")
        raise

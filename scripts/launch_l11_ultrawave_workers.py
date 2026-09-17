"""Start two resumable raw workers only after explicit pilot gates passed."""
import csv
import json
import os
import subprocess
import sys
from pathlib import Path

PROJECT=Path(__file__).resolve().parents[1]
ROOT=Path("/data/zhuangyang/NumerialBreastPhantoms/l11_ultrawave_500_11angle")
COMPILER=Path("/data/zhuangyang/nvidia/hpc_sdk/Linux_x86_64/23.7/compilers/bin")


def main():
    if (ROOT/"READY").exists(): raise RuntimeError("dataset alreadyREADY")
    for name in ("pilot_smoke_m.json","pilot_smoke_joint.json"):
        gate=json.loads((ROOT/name).read_text())
        if not gate["ok"]: raise RuntimeError(f"failed pilot {name}")
    if not (ROOT/"PILOT_VISUAL_APPROVED").exists():
        raise RuntimeError("pilot natural RF B-mode must be visually reviewed before batch")
    if not (COMPILER/"nvc++").is_file(): raise RuntimeError("nvc++ missing")
    output=subprocess.check_output(["nvidia-smi","--query-gpu=index,memory.used,utilization.gpu","--format=csv,noheader,nounits"],text=True)
    inventory={int(r[0]):[int(r[1]),int(r[2])] for r in csv.reader(output.splitlines())}
    for gpu in (1,2):
        mem,util=inventory[gpu]
        if mem>512 or util>5: raise RuntimeError(f"GPU{gpu} isbusy")
    # Detect our already-running workers, rather than launching duplicates.
    existing=subprocess.run(["pgrep","-af","[g]enerate_l11_ultrawave_raw.py --mode worker|[f]inalize_l11_ultrawave_dataset.py"],capture_output=True,text=True)
    if existing.stdout.strip(): raise RuntimeError("worker/finalizer alreadyrunning: "+existing.stdout)
    processes={}
    for worker,gpu in enumerate((1,2)):
        env=os.environ.copy(); env["CUDA_VISIBLE_DEVICES"]=str(gpu)
        env["PATH"]=str(COMPILER)+os.pathsep+env["PATH"]; env["NVCOMPILER_ACC_NOTIFY"]="0"
        log=open(ROOT/f"worker{worker}.log","a",buffering=1)
        p=subprocess.Popen([sys.executable,str(PROJECT/"scripts/generate_l11_ultrawave_raw.py"),"--mode","worker","--workers","2","--worker-id",str(worker)],
                           cwd=PROJECT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        log.close(); processes[f"worker{worker}"]=dict(pid=p.pid,gpu=gpu)
    log=open(ROOT/"finalizer.log","a",buffering=1)
    p=subprocess.Popen([sys.executable,str(PROJECT/"scripts/finalize_l11_ultrawave_dataset.py")],cwd=PROJECT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    log.close(); processes["finalizer"]=dict(pid=p.pid)
    (ROOT/"processes.json").write_text(json.dumps(processes,indent=2)+"\n")
    print(json.dumps(processes,indent=2))


if __name__=="__main__": main()

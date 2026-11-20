from pathlib import Path
import tempfile, numpy as np, importlib.util
p=Path(__file__).resolve().parents[1]/'scripts/01_source_contract_forensic.py'
s=importlib.util.spec_from_file_location('m',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
with tempfile.TemporaryDirectory() as td:
    q=Path(td)/'x.npy';np.save(q,np.zeros((7,3),dtype=np.float32));meta=m.binary_meta(q)
    assert meta['rows']==7 and meta['shape']=='7x3' and meta['dtype']=='float32'
print('[PASS] Stage08A-R3 binary metadata selftest')

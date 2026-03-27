# t23d text-to-glb 

## what it does
- input prompt
- stage 1: qwen-image generates png
- stage 2: trellis2 generates glb

## prerequisites

```bash
. ./setup.sh --new-env --basic --flash-attn --nvdiffrast --nvdiffrec --cumesh --o-voxel --flexgemm
```

```bash
bash download.sh
```


## run
seed is fixed to official Qwen-Image seed `42` in code (no cli override).

single command:
```bash
python run_text2glb.py "a red racing car toy"
```

interactive mode:
```bash
python run_text2glb.py
```

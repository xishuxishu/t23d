# t23d text-to-glb 

## what it does
- input prompt
- stage 1: qwen-image generates png
- stage 2: trellis2 generates glb

## prerequisites

Device Requirements with reference to https://github.com/microsoft/TRELLIS.2/tree/main

```bash
. ./setup.sh --new-env --basic --flash-attn --nvdiffrast --nvdiffrec --cumesh --o-voxel --flexgemm
```

```bash
bash download.sh
```


## run

```bash
python run_text2glb.py "a yellow taxi"
```



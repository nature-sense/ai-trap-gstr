# models/

Place model files here before building.

## Available models

| Directory | Input size | Backend | Notes |
|---|---|---|---|
| `yolo11n-320/` | 320×320 | ncnn | Lightweight — recommended for Pi 5 real-time inference |

Each model directory contains two files:

| File | Description |
|---|---|
| `model.ncnn.param` | Network architecture (~50 KB) |
| `model.ncnn.bin` | Weights (~5.6 MB) |

## Config

Model paths and input size are set in `trap_config.toml` (in the build directory):

```toml
[model]
param       = "../firmware/models/yolo11n-320/model.ncnn.param"
bin         = "../firmware/models/yolo11n-320/model.ncnn.bin"
width       = 320
height      = 320
num_classes = 1
format      = "anchor_grid"
```

Paths are relative to the working directory the binary is run from,
or use absolute paths for deployment.

## Adding a new model

1. Create a subdirectory named `<modelname>-<size>/` (e.g. `yolo11n-640/`)
2. Place `model.ncnn.param` and `model.ncnn.bin` inside it
3. Update `trap_config.toml` with the new paths, width, height, and format

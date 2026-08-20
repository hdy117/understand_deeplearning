# 推理输出

`dog_predictions.png` 由下面的真实命令生成：

```bash
.venv/Scripts/python.exe -m resnet50.pretrained_inference \
  --image resnet50/data/dog.jpg \
  --output resnet50/outputs/dog_predictions.png \
  --top-k 5 \
  --device cpu
```

该 PNG 左侧显示输入图像，右侧显示官方 `IMAGENET1K_V2` 权重的 Top-5 probability bars。

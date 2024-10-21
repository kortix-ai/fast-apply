# Deploy model on Fireworks

1. Clone model 16bit from HuggingFace : 
```
mkdir model && cd model
git-lfs install
git clone https://huggingface.co/Kortix/FastApply-7B-v1.0
```

2. Upload model to Fireworks :
```
firectl create model FastApply-7B model/FastApply-7B-v1.0
```

3. On-demand deploy : 
```
firectl create deployment "accounts/.../models/..." --scale-to-zero-window 5m --long-prompt  --wait
```

Same for 1.5B : `https://huggingface.co/Kortix/FastApply-1.5B-v1.0`



## Delete deployment

```
firectl list deployed-models
```

```
firectl delete deployment <id>
```
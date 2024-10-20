❯ firectl create deployment "accounts/marko-1d84ff/models/fast-apply-v16-1p5b-instruct" --scale-to-zero-window 5m  --accelerator-type="NVIDIA_H100_80GB" --accelerator-count=2 --long-prompt --wait



❯  firectl list deployed-models


✔ marko-1d84ff
NAME                                   MODEL                                                      DEPLOYMENT ID  STATE     DEFAULT  PUBLIC  CREATE TIME
70b-v13-ec34fde8                       accounts/marko-1d84ff/models/70b-v13                       serverless     DEPLOYED  true     false   2024-10-14 07:44:21
fast-apply-v16-1p5b-instruct-a3c03f21  accounts/marko-1d84ff/models/fast-apply-v16-1p5b-instruct  790c6851       DEPLOYED  true     false   2024-10-18 19:25:36

Page 1 of 1
Total size: 2


❯ firectl delete deployment 790c6851
✔ marko-1d84ff

# 1B
# firectl create model 1b-v12 1B/ --base-model accounts/fireworks/models/llama-v3p2-1b-instruct -a     marko-1d84ff


# 70B
firectl create model 70b-v13 70B/ --base-model accounts/fireworks/models/llama-v3p1-70b-instruct -a     marko-1d84ff

# 1B-v13
# firectl create model 1b-v13 1B-v13/ --base-model accounts/fireworks/models/llama-v3p2-1b-instruct -a     marko-1d84ff

# 8B
# firectl create model 8b-v12 8B/ --base-model accounts/fireworks/models/llama-v3p1-8b-instruct -a     marko-1d84ff


# firectl create model 8b-v13-2 8B-v13-2/ --base-model accounts/fireworks/models/llama-v3p1-8b-instruct -a     marko-1d84ff


# Spec : 8B
# firectl create model 8b-v13-2-spec3 8B-v13-2/ \
  # --base-model accounts/fireworks/models/llama-v3p1-8b-instruct \
  # -a     marko-1d84ff \
  # --default-draft-model accounts/marko-1d84ff/models/1b-v13-2 \
  # --default-draft-token-count 10

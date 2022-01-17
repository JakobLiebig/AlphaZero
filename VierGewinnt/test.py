import numpy as np
import torch

a = torch.tensor([1, 2, 3])
b = torch.tensor([1, 2, 3])

result_a = torch.stack([a, b])
result_b = torch.concat([a, b], 1)

print(result_a)
print(result_b)
import sys

sys.path.append("././")

# from pack.test import test
from barriertransformer.barrier_generate import (
    get_min_max,
    extract_barrier,
    generate_barrier,
)

cen = [0, 0, 0]
len = [1.25, 0.5, 0.9]

# min, max = get_min_max(cen, len)
# print(min, max)

mi, ma = generate_barrier()
print(mi, ma)

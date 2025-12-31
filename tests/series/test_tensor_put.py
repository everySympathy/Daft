import ray
import numpy as np
import time

data = np.random.rand(10000, 10000) # 约 800MB

# 错误示范：在循环里 put
for _ in range(5):
    # 每次循环，Ray 都会把这 800MB 数据重新拷贝进 Object Store
    # 尽管 data 没变，但 Ray 不会自动进行内容哈希去重
    time1 = time.time()
    ref = ray.put(data)
    time2 = time.time()
    print(f"put time: {time2 - time1}")
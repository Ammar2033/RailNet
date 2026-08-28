import numpy as np
routes=np.array([[1,0],[2,0]],dtype=np.int16)
try:
    cnt=np.bincount((routes-1).reshape(-1), minlength=5)
    print(cnt)
except Exception as e:
    print("error",e)

# fixed
mask=routes>0
cnt2=np.bincount((routes[mask]-1), minlength=5)
print("fixed",cnt2)

# also test greedy sparse scenario after fix

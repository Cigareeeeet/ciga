import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#%matplotlib inline

def true_function(x):
    y=math.sin(math.pi * x * 0.8) * 10
    return y

elements_size=100

Yarray=np.ndarray(elements_size)
#xarray=np.random.rand(elements_size)
#xarray=np.zeros(elements_size)
xarray=np.linspace(-1,1,elements_size)
xarray2=np.random.uniform(-1.0,1.0,20)

k=0
for i in xarray:
    y=true_function(i)
    Yarray[k]=y#20この真値
    k=k+1

k=0
Yarray2=np.ndarray(20)

for i in xarray2:
    y=true_function(i)
    Yarray2[k]=y#20この真値
    k=k+1

array1_2=np.column_stack([xarray2,Yarray2])

df=pd.DataFrame(array1_2,columns=["観測点","真値点"])
df.plot.scatter(x="観測点",y="真値点")

plt.plot(xarray,Yarray,label="sin")
plt.legend()
plt.savefig("ex1-2.png")
plt.show()


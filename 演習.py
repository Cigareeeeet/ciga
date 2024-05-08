import math
import numpy as np
import matplotlib.pyplot as plt

#%matplotlib inline

def true_function(x):
    y=math.sin(math.pi * x * 0.8) * 10
    return y

elements_size=1000

Yarray=np.ndarray(elements_size)
#xarray=np.random.rand(elements_size)
#xarray=np.zeros(elements_size)
xarray=np.linspace(-1,1,elements_size)
k=0
for i in xarray:
    y=true_function(i)
    Yarray[k]=y
    k=k+1

#print(Yarray)

plt.plot(xarray,Yarray,label="sin")
plt.legend()
plt.savefig("ex1-1.png")
plt.show()


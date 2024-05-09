import math
import numpy as np
import matplotlib.pyplot as plt

#%matplotlib inline

def true_function(x):
    y=math.sin(math.pi * x * 0.8) * 10
    return y

elements_size=20

Yarray=np.ndarray(elements_size)
#xarray=np.random.rand(elements_size)
#xarray=np.zeros(elements_size)
#xarray=np.linspace(-1,1,elements_size)
xarray=np.random.uniform(-1.0,1.0,20)
print(xarray)
k=0
for i in xarray:
    y=true_function(i)
    Yarray[k]=y#20この真値
    k=k+1
print(xarray[0])

array1_2=np.column_stack([xarray,Yarray])
print(np.shape(array1_2))
print(array1_2)




#print(Yarray)

plt.plot(xarray,Yarray,label="sin")
plt.legend()
plt.savefig("ex1-2.png")
plt.show()


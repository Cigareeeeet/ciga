import numpy
n=1000
list=[i for i in range(n+1)]
list[1]=0
a=2
while a <= numpy.sqrt(n):
    if list[a]==0:
        pass
    else:
        for k in list: #kは1-1000
            if list[k]!=a and list[k]%a==0:
                list[k]=0
            else:
                pass
    a=a+1

for b in list:
    if list[b]!=0:
        print(list[b],end=", ")

    else:
        pass
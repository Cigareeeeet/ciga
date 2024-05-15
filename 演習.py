import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#%matplotlib inline

def true_function(x):
    y=math.sin(math.pi * x * 0.8) * 10
    return y

def makeArray(array,elements_size): #第一変数にxとなる配列、第二変数に配列サイズとなる変数
    k=0
    Yarray=np.ndarray(elements_size)
    for i in array:
        y=true_function(i)
        Yarray[k]=y#20この真値
        k=k+1
    return Yarray

def makeNoize(array): #配列を受け取って、ノイズをかけた配列をわたす
    noize=(np.random.normal(loc=0.0, scale=2.0, size=20))/2
    NoizeArray=array+noize
    return NoizeArray

def save_dataframe(df): #fileに保存する
    df.to_csv("dataset1.tsv",sep='\t', index=False) #dataframeをtsv形式で保存

def Read_files(): #fileを読んで出力する
    df1 = pd.read_csv("dataset1.tsv",sep="\t")
    return df1

def data_plot(df,xarray,yarray):
    ax=df.plot.scatter(x="観測点",y="真値点",label="true value",c="g")
    df.plot.scatter(x="観測点",y="観測値",c="r",ax=ax, label="observed value")
    
    plt.plot(xarray,Yarray,label="sin")
    plt.legend()
    #plt.savefig("ex1-3.png")
    plt.show()



elements_size=20
xarray=np.linspace(-1,1,elements_size)
xarray2=np.random.uniform(-1.0,1.0,elements_size)
Yarray=makeArray(xarray,elements_size)
Yarray2=makeArray(xarray2,elements_size)
Yarray3=makeNoize(Yarray2)
array1_2=np.column_stack([xarray2,Yarray2,Yarray3])#配列を結合
df=pd.DataFrame(array1_2,columns=["観測点","真値点","観測値"])#dataframe作成
data_plot(df,xarray,Yarray)
    


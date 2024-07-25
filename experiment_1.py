from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import time
#from imblearn.ensemble import BalancedBaggingClassifier, BalancedRandomForestClassifier
import pickle
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
from sklearn.model_selection import validation_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import plot_tree
from sklearn.decomposition import PCA

#csvファイルからデータのロード
file_path="winequality-white-re.csv"
df = pd.read_csv(file_path,encoding="shift-jis")


def DecisionTree():
    #x=pd.DataFrame(df[["fixed acidity","volatile acidity","citric acid","residual sugar","chlorides","free sulfur dioxide","total sulfur dioxide","density","pH","sulphates","alcohol"]])
    #x=pd.DataFrame(df[["residual sugar","free sulfur dioxide","total sulfur dioxide","alcohol"]])#pca後2
    x=pd.DataFrame(df[["fixed acidity","volatile acidity","chlorides","total sulfur dioxide","density","alcohol"]])
    y=pd.DataFrame(df[["quality"]])

    #データを学習用とテスト用に分ける
    X_train,X_test,Y_train,Y_test = train_test_split(x,y, test_size=0.3, shuffle=True, random_state=3, stratify=y)
    Y_train=np.reshape(Y_train,(-1))
    Y_test=np.reshape(Y_test,(-1))

    #k近傍
    
    model = KNeighborsClassifier(
        n_neighbors=65,
        weights="uniform",
        algorithm="auto",
        metric="canberra",#"euclidean","manhattan","chebyshev","minkowski","hamming","canberra","braycurtis"                  
        p=1
        ) 


    """
    # モデル学習
    model.fit(x, y)
    
    # モデル保存
    with open('emotion_model8400re2.pickle', mode='wb') as f:
        pickle.dump(model, f)
    """

    #検証曲線の時の
    #param_range=[4,8,12,16,20,24]
    param_range=[5,10,15,20,25,30,35,40]
    #param_range=[1,2]
    train_scores, test_scores = validation_curve(
        estimator=model,
        X=x, y=np.reshape(y,(-1)),
        param_name="n_neighbors",
        param_range=param_range, cv=10)

    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)
    plt.figure(figsize=(8, 6))
    plt.plot(param_range, train_mean, marker='o', label='Train accuracy')
    plt.fill_between(param_range, train_mean + train_std, train_mean - train_std, alpha=0.2)
    plt.plot(param_range, test_mean, marker='s', linestyle='--', label='Validation accuracy')
    plt.fill_between(param_range, test_mean + test_std, test_mean - test_std, alpha=0.2)
    plt.grid()
    plt.xscale('log')
    plt.title('Validation curve (wminkowski)', fontsize=16)
    plt.xlabel('n_neighbors', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.legend(fontsize=12)
    plt.ylim([0.6, 1.05])
    plt.show()

    """"""
    model.fit(X_train,Y_train)
    #モデル評価
    Y_pred_tree=model.predict(X_test)
    print(f'正解率: {accuracy_score(Y_test, Y_pred_tree)}')


def doPCA():
    dfs = df.iloc[:, :-1].apply(lambda x:(x-x.mean())/x.std(), axis=0)
    print(dfs.head())

    #主成分分析の実行
    pca = PCA()
    pca.fit(dfs)
    # データを主成分空間に写像
    feature = pca.transform(dfs)
    pd.DataFrame(feature, columns=["PC{}".format(x + 1) for x in range(len(dfs.columns))]).head()

    plt.figure(figsize=(6, 6))
    plt.scatter(feature[:, 0], feature[:, 1], alpha=0.8, c=list(df.iloc[:, 0]))
    plt.grid()
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.show()
    from pandas import plotting 
    plotting.scatter_matrix(pd.DataFrame(feature, 
                            columns=["PC{}".format(x + 1) for x in range(len(dfs.columns))]), 
                            figsize=(8, 8), c=list(df.iloc[:, 0]), alpha=0.5) 
    plt.show()

    # 寄与率
    print("---------寄与率---------")
    print(pd.DataFrame(pca.explained_variance_ratio_, index=["PC{}".format(x + 1) for x in range(len(dfs.columns))]))
    
    # 累積寄与率を図示する
    import matplotlib.ticker as ticker
    plt.gca().get_xaxis().set_major_locator(ticker.MaxNLocator(integer=True))
    plt.plot([0] + list( np.cumsum(pca.explained_variance_ratio_)), "-o")
    plt.xlabel("Number of principal components")
    plt.ylabel("Cumulative contribution rate")
    plt.grid()
    plt.show()

    # PCA の固有値
    print("---------固有値---------")
    print(pd.DataFrame(pca.explained_variance_, index=["PC{}".format(x + 1) for x in range(len(dfs.columns))]))

def presyori():
    import pandas as pd
    import umap
    import seaborn as sns
    import matplotlib.pyplot as plt


    # 列名の表示
    print(df.columns)


    # 特徴量とターゲットの分離（適切な列名を使用）
    X = df.drop('quality', axis=1)  # 正しいターゲット列名を使用
    y = df['quality']  # 正しいターゲット列名を使用

    # UMAPの適用
    reducer = umap.UMAP(random_state=42)
    embedding = reducer.fit_transform(X)

    # 結果のデータフレーム化
    embedding_df = pd.DataFrame(embedding, columns=['UMAP1', 'UMAP2'])
    embedding_df['quality'] = y

    # 可視化
    plt.figure(figsize=(12, 8))
    sns.scatterplot(x='UMAP1', y='UMAP2', hue='quality', palette='viridis', data=embedding_df, legend='full')
    plt.title('UMAP projection of the Wine quality dataset')
    plt.show()

DecisionTree()
#doPCA()
#presyori()




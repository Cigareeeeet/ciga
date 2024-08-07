from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.model_selection import validation_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm

max_score=0
file_path="winequality-white-re.csv"
df = pd.read_csv(file_path,encoding="shift-jis")

#そのままの特徴量
x=pd.DataFrame(df[["fixed acidity","volatile acidity","citric acid","residual sugar","chlorides","free sulfur dioxide","total sulfur dioxide","density","pH","sulphates","alcohol"]])
#pca後の特徴量
#x=pd.DataFrame(df[["residual sugar","free sulfur dioxide","total sulfur dioxide","alcohol"]])
#umap後の特徴量
#x=pd.DataFrame(df[["fixed acidity","volatile acidity","chlorides","total sulfur dioxide","density","alcohol"]])

y=pd.DataFrame(df[["quality"]])
#XとYを学習データとテストデータに分割
X_train,X_test,Y_train,Y_test = train_test_split(x,y, test_size=0.3, shuffle=True, random_state=3, stratify=y)
Y_train=np.reshape(Y_train,(-1))
Y_test=np.reshape(Y_test,(-1))

#model = RandomForestClassifier()

RFC_grid = {RandomForestClassifier(): {"n_estimators": [i for i in range(1, 20)],
                                       "criterion": ["gini", "entropy"],
                                       "max_depth":[i for i in range(1, 10)],
                                       "random_state": [i for i in range(0, 3)]
                                      }}

#ランダムフォレストの実行
for model, param in tqdm(RFC_grid.items()):
    clf = GridSearchCV(model, param)
    clf.fit(X_train, Y_train)
    pred_y = clf.predict(X_test)
    score = f1_score(Y_test, pred_y, average="micro")

    if max_score < score:
        max_score = score
        best_param = clf.best_params_
        best_model = model.__class__.__name__

print("サーチ方法:グリッドサーチ")
print("ベストスコア:{}".format(max_score))
print("モデル:{}".format(best_model))
print("パラメーター:{}".format(best_param))
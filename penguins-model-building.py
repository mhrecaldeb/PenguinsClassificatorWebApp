import pandas as pd
# se carga el data set de un archivo ya limpio
# E:\Proyectos de Python\AmazonWebScrapping\boxofficehome\boxoffice
path = ""
penguins = pd.read_csv(path + "penguins_cleaned.csv")

# Ordinal feature encoding
# https://www.kaggle.com/pratik1120/penguin-dataset-eda-classification-and-clustering
# se define la variable de datos y las columnas target y encode
df = penguins.copy()
target = 'species'
encode = ['sex','island']

# ordinal feature encoding de las columnas encode
for col in encode:
    dummy = pd.get_dummies(df[col], prefix=col)
    df = pd.concat([df,dummy], axis=1)
    del df[col]

# se codifica el targed con un mapa de variables con la función
target_mapper = {'Adelie':0, 'Chinstrap':1, 'Gentoo':2}
def target_encode(val):
    return target_mapper[val]

df['species'] = df['species'].apply(target_encode)

# Separating X and y
X = df.drop('species', axis=1)
Y = df['species']

# Build random forest model
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier()
clf.fit(X, Y)

# Saving the model
import pickle
pickle.dump(clf, open(path + 'penguins_clf.pkl', 'wb'))
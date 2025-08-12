import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.pipeline import Pipeline

from utils import db_connect

engine = db_connect()
df = pd.read_sql_table('winequality_red', engine)
df = df.rename(columns={'quality': 'label'})

def get_label(q):
    if q <= 5: return 0
    if q == 6: return 1
    return 2
df['label'] = df['label'].apply(get_label)

X = df.drop('label', axis=1)
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('knn', KNeighborsClassifier(n_neighbors=5))
])

pipeline.fit(X_train, y_train)
y_pred_base = pipeline.predict(X_test)
print("--- Evaluación del Modelo Base (k=5) ---")
print(f"Accuracy: {accuracy_score(y_test, y_pred_base):.4f}")
print("\nMatriz de Confusión:\n", confusion_matrix(y_test, y_pred_base))
print("\nInforme de Clasificación:\n", classification_report(y_test, y_pred_base))

k_range = range(1, 21)
accuracy_scores = []

for k in k_range:
    pipeline.set_params(knn__n_neighbors=k)
    pipeline.fit(X_train, y_train)
    y_pred_k = pipeline.predict(X_test)
    accuracy_scores.append(accuracy_score(y_test, y_pred_k))

best_k = k_range[np.argmax(accuracy_scores)]
print(f"\nEl mejor valor de k es: {best_k} con una precisión de {accuracy_scores[best_k-1]:.4f}")

os.makedirs('reports/figures', exist_ok=True)
plt.figure(figsize=(10, 6))
plt.plot(k_range, accuracy_scores, marker='o', linestyle='--')
plt.title('Accuracy vs. Valor de k')
plt.xlabel('Valor de k')
plt.ylabel('Accuracy')
plt.xticks(k_range)
plt.grid(True)
plt.savefig('reports/figures/accuracy_vs_k.png')
plt.show()

best_model = Pipeline([
    ('scaler', StandardScaler()),
    ('knn', KNeighborsClassifier(n_neighbors=best_k))
])
best_model.fit(X_train, y_train)

os.makedirs('models', exist_ok=True)
joblib.dump(best_model, 'models/knn_model.pkl')
print("\nModelo guardado en 'models/knn_model.pkl'")

def predict_wine_quality(values):
    model = joblib.load('models/knn_model.pkl')
    pred = model.predict([values])[0]
    
    mapping = {0: "baja calidad 😕", 1: "calidad media 🍷", 2: "alta calidad 🌟"}
    return f"Este vino probablemente sea de {mapping.get(pred, 'calidad desconocida')}"

ejemplo_vino = df.drop('label', axis=1).iloc[0].tolist()
print("\nPredicción con un ejemplo:")
print(predict_wine_quality(ejemplo_vino))

y_pred_final = best_model.predict(X_test)
print("\n--- Evaluación Final del Mejor Modelo (k=%d) ---" % best_k)
print(confusion_matrix(y_test, y_pred_final))
print(classification_report(y_test, y_pred_final))
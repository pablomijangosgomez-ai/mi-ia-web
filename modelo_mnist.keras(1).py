!pip install shap

import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report # Se eliminó la coma sobrante
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd

# Instala shap si no lo tienes (necesario en Colab)
try:
    import shap
except ImportError:
    !pip install shap
    import shap

# --- CONFIGURACIÓN ---
DATOS_ENTRENAMIENTO = 1000
NUM_EPOCHS = 10

# Carga de datos MNIST
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Normalización y expansión de dimensiones
x_train_exp = np.expand_dims(x_train / 255.0, axis=-1)
x_test_exp = np.expand_dims(x_test / 255.0, axis=-1)

# Definición del modelo [cite: 42, 44, 45]
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28, 1)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

print("Modelo preparado y listo para entrenar.")

history = model.fit(
x_train_exp[:DATOS_ENTRENAMIENTO],
y_train[:DATOS_ENTRENAMIENTO],
epochs=NUM_EPOCHS,
validation_split=0.2,
verbose=1
)

preds = np.argmax(model.predict(x_test_exp), axis=1)
plt.figure(figsize=(8,6))
sns.heatmap(confusion_matrix(y_test, preds), annot=True, fmt='d', cmap='Blues')
plt.title("Mapa de aciertos y errores (Matriz de Confusión)") # Todo en una línea
plt.show()

print("\nREPORTE FINAL:")
print(classification_report(y_test, preds))

plt.figure(figsize=(10, 4))
plt.plot(history.history['accuracy'],
label='Entrenamiento', marker='o')
plt.plot(history.history['val_accuracy'],
label='Validación', marker='s')
plt.title("¿Cómo está aprendiendo mi IA?")
plt.gca().xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
plt.legend()
plt.grid(alpha=0.3)
plt.show()

plt.figure(figsize=(10,4))
plt.plot(history.history['accuracy'], label='Entrenamiento', marker='o')
plt.plot(history.history['val_accuracy'], label='Validación', marker='s')
plt.title("¿Cómo está aprendiendo mi IA?")

# Se corrigió el error en 'integer=True'
plt.gca().xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

plt.legend()
plt.grid(alpha=0.3)
plt.show()

idx = 0
explainer = shap.DeepExplainer(model, x_train_exp[:100])
shap_values = explainer.shap_values(x_test_exp[idx:idx+1])
print(f"Número real: {y_test[idx]}")
shap.image_plot(shap_values, x_test_exp[idx:idx+1])

# Detectar imágenes que eran "0" pero la IA falló
idx_errores = np.where((y_test == 0) & (preds != 0))[0]

if len(idx_errores) > 0:
    sel = idx_errores[:3]  # Seleccionamos los primeros 3 ejemplos para el análisis
    shap_vals_err = explainer.shap_values(x_test_exp[sel])
    shap.image_plot(shap_vals_err, x_test_exp[sel])

    for i, s in enumerate(sel):
        # Se corrigió el salto de línea en el print
        print(f"Ejemplo {i+1}: Real 0 | IA dijo {preds[s]}")
else:
    print("No se encontraron errores para el número 0.")

# 1. Predicciones y comparativa
probs = model.predict(x_test_exp, verbose=0)
conf, preds = np.max(probs, axis=1), np.argmax(probs, axis=1)
correctos = (preds == y_test)

# 2. Cálculo de métricas por umbral
umbrales = [0.99, 0.95, 0.90, 0.80, 0.70, 0.50, 0.20]
res = []

for u in umbrales:
    pasan = conf >= u
    n = np.sum(pasan)
    aciertos = np.sum(correctos[pasan]) if n > 0 else 0
    # Cálculo de: Umbral, Aciertos%, Errores%, Descartes%, Precisión_IA%
    res.append([u*100, (aciertos/100), (n-aciertos)/100,
                (10000-n)/100, (aciertos/n*100 if n>0 else 100)])

df = pd.DataFrame(res, columns=['U', 'A', 'F', 'D', 'P'])

# 3. Gráfica optimizada
fig, ax1 = plt.subplots(figsize=(10, 5))
x = range(len(df))

# Barras apiladas
ax1.bar(x, df['A'], color='#2ecc71', label='Aciertos')
ax1.bar(x, df['F'], bottom=df['A'], color='#e74c3c', label='Errores')
ax1.bar(x, df['D'], bottom=df['A']+df['F'], color='#dfe6e9', label='Duda (Humano)')

# Línea de calidad (eje secundario)
ax2 = ax1.twinx()
ax2.plot(x, df['P'], color='#2980b9', marker='o', label='Calidad IA')

# Configuración de etiquetas y título
ax1.set_xticks(x)
ax1.set_xticklabels([f"{int(i)}%" for i in df['U']])
ax1.set_title("SEGURIDAD: PRECISIÓN VS AUTOMATIZACIÓN")
ax1.set_ylabel("Porcentaje de la base de datos")
ax2.set_ylabel("Precisión de la IA (%)")
ax1.legend(loc='upper left', bbox_to_anchor=(1.15, 1))

plt.tight_layout()
plt.show()

# Guardamos el modelo entrenado en formato (.keras)
model.save('modelo_mnist.keras')

#Es necesario guardar el módelo en vez de entrenarlo cada vez, porque al entrenarlo tiene varios problemas, como que el consumo es mayor o que tarda mucho timepo

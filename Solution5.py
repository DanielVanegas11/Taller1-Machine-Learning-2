from sklearn.linear_model import LogisticRegression
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Cargar el conjunto de datos MNIST
mnist = fetch_openml('mnist_784')

# Preparar los datos para el entrenamiento
X = mnist.data
y = mnist.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Escalar los datos de entrada
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Seleccionar solo los datos correspondientes a los dígitos 0 y 8
X_train_08 = X_train[(y_train == '0') | (y_train == '8')]
y_train_08 = y_train[(y_train == '0') | (y_train == '8')]
X_test_08 = X_test[(y_test == '0') | (y_test == '8')]
y_test_08 = y_test[(y_test == '0') | (y_test == '8')]

# Crear y entrenar el modelo de regresión logística
model = LogisticRegression(random_state=42, max_iter=1000)
model.fit(X_train_08, y_train_08)

# Evaluar el rendimiento del modelo en los datos de prueba
accuracy = model.score(X_test_08, y_test_08)

# Imprimir el mensaje de precisión del modelo
print(f"La precisión de la línea de base es: {accuracy:.2f}")
print("The performance of the base model is quite good since it can distinguish with great accuracy between the numbers 0 and 8 with an accuracy of {:.2f}%.".format(accuracy*100))
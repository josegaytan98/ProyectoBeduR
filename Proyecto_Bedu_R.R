# Instala paquetes si no los tienes
if(!require(ggplot2)) install.packages("ggplot2")
if(!require(dplyr)) install.packages("dplyr")
if(!require(readr)) install.packages("readr")
if(!require(caret)) install.packages("caret")
if(!require(randomForest)) install.packages("randomForest")
if(!require(nnet)) install.packages("nnet")

# Carga los paquetes
library(ggplot2)
library(dplyr)
library(readr)
library(caret)
library(randomForest)
library(nnet)

# Cargar los datos
data <- read_csv("seattle-weather.csv")

# Resumen estadístico
summary(data)

# Ver estructura de los datos
str(data)

# Distribución de la variable objetivo
table(data$weather)

# Visualización de las variables
ggplot(data, aes(x = precipitation)) + geom_histogram(binwidth = 0.5) + labs(title = "Distribución de Precipitation", x = "Precipitation", y = "Frecuencia")
ggplot(data, aes(x = temp_max)) + geom_histogram(binwidth = 2) + labs(title = "Distribución de Temp Max", x = "Temp Max", y = "Frecuencia")
ggplot(data, aes(x = temp_min)) + geom_histogram(binwidth = 2) + labs(title = "Distribución de Temp Min", x = "Temp Min", y = "Frecuencia")
ggplot(data, aes(x = wind)) + geom_histogram(binwidth = 1) + labs(title = "Distribución de Wind", x = "Wind", y = "Frecuencia")
# Convertir la variable objetivo en factor
data$weather <- as.factor(data$weather)

# Dividir los datos en entrenamiento y prueba
set.seed(123)
trainIndex <- createDataPartition(data$weather, p = 0.7, list = FALSE)
dataTrain <- data[trainIndex,]
dataTest <- data[-trainIndex,]

# Entrenar el modelo Random Forest
model <- train(weather ~ precipitation + temp_max + temp_min + wind, data = dataTrain, method = "rf")

# Predicción en el conjunto de prueba
predictions <- predict(model, newdata = dataTest)

# Evaluar el modelo
confMatrix <- confusionMatrix(predictions, dataTest$weather)
print(confMatrix)

# Importancia de las variables
importance <- varImp(model)
print(importance)

# Convertir la importancia en un dataframe para ggplot
importance_df <- as.data.frame(importance$importance)
importance_df$Variable <- rownames(importance_df)

# Gráfico de importancia de las variables
ggplot(importance_df, aes(x = reorder(Variable, Overall), y = Overall)) +
  geom_bar(stat = "identity") +
  coord_flip() +
  labs(title = "Importancia de las Variables", x = "Variable", y = "Importancia")

# Crear tabla resumen de métricas del modelo
accuracy <- confMatrix$overall['Accuracy']
kappa <- confMatrix$overall['Kappa']
results <- data.frame(Metric = c("Accuracy", "Kappa"), Value = c(accuracy, kappa))
print(results)

# Gráfico de resumen de métricas del modelo
ggplot(results, aes(x = Metric, y = Value)) +
  geom_bar(stat = "identity") +
  labs(title = "Métricas del Modelo", x = "Métrica", y = "Valor")
# Ajustar un modelo de regresión logística multinomial
model_multinom <- multinom(weather ~ precipitation + temp_max + temp_min + wind, data = dataTrain)

# Resumen del modelo
summary(model_multinom)

# Prueba de significancia de las variables
z <- summary(model_multinom)$coefficients / summary(model_multinom)$standard.errors
p_values <- (1 - pnorm(abs(z), 0, 1)) * 2
print(p_values)

#Interpretación
#Exactitud General (Accuracy): La exactitud del modelo es del 83.49%, lo que indica que el
#modelo predice correctamente las condiciones meteorológicas la mayor parte del tiempo.

#Consistencia (Kappa): El valor de Kappa de 0.7158 sugiere una buena consistencia del modelo
#más allá de lo que se esperaría por azar.

#Desempeño por Clase:
  
#El modelo predice con alta precisión las clases rain y sun, con sensibilidades de 0.9323 y 0.9219 respectivamente.
#Las clases drizzle y fog tienen sensibilidades más bajas, lo que indica que el modelo tiene dificultades
#para identificar correctamente estas condiciones.
#La clase snow también tiene una sensibilidad moderada, lo que sugiere una necesidad de mejora en la identificación
#de esta condición.
#Especificidad y Valores Predictivos:
  
#La especificidad es alta en todas las clases, especialmente en snow y drizzle,
#lo que significa que el modelo es bueno en identificar la ausencia de estas condiciones.

#fog: La variable precipitation es la única que muestra un valor p significativo (p < 0.05),
#lo que indica que tiene un impacto significativo en la predicción de la condición meteorológica de niebla (fog).

#rain: Las variables precipitation, temp_max, y temp_min tienen valores p significativos (p < 0.05),
#lo que indica que tienen un impacto significativo en la predicción de la condición meteorológica de lluvia (rain).

#snow: Las variables precipitation y wind tienen valores p significativos (p < 0.05), 
#lo que indica que tienen un impacto significativo en la predicción de la condición meteorológica de nieve (snow).

#sun: Las variables precipitation, temp_max, y wind tienen valores p significativos (p < 0.05),
#lo que indica que tienen un impacto significativo en la predicción de la condición meteorológica de sol (sun).

#Estos resultados confirman que las variables predictoras tienen un impacto significativo en la predicción de las diferentes condiciones meteorológicas, respaldando así la hipótesis inicial
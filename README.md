# Aumento de datos con GANs para la detección de cáncer de mama en imágenes térmicas

> **Trabajo de Fin de Máster** · Máster Universitario en Ingeniería y Ciencia de Datos (UNED) · Junio de 2025
> Dirigido por Mariano Rincón Zamorano y José Manuel Cuadra Troncoso
> 📄 [Memoria completa del TFM (PDF)](docs/Roman_Aguilar_Rafael_TFM.pdf)

¿Puede un **StyleGAN** entrenado sobre solo 260 imágenes térmicas generar muestras lo bastante buenas como para mejorar un clasificador de cáncer de mama? Este proyecto lo mide con un diseño experimental pensado para evitar sesgos: **400 modelos StyleGAN entrenados, 5 configuraciones de aumento (una de ellas la base sin GANs), validación cruzada de 5 pliegues repetida 10 veces y unos 16 días de cómputo en GPU**.

<p align="center">
  <img src="docs/assets/generated_healthy.png" alt="Imágenes térmicas sanas generadas por StyleGAN" width="48%">
  <img src="docs/assets/generated_sick.png" alt="Imágenes térmicas enfermas generadas por StyleGAN" width="48%">
  <br><em>Imágenes sintéticas generadas por dos StyleGAN, uno por clase. Izquierda: sanas. Derecha: enfermas.</em>
</p>

## Resumen

| | |
|---|---|
| **Problema** | Detección temprana de cáncer de mama con termografía. El dataset es pequeño: 260 imágenes (166 sanas, 94 enfermas), lo que limita la generalización. |
| **Idea** | Generar imágenes sintéticas con StyleGAN, filtrarlas por calidad y añadirlas al entrenamiento de un clasificador con **atención dura (hard attention)**, el mejor modelo de mi trabajo previo. |
| **Variable estudiada** | Cantidad de imágenes sintéticas añadidas: 0, 768 (~25 %), 1536 (~50 %), 2304 (~75 %) y 3072 (~100 %). |
| **Resultado principal** | Añadir un ~50 % de imágenes sintéticas dio el mejor equilibrio: mayor exhaustividad (0.845 frente a 0.838), AUC 0.87 frente a 0.86 y menos falsos negativos (186 frente a 194), con un coste de cómputo menor que el 75 % y el 100 %. |
| **Conclusión honesta** | Las mejoras son **pequeñas** (de 1 a 2 milésimas o centésimas según la métrica) y cuestan una ejecución ~80-90 veces más lenta (de 66 min a ~4 días). **No hay datos ni pruebas suficientes para afirmar con rigor que este método mejore la detección**: los resultados son indicios, no una demostración. |

## Qué se hizo

### 1. Elección del modelo generativo
Se probaron un **sGAN** (GAN semisupervisado) y un **StyleGAN**. El sGAN no dio resultados útiles ni tras 10.000 épocas (24 h). StyleGAN, implementado desde cero en PyTorch, sí lo hizo.

Mejora clave: en lugar de entrenar a 56×56 (resolución del clasificador), se **reescala el dataset a 128×128** para tener 6 pasos de entrenamiento progresivo en vez de 4. Con ello se pasó de imágenes borrosas a imágenes nítidas en solo 2,5 h.

<p align="center">
  <img src="docs/assets/sgan_vs_stylegan.png" alt="Comparación entre sGAN y StyleGAN" width="70%">
  <br><em>Arriba: sGAN tras 24 h. Abajo: StyleGAN tras 2,5 h, usando 128×128 durante el entrenamiento.</em>
</p>

### 2. Dos GANs, uno por clase, con afinación de hiperparámetros
Cada clase tiene su propio StyleGAN, de modo que cada generador aprende una sola distribución. Se hizo una **búsqueda en cuadrícula** sobre las tasas de aprendizaje del generador y el discriminador y sobre β₁ y β₂ de Adam. El criterio fue el **KID Score** calculado con 10.000 imágenes. Las dos búsquedas sumaron unas 78 h.

| Modelo | Tiempo | KID | FID | LR gen. | LR dis. | β₁ | β₂ |
|---|---|---|---|---|---|---|---|
| Sanas | 45,2 h | 0.107 | 178.8 | 1e-4 | 1e-3 | 1e-5 | 0.9 |
| Enfermas | 33,1 h | 0.128 | 212.9 | 2e-4 | 1e-3 | 0.5 | 0.9 |

La afinación redujo los **fallos de convergencia** (imágenes totalmente negras) y mejoró la calidad de las muestras.

<p align="center">
  <img src="docs/assets/kid_fid_curves.png" alt="Curvas KID y FID durante el entrenamiento" width="70%">
</p>

### 3. Filtrado de las imágenes sintéticas por calidad (KID)
Los GANs con pocos datos generan a veces muestras poco parecidas a las reales. Por eso los lotes de 64 imágenes solo entran en el dataset si su **KID Score** frente al conjunto de validación es ≤ 0.25. Si un lote no lo cumple, se regenera. Tras 3 intentos fallidos se acepta el último para evitar bucles infinitos. KID es la métrica principal porque FID está sesgado con pocas muestras.

### 4. Validación cruzada sin fuga de datos (*data leakage*)
El diseño de la evaluación es la parte más cuidada del proyecto. El aumento clásico (volteo y brillo) es el de la figura siguiente:

<p align="center">
  <img src="docs/assets/data_augmentation_examples.png" alt="Ejemplos de aumento de datos clásico" width="60%">
</p>

- Cada imagen real tiene 10 versiones aumentadas clásicas (volteo horizontal y brillo), lo que da **3113 muestras**. Con `StratifiedGroupKFold` agrupado por paciente, ninguna imagen aumentada acaba en un conjunto distinto del de su original.
- El conjunto de prueba contiene **solo imágenes reales**.
- Los GANs se entrenan **únicamente con el conjunto de entrenamiento de cada pliegue**, y el subconjunto de validación sirve como referencia externa para KID/FID. Después ese subconjunto se reincorpora al entrenamiento del clasificador para no desperdiciar datos.
- Todo el proceso se repite **10 veces** para medir media, varianza y desviación típica, y no quedarse con un solo resultado favorable.

<p align="center">
  <img src="docs/assets/cv_pipeline.png" alt="Flujo de la validación cruzada" width="45%">
</p>

### 5. Un experimento que no funcionó
Probé la técnica de *non-leaking augmentations* de Karras et al. (2020), que aplica transformaciones al discriminador con p = 0,8 para estabilizar el entrenamiento con pocos datos. **Empeoró el KID** (de ~0.2 a ~0.3) y dio imágenes más borrosas. Con solo 260 muestras el dataset es demasiado pequeño para que la técnica funcione, así que la descarté.

<p align="center">
  <img src="docs/assets/non_leaking_kid.png" alt="KID con non-leaking augmentations" width="45%">
  <br><em>KID Score con la técnica non-leaking: valores en torno a 0.3 al final del entrenamiento.</em>
</p>

## Resultados

Clasificador de atención dura, 10 repeticiones × 5 pliegues. Se muestra la media de cada métrica.

| Imágenes sintéticas | Exactitud | Especificidad | Precisión | Exhaustividad | F1 | AUC | Tiempo |
|---|---|---|---|---|---|---|---|
| 0 (base) | 0.865 | 0.885 | 0.848 | 0.838 | 0.839 | 0.86 | 66 min |
| ~25 % (768) | 0.867 | 0.888 | 0.852 | 0.841 | 0.841 | 0.86 | 91,5 h |
| **~50 % (1536)** | 0.870 | 0.887 | 0.859 | **0.845** | **0.845** | **0.87** | 93 h |
| ~75 % (2304) | 0.869 | 0.896 | 0.841 | 0.833 | 0.833 | 0.86 | 97,8 h |
| ~100 % (3072) | 0.870 | **0.897** | **0.862** | 0.835 | 0.844 | **0.87** | 98 h |

*En negrita, los mejores valores de cada columna sin contar el conjunto base. Las varianzas y desviaciones típicas completas están en la memoria. Aquí se usa un criterio de mejora estricto: media superior con dispersión igual o menor.*

<p align="center">
  <img src="docs/assets/confusion_matrices.png" alt="Matrices de confusión por experimento" width="85%">
  <br>
  <img src="docs/assets/roc_curves.png" alt="Curvas ROC por experimento" width="85%">
</p>

**Lectura de los resultados**

- Todas las configuraciones con imágenes sintéticas igualan o superan al conjunto base en exactitud y especificidad.
- El **50 %** reduce los falsos negativos (194 → 186) y sube la exhaustividad. En un contexto clínico es la métrica más importante, porque un falso negativo es un caso de cáncer no detectado.
- El **75 %** empeora precisión y consistencia (varianza de F1 de 0.016 frente a 0.001 en la base).
- Hay una relación directa entre imágenes generadas y tiempo, porque el filtrado por KID obliga a regenerar lotes.
- Las diferencias son pequeñas respecto a la variabilidad entre ejecuciones (por ejemplo, exhaustividad de 0.845 ± 0.09 frente a 0.838 ± 0.08: las desviaciones típicas son mucho mayores que la diferencia de medias). No se ha realizado ningún contraste estadístico de significancia, así que **estos resultados no permiten concluir que el método mejore la detección**, solo que no la empeora de forma evidente y que hay una tendencia ligeramente favorable.

### Coste computacional
Cada experimento con sintéticas entrena 100 StyleGAN (5 pliegues × 2 clases × 10 repeticiones), en total 400 para los 4 experimentos con GANs. Se ejecutó en el servidor de la UNED (2 × NVIDIA Tesla V100, 96 GB de RAM).

<p align="center">
  <img src="docs/assets/execution_time_plot.png" alt="Tiempo de ejecución por experimento" width="50%">
  <img src="docs/assets/convergence_failures.png" alt="Fallos de convergencia por experimento" width="45%">
</p>

Se produjeron fallos de convergencia en torno al **10 %** de los entrenamientos (12 por experimento, 8 en el del 100 %). Se conservaron las imágenes sintéticas de esos casos en vez de reentrenar, para no multiplicar el tiempo de cómputo.

### Interpretabilidad con Grad-CAM
Se aplicó Grad-CAM sobre pacientes enfermas con los modelos entrenados en cada experimento, para ver cómo cambia la zona en la que se fija el clasificador. Los modelos con 25 %, 50 % y 100 % de sintéticas desplazan la atención hacia la zona inferior de la mama derecha, mientras que el modelo base se centra en la izquierda.

<p align="center">
  <img src="docs/assets/gradcam_patient_192.png" alt="Grad-CAM paciente 192" width="75%">
  <br><em>Grad-CAM sobre la paciente enferma 192 según el porcentaje de imágenes sintéticas.</em>
</p>

<p align="center">
  <img src="docs/assets/gradcam_patient_245.png" alt="Grad-CAM paciente 245" width="75%">
  <br><em>Grad-CAM sobre la paciente enferma 245.</em>
</p>

## Tecnologías y habilidades demostradas

- **Deep learning**: PyTorch, StyleGAN (implementación desde cero con entrenamiento progresivo, AdaIN y mapping network), CNN con mecanismos de atención (self, soft y hard), Grad-CAM.
- **Evaluación de modelos generativos**: KID y FID con `torchmetrics`.
- **Metodología experimental**: validación cruzada agrupada y estratificada, prevención de fuga de datos, repeticiones para estimar la varianza, análisis estadístico.
- **Optimización**: búsqueda en cuadrícula de hiperparámetros, detección y mitigación de colapso y fallos de convergencia.
- **Cómputo**: experimentos de varios días en un servidor con GPU, parametrizados mediante un fichero de constantes.
- **Python científico**: scikit-learn, pandas, NumPy, Matplotlib, OpenCV.

## Estructura del repositorio

```
├── models/                    # StyleGAN, cGAN y los 4 clasificadores (CNN, self/soft/hard attention)
├── datasets/                  # Clase CancerDetectionDataset y CSV de etiquetas
├── constants.py               # Hiperparámetros y configuración de los experimentos
├── utils.py                   # KID/FID, filtrado por calidad, generación y división de datos
├── train_StyleGAN.py          # Entrenamiento del StyleGAN
├── hyperparameter_tuning.py   # Búsqueda en cuadrícula de hiperparámetros
├── cross_validation.py        # Validación cruzada con aumento de datos (10 repeticiones)
├── cross_validation_0_gen.py  # Línea base sin imágenes sintéticas
├── augment_images.py          # Aumento clásico (volteo y brillo) del dataset
├── apply_grad_cam.py          # Mapas Grad-CAM sobre los modelos entrenados
├── sGAN.py, train_sGAN.py     # Primer intento con GAN semisupervisado
├── results/                   # Resultados en texto de las validaciones y afinaciones, y Grad-CAM
└── docs/assets/               # Figuras del README
```

## Limitaciones y trabajo futuro

- **Datos insuficientes**: el dataset original tiene solo 260 imágenes (94 enfermas) de una única fuente, y los GANs necesitan normalmente muchos más datos. Sin validación en un conjunto externo, los resultados no se pueden generalizar.
- **Pruebas insuficientes**: solo hay 10 repeticiones por experimento, una única arquitectura de clasificador y un único GAN. No se hicieron tests de significancia estadística, y las mejoras observadas (milésimas o pocas centésimas) quedan dentro de la variabilidad entre ejecuciones.
- **Conclusión**: este trabajo no demuestra que el aumento con GANs mejore la detección de cáncer de mama. Muestra un diseño experimental riguroso y una tendencia que merecería confirmarse con más datos y más experimentos.
- **Sin uso clínico**: el modelo es un experimento académico y no debe usarse para diagnóstico.
- Posibles líneas: **StyleGAN2/3**, **transfer learning** desde un generador preentrenado, ampliar el dataset y más iteraciones de afinación y entrenamiento.

## Autor

**Rafael Román Aguilar** · [GitHub](https://github.com/rafaroman18)

La implementación del StyleGAN parte del código de referencia de [Elilah (2023)](https://www.kaggle.com/code/tauilabdelilah/stylegan-implementation-from-scratch-pytorch).

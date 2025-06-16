# 🧠 MAMA-MIA Challenge 2025

Participación en el reto **MAMA-MIA**: *Advancing Generalizability and Fairness in Breast MRI Tumour Segmentation and Treatment Response Prediction*, organizado por la Universitat de Barcelona dentro del congreso MICCAI 2025.

![Logo reto](https://github.com/user-attachments/assets/97013f7f-34de-44e4-8cd7-5c350b47b282)


---

## 📌 Descripción del reto

El cáncer de mama es el más común entre mujeres y una causa principal de mortalidad. El reto MAMA-MIA propone avanzar en:

1. **Segmentación automática de tumores primarios** en imágenes DCE-MRI (resonancia magnética con contraste).
2. **Predicción de respuesta completa patológica (pCR)** tras quimioterapia neoadyuvante (NAC).

El objetivo es desarrollar soluciones de IA precisas y equitativas, evaluadas también por su **robustez frente a distintas poblaciones y protocolos clínicos**.

---

## 📁 Estructura del repositorio

Este repositorio contiene todos los archivos desarrollados durante la participación en el reto **MAMA-MIA Challenge 2025**: *Advancing Generalizability and Fairness in Breast MRI Tumour Segmentation and Treatment Response Prediction*, organizado por la Universitat de Barcelona en el marco de MICCAI 2025.

A continuación se detalla el contenido de cada carpeta:

### 📊 EDA/
Contiene el análisis exploratorio de datos.
- `EDA.ipynb`: análisis general de las variables clínicas y de respuesta.
- `EDA_imagenes/`: análisis detallado de las imágenes DCE-MRI y segmentaciones tumorales.

### 🧠 Modelos_Segmentación/
Implementación de los modelos de segmentación de tumores primarios, incluyendo configuraciones y scripts para su entrenamiento con nnU-Net v2.

### 🔬 Modelos_pcr/
Modelos de predicción de respuesta patológica completa (pCR), incluyendo enfoques clásicos (radiomics) y deep learning, con distintas entradas (imagen completa vs. recortes centrados).

### 📂 datos/
Archivos auxiliares usados en los experimentos (splits de entrenamiento, anotaciones, coordenadas, etc.).

> ⚠️ **Importante:** Las imágenes DCE-MRI no están incluidas por su gran tamaño. Deben descargarse desde el sitio oficial del reto:  
> 👉 [https://www.synapse.org/Synapse:syn60868042/wiki/628716](https://www.synapse.org/Synapse:syn60868042/wiki/628716)

### 🧪 extra/
Scripts y pruebas adicionales realizadas durante el desarrollo del proyecto. No forman parte del pipeline final, pero fueron relevantes en etapas intermedias.

### ⚖️ fairness_analisis/
Código completo del análisis de equidad aplicado a los modelos, evaluando su rendimiento en distintos subgrupos clínicos.

### 🧾 nnUNet_preprocessed/ y nnUNet_results/
Carpetas estructurales para reproducir los resultados de segmentación con nnU-Net v2.

> ⚠️ Estas carpetas no contienen datos por limitaciones de tamaño. Para obtener los resultados completos es necesario ejecutar localmente los experimentos descritos en `Modelos_Segmentación/`.

### ⚙️ normalizaciones/
Contiene pruebas de normalización aplicadas sobre las imágenes durante la etapa de preprocesamiento.


---

## 🧪 Dataset: MAMA-MIA

El dataset incluye:

- **1506 estudios DCE-MRI pre-tratamiento** de pacientes con cáncer de mama.
- **Segmentaciones manuales y automáticas** del tumor primario.
- **49 variables clínicas, demográficas y de imagen** armonizadas.
- **Pesos preentrenados de nnU-Net** para facilitar el desarrollo de modelos.

![Contenido Dataset](https://github.com/user-attachments/assets/178572c6-501b-4db1-bf39-2e61f9420d92)


> Fuente: Garrucho et al. (2025) [DOI: 10.1038/s41597-025-04707-4](https://doi.org/10.1038/s41597-025-04707-4)

---

## 🖼️ Proceso de Anotación

16 radiólogos de 8 países participaron en la anotación de más de 1000 estudios.

![image](https://github.com/user-attachments/assets/11105224-3747-4974-b12d-8203fa7d4821)


**Protocolo de anotación en 3 pasos:**

1. Imagen restada como guía.
2. Localización del tumor en la primera imagen post-contraste.
3. Segmentación 3D con Mango Viewer.


---

## 🧠 Tareas del reto

### 1. Segmentación de Tumor Primario
- Segmentación automática 3D del tumor en la primera imagen post-contraste.
- Métricas: Dice, Hausdorff Distance.

### 2. Predicción de pCR
- Clasificación binaria: ¿el paciente logrará una respuesta completa patológica?
- Sin usar variables clínicas, solo imágenes.

---

## ⚙️ Plataforma

El reto se ejecuta en [**Codabench**](https://codabench.org), donde:

- Subirás tu modelo en un contenedor Docker.
- Podrás hacer **Sanity Checks** y **validaciones diarias** antes del envío final.
- Todo el proceso es **reproducible y justo** (hardware unificado: GPU RTX 3090).

---

## 🔗 Recursos útiles

- 🌐 Sitio oficial del reto: [https://www.ub.edu/mama-mia](https://www.ub.edu/mama-mia)
- 📄 Paper del dataset: [DOI: 10.1038/s41597-025-04707-4](https://doi.org/10.1038/s41597-025-04707-4)
- 📦 Dataset en Synapse: [syn60868042](https://www.synapse.org/Synapse:syn60868042)
- 🧪 Codabench (plataforma de envío): [https://codabench.org](https://codabench.org)
- ⏱️ Link para inscripción y deadlines: [https://www.codabench.org/competitions/7425/](https://www.codabench.org/competitions/7425/)


---

## ✨ Próximos pasos

1. Preparar entorno de desarrollo con `nnUNet v2`.
2. Crear el `Dockerfile` y `run.sh`.
3. Probar un modelo dummy para Sanity Check.
4. Subirlo a Codabench en cuanto se active el reto.

---

## 👤 Autor

- Nombre: Alba Bernal Rodríguez
- Email: albabero@myuax.com
- GitHub: [albabernal03](https://github.com/albabernal03)

---


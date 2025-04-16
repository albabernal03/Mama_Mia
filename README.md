# 🧠 MAMA-MIA Challenge 2025

Participación en el reto **MAMA-MIA**: *Advancing Generalizability and Fairness in Breast MRI Tumour Segmentation and Treatment Response Prediction*, organizado por la Universitat de Barcelona dentro del congreso MICCAI 2025.

![Logo del reto](https://www.ub.edu/mama-mia/wp-content/uploads/2024/12/logo-2.png)

---

## 📌 Descripción del reto

El cáncer de mama es el más común entre mujeres y una causa principal de mortalidad. El reto MAMA-MIA propone avanzar en:

1. **Segmentación automática de tumores primarios** en imágenes DCE-MRI (resonancia magnética con contraste).
2. **Predicción de respuesta completa patológica (pCR)** tras quimioterapia neoadyuvante (NAC).

El objetivo es desarrollar soluciones de IA precisas y equitativas, evaluadas también por su **robustez frente a distintas poblaciones y protocolos clínicos**.

---

## 📆 Fechas clave

| Evento | Fecha |
|--------|-------|
| 📁 Dataset disponible | Ya publicado (marzo 2025) |
| 🧪 Fase de Sanity Check | 16 de abril de 2025 |
| 📉 Fase de Validación | 15 de mayo de 2025 |
| 📝 Envío del Paper (opcional) | 25 de junio de 2025 |
| 🔬 Fase de Test final | 15 de julio de 2025 |
| 🧾 Último día de envío | 31 de julio de 2025 |
| 🎤 MICCAI 2025 Workshop | 23-27 de septiembre de 2025 |
| 🏆 Anuncio de ganadores | Deep-Breath Workshop |

---

## 🧪 Dataset: MAMA-MIA

El dataset incluye:

- **1506 estudios DCE-MRI pre-tratamiento** de pacientes con cáncer de mama.
- **Segmentaciones manuales y automáticas** del tumor primario.
- **49 variables clínicas, demográficas y de imagen** armonizadas.
- **Pesos preentrenados de nnU-Net** para facilitar el desarrollo de modelos.

![Descripción del dataset](./images/mamamia-dataset-summary.png)

> Fuente: Garrucho et al. (2025) [DOI: 10.1038/s41597-025-04707-4](https://doi.org/10.1038/s41597-025-04707-4)

---

## 🖼️ Proceso de Anotación

16 radiólogos de 8 países participaron en la anotación de más de 1000 estudios.

![Anotadores](./images/radiologists.png)

**Protocolo de anotación en 3 pasos:**

1. Imagen restada como guía.
2. Localización del tumor en la primera imagen post-contraste.
3. Segmentación 3D con Mango Viewer.

![Protocolo de anotación](./images/annotation_protocol.png)

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


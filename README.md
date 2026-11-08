# Detección de defectos de audio en la mezcla musical

MVP del sistema descrito en el trabajo de grado: una aplicación web que
recibe una pista de audio, la analiza con una red PANN ajustada por
transferencia y reporta zumbido (*hum*), ruido de fondo (*hiss*),
distorsión y clics con su ubicación temporal.

## Estructura

```
nucleo/            cadena de señal y arquitectura del modelo, compartidas
entrenamiento/     construcción del artefacto: ajuste fino y evaluación
sistema/           el sistema desplegable
  servicio/        contenedor de análisis (FastAPI)
  web/             aplicación de página única
  nginx/           servidor web y proxy inverso
```

`nucleo/` está compartido a propósito. El servicio **no reimplementa** el
preprocesamiento: importa el mismo código que produjo el conjunto de datos.
Una divergencia entre el preprocesamiento de entrenamiento y el de
inferencia produciría una caída de desempeño sin síntoma visible.

La separación entre `entrenamiento/` y `sistema/` sigue la decisión ADR-02
del documento: el modelo se entrena aparte y se incorpora al sistema como
artefacto. El sistema no reentrena ni mejora con el uso.

## Puesta en marcha

### 1. Pesos preentrenados

```bash
mkdir -p entrenamiento/modelos
curl -L -o "entrenamiento/modelos/Cnn6_mAP=0.343.pth" \
  "https://zenodo.org/records/3987831/files/Cnn6_mAP%3D0.343.pth?download=1"
```

### 2. Entrenamiento

Requiere el conjunto de datos construido (`manifiesto_v2.csv` y la carpeta
`audio/`). Con GPU tarda horas; sin ella, mucho más.

```bash
cd entrenamiento
pip install -r requirements.txt
PYTHONPATH=.. python -m src.entrenar \
    --datos "/ruta/a/datos/dataset_defectos" \
    --pesos "modelos/Cnn6_mAP=0.343.pth" \
    --epocas 30 --lote 64
```

Produce `modelos/mejor.pt` y `modelos/registro_entrenamiento.json` con la
semilla, los hiperparámetros y las métricas por época (RNF-09).

Para una prueba rápida, `--limite-por-clase 500` submuestrea cada clase.

### 3. Evaluación sobre el conjunto de prueba

El conjunto de prueba se abre **una sola vez**, cuando la configuración ya
está fijada con el de validación.

```bash
PYTHONPATH=.. python -m src.evaluar \
    --datos "/ruta/a/datos/dataset_defectos" \
    --modelo modelos/mejor.pt
```

Escribe `informes/evaluacion.md` con la matriz de confusión, las métricas
por clase y el contraste explícito contra RNF-01 y RNF-02.

### 4. Despliegue

```bash
cp entrenamiento/modelos/mejor.pt sistema/servicio/modelo/
cd sistema
docker compose up --build
```

La aplicación queda en <http://localhost:8080>.

## Interfaz de programación

El análisis es **asíncrono**: la carga responde de inmediato y el avance se
consulta aparte, que es lo que exige RF-07. La validación, en cambio, es
síncrona, porque RF-02 pide que un archivo inadmisible se rechace al instante.

| Método y ruta | Qué hace |
|---|---|
| `GET /api/salud` | Estado del servicio, metadatos del modelo y las cinco clases |
| `POST /api/analisis` | Recibe el WAV. `202` con el identificador, o `400` si no es admisible |
| `GET /api/analisis/{id}` | Avance (`estado`, `progreso`, `etapa_texto`) o el reporte ya terminado |
| `DELETE /api/analisis/{id}` | Cierra la sesión y borra pista y reporte (RF-10) |

## Pruebas

Hacen falta dos rutas en `PYTHONPATH`: la raíz, donde está `nucleo`, y
`sistema/servicio`, donde está `app`. El separador es `:` en Linux y macOS
y `;` en Windows.

```bash
# Linux / macOS
PYTHONPATH=".:sistema/servicio" pytest sistema/servicio/tests/test_componentes.py -q
# Windows (Git Bash)
PYTHONPATH=".;sistema/servicio" pytest sistema/servicio/tests/test_componentes.py -q
```

La prueba de sistema corre contra un servicio ya levantado y toma su
dirección de la variable `SERVICIO`:

```bash
SERVICIO="http://127.0.0.1:8080" python sistema/servicio/tests/prueba_sistema.py
```

## Correspondencia con el documento

| Elemento del documento | Dónde está |
|---|---|
| Contenedores (Figura 3) | `sistema/docker-compose.yml` |
| Componentes (Figura 4) | `sistema/servicio/app/` |
| Preprocesador | `nucleo/senal.py` |
| CNN6 y carga de pesos | `nucleo/modelo.py` |
| Validador (RF-02) | `app/validador.py` |
| Segmentación y clases (RF-03) | `nucleo/senal.py`, `app/clasificador.py` |
| Consolidador (RF-04, RF-06) | `app/consolidador.py` |
| Línea de tiempo (RF-05) | `web/app.js`, `web/estilos.css` |
| Seguimiento del avance (RF-07) | `app/main.py`, `web/app.js` |
| Advertencia permanente (RF-08) | `web/index.html` |
| Exportación del reporte (RF-09) | `web/app.js` |
| Borrado al cerrar (RF-10) | `app/almacenamiento.py` |
| Lectura de la confianza (RNF-10) | `web/index.html` |
| Proxy como único punto de entrada (ADR-04, RNF-07) | `nginx/nginx.conf` |
| Pruebas unitarias (nivel 4 de V&V) | `sistema/servicio/tests/` |

## Estado

Lo que funciona de punta a punta: la cadena de señal, la carga de los pesos
preentrenados, el ajuste fino, la evaluación, el servicio y la interfaz.

Lo que falta: **el modelo entrenado en condiciones**. El artefacto incluido
proviene de una corrida de humo sobre un subconjunto pequeño y pocas
épocas; sirve para comprobar que el sistema funciona, no para medir nada.
Las cifras de desempeño del trabajo deben salir de un entrenamiento
completo sobre GPU.

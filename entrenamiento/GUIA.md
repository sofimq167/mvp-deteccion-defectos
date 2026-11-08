# Guía para entrenar el modelo

Procedimiento para obtener el modelo que producirá las cifras del Capítulo 4.
El artefacto que viene con el MVP **no sirve para eso**: se entrenó sobre 3.500
segmentos (700 por clase) de los 57.058 disponibles, durante 10 épocas en CPU,
para comprobar que la cañería funciona.

## Qué hay que lograr

El objetivo no es "que entrene", es cumplir dos requisitos de la SRS:

| | Umbral |
|---|---|
| **RNF-01** · Exactitud balanceada sobre prueba | ≥ 0,90 |
| **RNF-01** · Exhaustividad en *cada* una de las 5 clases | ≥ 0,80 |
| **RNF-02** · Superar la línea base acústica | > 0,463 |

El 0,463 es una regresión logística sobre siete descriptores elementales de la
forma de onda: RMS, pico, factor de cresta, centroide espectral, tasa de cruces
por cero, y las proporciones de energía bajo 200 Hz y sobre 6000 Hz (estas dos
últimas puestas a propósito como pistas de *hum* y de *hiss*). Se entrena en la
partición de entrenamiento y se mide en la de validación, y la reproduce
`src/linea_base.py`. Si el modelo no la supera con holgura, la red no está
aportando nada que no diera un clasificador trivial.

---

## Paso 0 · Precalcular los rasgos (una vez, en tu máquina)

El audio generado ocupa 13 GB y en cada época habría que volver a decodificarlo.
Precalcular el log-mel lo reduce a **1,45 GB** y convierte el costo por lote en
una lectura de memoria.

```bash
cd entrenamiento
PYTHONPATH=.. python -m src.precalcular \
    --datos  "C:/Users/sofim/OneDrive/Escritorio/datos/dataset_defectos" \
    --salida "C:/Users/sofim/OneDrive/Escritorio/datos/rasgos"
```

Tarda unos **11 minutos** y escribe, por partición, `rasgos_*.npy` (float16),
`etiquetas_*.npy` y `meta_*.json`. Este último guarda `clase_simulacion` y
`n_clics_segmento`, que hacen falta para los desgloses del Capítulo 4.

> La partición de validación **ya está hecha**: 15.250 segmentos, 197 MB, 88 s.
> Para no repetirla, pedí solo las que faltan:
> `--particiones entrenamiento prueba`

**Comprobá que los conteos cuadran con la Tabla 4** antes de seguir. Si no
cuadran, hay algo mal y entrenar sería perder horas.

---

## Paso 1 · Llevarlo a una GPU

No es opcional. La corrida de humo midió entre 192 y 388 segundos por época
sobre 3.500 segmentos. El conjunto de entrenamiento completo tiene 57.058,
unas 16 veces más, así que en CPU cada época ronda **la hora** y 40 épocas se
irían a casi dos días. En una GPU de gama media son minutos.

1. Subí a Google Drive la carpeta `rasgos/` (1,45 GB) y los pesos
   `Cnn6_mAP=0.343.pth` (23 MB).
2. Subí también el código: `nucleo/` y `entrenamiento/src/`.
3. En Colab, con entorno de ejecución **GPU**:

```python
from google.colab import drive
drive.mount('/content/drive')

%cd /content/drive/MyDrive/deteccion
!pip -q install soundfile scikit-learn

import os; os.environ['PYTHONPATH'] = '/content/drive/MyDrive/deteccion'
!python -m src.entrenar --rasgos datos/rasgos \
        --pesos modelos/Cnn6_mAP=0.343.pth \
        --epocas 40 --lote 64 --hilos 2
```

Guardá los puntos de control en Drive, no en el disco de la sesión: si Colab
corta la sesión, se pierde todo lo que esté fuera de Drive.

---

## Paso 2 · Entrenar

```bash
PYTHONPATH=.. python -m src.entrenar \
    --rasgos  "/ruta/a/datos/rasgos" \
    --pesos   "modelos/Cnn6_mAP=0.343.pth" \
    --epocas 40 --lote 64 --tasa 1e-3 --paciencia 6
```

### Qué hacen los parámetros que importan

**`--tasa` y `--tasa-cuerpo`.** El cuerpo de la red llega preentrenado sobre
AudioSet; la cabeza de 5 clases parte de cero. Aplicarles la misma tasa
destruiría lo aprendido mientras la cabeza todavía no sabe nada. Por omisión el
cuerpo usa la décima parte: `1e-4` contra `1e-3`. Es el ajuste más importante
de todos.

**`--paciencia`.** Detiene el entrenamiento tras N épocas sin mejorar la
exactitud balanceada en validación. Protege del sobreajuste, que acá es un
riesgo real: tu tamaño de muestra efectivo son **80 canciones**, no 57.058
segmentos. Segmentos de la misma canción comparten instrumentación, sala y
cadena de mezcla.

**`--lote`.** 64 va bien en una T4. Si da error de memoria, bajá a 32.

### Qué vigilar mientras corre

La salida imprime, por época, pérdida y exactitud balanceada en validación.

- **La pérdida baja y la exactitud balanceada sube** → va bien, dejalo correr.
- **La pérdida baja y la exactitud balanceada se estanca o cae** → sobreajuste.
  La parada temprana lo corta sola; el remedio es bajar `--tasa` o subir el
  número de canciones, no más épocas.
- **Ninguna de las dos se mueve en 3 o 4 épocas** → algo está mal. Revisá que
  los pesos preentrenados hayan cargado: la salida dice cuántos tensores cargó.
  Deben ser **31**, con 2 sin inicializar (la cabeza nueva).

---

## Paso 3 · Si no llega a 0,90

Probá en este orden, que va de lo más barato a lo más caro, y **registrá cada
intento** con su resultado:

1. **Más épocas con más paciencia** (`--epocas 60 --paciencia 10`). Lo primero,
   porque no cuesta decidir nada.
2. **Bajar la tasa de la cabeza** a `5e-4`. Si la curva de validación oscila
   mucho entre épocas, la tasa es alta.
3. **Subir la tasa del cuerpo** a `2e-4` (`--tasa-cuerpo 2e-4`). El dominio de
   la música está lejos de AudioSet; puede hacer falta mover más el extractor.
4. **Probar CNN10** en lugar de CNN6. Cuesta casi lo mismo (5,2 M de parámetros
   contra 4,8 M) y rinde bastante mejor en AudioSet. Requiere implementar el
   bloque de dos convoluciones de 3×3 en `nucleo/modelo.py`.

Si tras esto la clase **clic** sigue siendo la peor, no es necesariamente culpa
del modelo: ya medimos que la mediana de las discontinuidades del clic
reductivo es indistinguible de los transitorios propios de la música. Reportalo
desglosado por `clase_simulacion` antes de concluir nada.

### Qué tan difícil es la tarea, medido

Antes de culpar al entrenamiento conviene tener presente lo que ya sabemos del
conjunto, porque acota lo que se puede esperar:

- La Figura 5 del documento muestra que, a las severidades del conjunto, **solo
  el ruido de fondo se distingue a simple vista** en el log-mel. El zumbido, la
  saturación y el clic reductivo quedan enterrados bajo música densa.
- El clic reductivo produce discontinuidades cuya mediana es ×1,01 frente a los
  transitorios propios de la música.
- El tamaño de muestra efectivo son **80 canciones**, no 57.058 segmentos.

Nada de esto significa que 0,90 sea inalcanzable, pero sí que si el resultado
se queda corto la explicación probablemente esté en el conjunto y no en los
hiperparámetros. Decirlo con estas mediciones es un resultado del trabajo, no
una excusa.

---

## Paso 4 · Evaluar (una sola vez)

Cuando la configuración esté fijada —y solo entonces— abrí el conjunto de
prueba:

```bash
PYTHONPATH=.. python -m src.evaluar \
    --rasgos "/ruta/a/datos/rasgos" \
    --modelo modelos/mejor.pt
```

Escribe `informes/evaluacion.md` con la matriz de confusión, las métricas por
clase y el veredicto explícito de RNF-01 y RNF-02. Ese informe es la evidencia
del nivel 1 del plan de V&V.

---

## Paso 5 · Desplegar

```bash
cp modelos/mejor.pt ../sistema/servicio/modelo/
cd ../sistema && docker compose up --build
```

---

## Lo que no hay que hacer

**No abras el conjunto de prueba más de una vez.** Si lo usás para decidir
hiperparámetros, deja de ser una estimación del comportamiento sobre material
nuevo y pasa a ser otra partición de validación. El número que publiques ya no
significaría lo que decís que significa.

**No repartas segmentos de una misma canción entre particiones.** El manifiesto
ya está particionado por canción; respetalo.

**No cambies los parámetros del log-mel** (32 kHz, ventana 1024, salto 320, 64
bandas, 50–14000 Hz). Son los del preentrenamiento: cambiarlos invalida los
pesos. Hay una prueba unitaria que falla si alguien los toca.

---

## Qué registrar (RNF-09)

`src/entrenar.py` escribe `modelos/registro_entrenamiento.json` con semilla,
particiones, tasas, tamaño de lote y métricas por época. **Guardá un registro
por cada intento**, no solo del que funcionó: la tabla de intentos fallidos es
lo que justifica la configuración elegida cuando el jurado pregunte por qué esa
y no otra.

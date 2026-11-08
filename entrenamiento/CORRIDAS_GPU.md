# Qué correr cuando vuelva la GPU

Estado al 4 de noviembre de 2026. La configuración retenida es **CNN6 con
mixup y SpecAugment**, intento 03, que da **0,581** en validación y, fusionada
con los descriptores acústicos, **0,612** sobre el conjunto de prueba.

Lo que queda por probar es si una red más grande mejora eso. Hay dos
candidatas y conviene correrlas **en este orden**.

---

## Por qué CNN10 primero

| red | parámetros | mAP en AudioSet |
|---|---|---|
| CNN6 (actual) | 4,6 M | 0,343 |
| **CNN10** | **5,0 M** | **0,380** |
| CNN14 | 79,7 M | 0,431 |

CNN10 consigue buena parte de la mejora de CNN14 con un aumento de tamaño
del ocho por ciento. Importa porque el ADR-01 descartó las redes grandes por
el tamaño efectivo de la muestra, que son **80 canciones**, y porque ya se
midió que CNN6 con 4,6 M **se sobreajustaba** desde la época 25 antes de
agregar regularización. Meter 80 M va en contra de ese diagnóstico.

Además CNN10 es una ablación limpia: corre con las mismas 60 épocas y el
mismo plan de tasas que el intento 03, así que la única variable que cambia
es la arquitectura. CNN14 necesita menos épocas por costo, de modo que su
comparación arrastra una segunda diferencia y vale menos como evidencia.

---

## 0. Antes de nada: sincronizar tres archivos

Cambiaron después de la última subida. Súbelos a Drive **reemplazando**, nunca
«conservar ambos».

| archivo local | destino en Drive |
|---|---|
| `nucleo/modelo.py` | `deteccion/nucleo/` |
| `entrenamiento/src/entrenar.py` | `deteccion/src/` |
| `entrenamiento/src/exportar.py` | `deteccion/src/` |

Verificación, que es donde ya falló una vez:

```python
from google.colab import drive
drive.mount('/content/drive')

import glob, shutil
RAIZ = '/content/drive/MyDrive/deteccion/'
for d in glob.glob(RAIZ + '**/__pycache__', recursive=True):
    shutil.rmtree(d, ignore_errors=True)

for f in ['nucleo/modelo.py', 'src/entrenar.py', 'src/exportar.py']:
    s = open(RAIZ + f, encoding='utf-8').read()
    print('%-20s %s' % (f, 'OK' if 'cnn10' in s.lower() else 'FALTA SUBIR'))
```

Las tres tienen que decir OK.

---

## 1. Pesos preentrenados

```bash
%cd /content/drive/MyDrive/deteccion
!wget -q --show-progress -O "Cnn10_mAP=0.380.pth" \
  "https://zenodo.org/record/3987831/files/Cnn10_mAP%3D0.380.pth"
!ls -la *.pth
```

`Cnn10` debe pesar unas decenas de MB. Si bajó unos pocos kilobytes es una
página de error, no el archivo.

---

## 2. Rasgos en disco local

Son horas leyendo el mismo archivo en cada época, y Drive es lento para eso.

```python
import shutil, os
os.makedirs('/content/rasgos', exist_ok=True)
for f in ['rasgos_entrenamiento.npy', 'etiquetas_entrenamiento.npy',
          'rasgos_validacion.npy', 'etiquetas_validacion.npy']:
    o = '/content/drive/MyDrive/deteccion/rasgos/' + f
    d = '/content/rasgos/' + f
    shutil.copyfile(o, d)
    assert os.path.getsize(d) == os.path.getsize(o), 'copia incompleta: ' + f
    print('%-32s %8.1f MB' % (f, os.path.getsize(d) / 1e6))
```

La aserción está porque una copia recursiva ya se trajo 7 de 9 archivos en
silencio.

---

## 3. Prueba de humo, dos minutos. No la saltes

```bash
%cd /content/drive/MyDrive/deteccion
!python -m src.entrenar --arquitectura cnn10 \
    --rasgos /content/rasgos --pesos "Cnn10_mAP=0.380.pth" \
    --salida /content/humo --limite-por-clase 200 --epocas 1 --lote 32
```

Tiene que imprimir **«pesos preentrenados: N tensores cargados, 2 sin
inicializar»**, con N bastante mayor que los 31 de CNN6. Si dice 0 cargados,
los nombres no coinciden: parar y avisar.

---

## 4. Corrida A: CNN10

```bash
%cd /content/drive/MyDrive/deteccion
!mkdir -p intentos/05_cnn10
!python -m src.entrenar --arquitectura cnn10 \
    --rasgos /content/rasgos --pesos "Cnn10_mAP=0.380.pth" \
    --salida intentos/05_cnn10 \
    --mixup 0.3 --mascara-tiempo 16 \
    --epocas 60 --paciencia 10 --lote 32 \
    --tasa 1e-3 --tasa-cuerpo 2e-4 \
    2>&1 | tee intentos/05_cnn10/log.txt
```

Unas dos o tres horas. Mismas épocas, mismas tasas y mismo lote que el
intento 03: la única diferencia es la red.

---

## 5. Corrida B: CNN14, opcional

Solo si queda cuota después de la A.

```bash
%cd /content/drive/MyDrive/deteccion
!wget -q --show-progress -O "Cnn14_mAP=0.431.pth" \
  "https://zenodo.org/record/3987831/files/Cnn14_mAP%3D0.431.pth"

!mkdir -p intentos/04_cnn14
!python -m src.entrenar --arquitectura cnn14 \
    --rasgos /content/rasgos --pesos "Cnn14_mAP=0.431.pth" \
    --salida intentos/04_cnn14 \
    --mixup 0.3 --mascara-tiempo 16 \
    --epocas 25 --paciencia 6 --lote 32 \
    --tasa 1e-3 --tasa-cuerpo 2e-4 \
    2>&1 | tee intentos/04_cnn14/log.txt
```

`Cnn14` pesa varios cientos de MB. Son 25 épocas y no 60 porque cada una
tarda entre tres y cinco veces más.

---

## 6. Exportar y mandar

Por cada corrida que termine:

```bash
%cd /content/drive/MyDrive/deteccion
!python -m src.exportar \
    --modelo intentos/05_cnn10/mejor.pt \
    --rasgos /content/rasgos --particion validacion --hilos 0 \
    --salida /content/predicciones_validacion_cnn10.npz
```

```python
from google.colab import files
files.download('/content/predicciones_validacion_cnn10.npz')
files.download('/content/drive/MyDrive/deteccion/intentos/05_cnn10/log.txt')
```

No hace falta pasar `--arquitectura` al exportar: el checkpoint registra con
cuál se entrenó.

---

## Regla que no se puede romper

El conjunto de prueba **ya se abrió una vez** y dio 0,612. Si alguna de estas
corridas mejora y se decide cambiar el modelo retenido, hay que volver a
prueba **una sola vez más** y declararlo en el documento como segunda
apertura. Elegir entre CNN6, CNN10 y CNN14 mirando prueba invalidaría la
estimación.

La comparación se hace **sobre validación**. Prueba se abre al final, con el
ganador ya elegido.

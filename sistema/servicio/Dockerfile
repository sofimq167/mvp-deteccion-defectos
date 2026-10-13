# Imagen del servicio de analisis.
#
# El contexto de construccion es la RAIZ del proyecto, no esta carpeta,
# porque el servicio necesita el paquete `nucleo`: no reimplementa la cadena
# de preprocesamiento, usa literalmente la misma que produjo el conjunto de
# datos.
#
#   docker build -f sistema/servicio/Dockerfile -t deteccion-servicio .

FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    RUTA_MODELO=/app/modelo/mejor.pt \
    RUTA_TEMPORAL=/tmp/analisis

# libsndfile es la dependencia nativa de soundfile
RUN apt-get update \
 && apt-get install -y --no-install-recommends libsndfile1 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY sistema/servicio/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
 && pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY nucleo/ /app/nucleo/
COPY sistema/servicio/app/ /app/app/
COPY sistema/servicio/modelo/ /app/modelo/

# El servicio corre sin privilegios: control del analisis de amenazas.
RUN useradd --create-home --uid 10001 analisis \
 && mkdir -p /tmp/analisis && chown -R analisis /tmp/analisis /app
USER analisis

EXPOSE 8000

# Un solo proceso: el sistema atiende una pista a la vez y no incorpora
# cola de trabajos (antiescenario de la seccion 3.2.4).
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]

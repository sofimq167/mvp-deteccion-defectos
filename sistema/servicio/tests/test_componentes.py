"""
Pruebas unitarias de los componentes deterministas (nivel 4 del plan de V&V).

El enfasis esta en la cadena de preprocesamiento: un error silencioso en la
segmentacion o en el log-mel no se manifiesta como una falla visible sino
como una caida de desempeno facil de atribuir por equivocacion al modelo.

    pytest sistema/servicio/tests -q
"""

from __future__ import annotations

import io
import struct
import wave

import numpy as np
import pytest

from nucleo import config as cfg
from nucleo import senal
from app.consolidador import consolidar, resumen
from app.validador import EntradaInvalida, validar


# --------------------------------------------------------------- ayudas ----

def wav_bytes(segundos=3.0, sr=cfg.SR, canales=1, amplitud=0.2):
    n = int(segundos * sr)
    t = np.arange(n) / sr
    x = amplitud * np.sin(2 * np.pi * 220 * t)
    datos = (np.clip(x, -1, 1) * 32767).astype("<i2")
    if canales == 2:
        datos = np.repeat(datos[:, None], 2, axis=1).ravel()
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(canales)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(datos.tobytes())
    return buf.getvalue()


# ------------------------------------------------- parametros declarados ----

def test_los_parametros_son_los_del_preentrenamiento():
    """Si alguno cambia, los pesos de PANN dejan de ser validos."""
    assert (cfg.SR, cfg.VENTANA, cfg.SALTO) == (32_000, 1024, 320)
    assert (cfg.N_MEL, cfg.FMIN, cfg.FMAX) == (64, 50, 14_000)
    assert cfg.MUESTRAS_SEGMENTO == 32_000


# ------------------------------------------------------------- la senal ----

def test_banco_mel_tiene_la_forma_esperada():
    assert senal._BANCO.shape == (cfg.N_MEL, 1 + cfg.VENTANA // 2)
    assert np.all(senal._BANCO >= 0)
    assert np.all(senal._BANCO.sum(axis=1) > 0)        # ningun filtro vacio


def test_log_mel_devuelve_una_trama_por_salto():
    x = np.random.randn(2, cfg.MUESTRAS_SEGMENTO) * 0.1
    e = senal.log_mel(x)
    assert e.shape == (2, 1 + cfg.MUESTRAS_SEGMENTO // cfg.SALTO, cfg.N_MEL)
    assert e.dtype == np.float32


def test_segmentar_no_solapa_ni_deja_huecos():
    x = np.zeros(cfg.MUESTRAS_SEGMENTO * 3 + 500)
    segs, inicios = senal.segmentar(x)
    assert len(segs) == 3                               # el resto se descarta
    assert list(inicios) == [0, 32_000, 64_000]
    assert all(s.shape[0] == cfg.MUESTRAS_SEGMENTO for s in segs)


def test_segmentar_pista_mas_corta_que_un_segundo():
    segs, inicios = senal.segmentar(np.zeros(1000))
    assert len(segs) == 0 and len(inicios) == 0


def test_remuestreo_conserva_la_duracion():
    sr = 44_100
    x = np.random.randn(sr * 2) * 0.1
    y = senal.remuestrear(x, sr)
    assert abs(len(y) / cfg.SR - 2.0) < 0.01


def test_detecta_el_silencio():
    assert senal.es_silencio(np.zeros(cfg.MUESTRAS_SEGMENTO))
    assert not senal.es_silencio(np.random.randn(cfg.MUESTRAS_SEGMENTO) * 0.1)


# ---------------------------------------------------------- el validador ----

def test_acepta_un_wav_valido():
    meta = validar(wav_bytes(3.0), "pista.wav")
    assert meta.canales == 1 and abs(meta.duracion_s - 3.0) < 0.01


def test_acepta_estereo_y_otra_frecuencia():
    meta = validar(wav_bytes(2.0, sr=44_100, canales=2), "pista.wav")
    assert meta.canales == 2 and meta.frecuencia_hz == 44_100


def test_rechaza_lo_que_no_es_wav():
    with pytest.raises(EntradaInvalida, match="no es un WAV"):
        validar(b"%PDF-1.7 esto no es audio", "documento.pdf")


def test_rechaza_el_archivo_vacio():
    with pytest.raises(EntradaInvalida):
        validar(b"", "vacio.wav")


def test_rechaza_una_pista_demasiado_corta():
    with pytest.raises(EntradaInvalida, match="al menos"):
        validar(wav_bytes(0.4), "corta.wav")


def test_rechaza_una_cabecera_valida_con_contenido_roto():
    bueno = wav_bytes(2.0)
    roto = bueno[:60] + b"\x00" * 20        # se trunca el bloque de datos
    roto = roto[:4] + struct.pack("<I", 10 ** 9) + roto[8:]
    with pytest.raises(EntradaInvalida):
        validar(roto, "roto.wav")


# -------------------------------------------------------- el consolidador ----

def _entrada(clases, conf=0.9):
    idx = np.array([cfg.CLASES.index(c) for c in clases])
    return (idx, np.full(len(clases), conf),
            np.arange(len(clases), dtype=float),
            np.zeros(len(clases), dtype=bool))


def test_agrupa_segmentos_contiguos_en_un_hallazgo():
    h = consolidar(*_entrada(["sin_defecto", "hum", "hum", "hum", "sin_defecto"]))
    assert len(h) == 1
    assert h[0]["clase"] == "hum"
    assert (h[0]["inicio_s"], h[0]["fin_s"]) == (1.0, 4.0)
    assert h[0]["n_segmentos"] == 3


def test_un_hueco_de_un_segmento_no_parte_el_hallazgo():
    h = consolidar(*_entrada(["hum", "hum", "sin_defecto", "hum", "hum"]))
    assert len(h) == 1 and h[0]["n_segmentos"] == 4


def test_un_hueco_largo_si_parte_el_hallazgo():
    h = consolidar(*_entrada(["hum", "sin_defecto", "sin_defecto", "sin_defecto", "hum"]))
    assert len(h) == 2


def test_clases_distintas_no_se_mezclan():
    h = consolidar(*_entrada(["hum", "hum", "hiss", "hiss"]))
    assert [x["clase"] for x in h] == ["hum", "hiss"]


def test_los_silenciosos_no_se_reportan():
    idx, conf, inicios, _ = _entrada(["hum", "hum"])
    h = consolidar(idx, conf, inicios, np.array([True, True]))
    assert h == []


def test_pista_limpia_no_produce_hallazgos():
    assert consolidar(*_entrada(["sin_defecto"] * 5)) == []


def test_el_resumen_cuenta_todos_los_segmentos():
    idx, _, _, sil = _entrada(["hum", "hiss", "sin_defecto"])
    r = resumen(idx, sil)
    assert sum(r.values()) == 3 and r["hum"] == 1 and r["hiss"] == 1

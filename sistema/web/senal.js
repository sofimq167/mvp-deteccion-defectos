/* ════════════════════════════════════════════════════════════════
   sonar · lectura local de la senal

   Dibuja la forma de onda y el espectrograma a partir del archivo que
   eligio el usuario, decodificado en el navegador. No le pide nada al
   servicio ni toca el reporte: lo que se ve es el audio real, y los
   hallazgos se superponen despues, tal como los devolvio el modelo.

   Si el navegador no puede decodificar el WAV, `preparar` falla y la
   interfaz muestra la linea de tiempo sin estas dos vistas.
   ════════════════════════════════════════════════════════════════ */

const Senal = (() => {
  const N_PICOS = 3000;           // resolucion guardada de la forma de onda
  const N_FFT = 1024;
  const RANGO_DB = 80;            // lo que queda por debajo se ve negro
  const F_MIN = 40;               // Hz, borde inferior del eje

  /* Frecuencia de decodificacion: 32 kHz es la del modelo; en archivos
     grandes se baja a la mitad para no agotar la memoria del navegador. */
  function frecuenciaPara(bytes) {
    return bytes <= 80 * 1024 * 1024 ? 32000 : 16000;
  }

  async function decodificar(archivo) {
    const Ctx = window.OfflineAudioContext || window.webkitOfflineAudioContext;
    if (!Ctx) throw new Error("sin Web Audio");
    const ctx = new Ctx(1, 1, frecuenciaPara(archivo.size));
    const datos = await archivo.arrayBuffer();
    return await ctx.decodeAudioData(datos);
  }

  /* ─────────────────────────────────────────────── forma de onda ── */

  function picos(buf, n = N_PICOS) {
    const canales = [];
    for (let c = 0; c < buf.numberOfChannels; c++) canales.push(buf.getChannelData(c));
    const largo = buf.length;
    n = Math.min(n, largo);
    const min = new Float32Array(n), max = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      const a = Math.floor(i * largo / n), b = Math.floor((i + 1) * largo / n);
      let lo = 0, hi = 0;
      for (const d of canales) {
        for (let k = a; k < b; k++) {
          const v = d[k];
          if (v < lo) lo = v; else if (v > hi) hi = v;
        }
      }
      min[i] = lo; max[i] = hi;
    }
    return { min, max };
  }

  /* ─────────────────────────────────────────────── espectrograma ── */

  function crearFFT(n) {
    const bits = Math.log2(n);
    const rev = new Uint32Array(n);
    for (let i = 0; i < n; i++) {
      let r = 0;
      for (let b = 0; b < bits; b++) r |= ((i >> b) & 1) << (bits - 1 - b);
      rev[i] = r;
    }
    const cos = new Float32Array(n / 2), sin = new Float32Array(n / 2);
    for (let i = 0; i < n / 2; i++) {
      cos[i] = Math.cos(-2 * Math.PI * i / n);
      sin[i] = Math.sin(-2 * Math.PI * i / n);
    }
    const re = new Float32Array(n), im = new Float32Array(n);

    // Devuelve la potencia de los bins 0..n/2 de `x` (ya ventaneado).
    return function potencia(x, salida) {
      for (let i = 0; i < n; i++) { re[rev[i]] = x[i]; im[rev[i]] = 0; }
      for (let tam = 2; tam <= n; tam <<= 1) {
        const mitad = tam >> 1, paso = n / tam;
        for (let i = 0; i < n; i += tam) {
          for (let j = 0; j < mitad; j++) {
            const k = j * paso;
            const a = i + j, b = a + mitad;
            const tr = re[b] * cos[k] - im[b] * sin[k];
            const ti = re[b] * sin[k] + im[b] * cos[k];
            re[b] = re[a] - tr; im[b] = im[a] - ti;
            re[a] += tr; im[a] += ti;
          }
        }
      }
      for (let i = 0; i <= n / 2; i++) salida[i] = re[i] * re[i] + im[i] * im[i];
    };
  }

  /* Mapa "inferno": perceptualmente uniforme y legible en escala de
     grises; es el habitual para espectrogramas. */
  const INFERNO = ["#000004", "#160b39", "#420a68", "#6a176e", "#932667", "#bc3754",
                   "#dd513a", "#f37819", "#fca50a", "#f6d746", "#fcffa4"];
  const PALETA = (() => {
    const rgb = INFERNO.map((h) => [1, 3, 5].map((i) => parseInt(h.slice(i, i + 2), 16)));
    const lut = new Uint8ClampedArray(256 * 3);
    for (let i = 0; i < 256; i++) {
      const t = i / 255 * (rgb.length - 1), k = Math.min(Math.floor(t), rgb.length - 2), f = t - k;
      for (let c = 0; c < 3; c++) lut[i * 3 + c] = rgb[k][c] + (rgb[k + 1][c] - rgb[k][c]) * f;
    }
    return lut;
  })();

  const pausa = () => new Promise((r) => setTimeout(r, 0));

  async function espectrograma(buf, columnas = 1400, filas = 160) {
    const sr = buf.sampleRate;
    const fMax = sr / 2;
    const canales = [];
    for (let c = 0; c < buf.numberOfChannels; c++) canales.push(buf.getChannelData(c));
    const largo = buf.length;
    columnas = Math.max(1, Math.min(columnas, Math.floor(largo / (N_FFT / 2))));

    const hann = new Float32Array(N_FFT);
    for (let i = 0; i < N_FFT; i++) hann[i] = 0.5 - 0.5 * Math.cos(2 * Math.PI * i / (N_FFT - 1));

    // Bandas logaritmicas: cada fila promedia los bins que le tocan.
    const df = sr / N_FFT;
    const desde = new Uint16Array(filas), hasta = new Uint16Array(filas);
    for (let r = 0; r < filas; r++) {
      const f0 = F_MIN * Math.pow(fMax / F_MIN, r / filas);
      const f1 = F_MIN * Math.pow(fMax / F_MIN, (r + 1) / filas);
      desde[r] = Math.min(N_FFT / 2, Math.round(f0 / df));
      hasta[r] = Math.min(N_FFT / 2, Math.max(desde[r], Math.round(f1 / df) - 1));
    }

    const potencia = crearFFT(N_FFT);
    const marco = new Float32Array(N_FFT);
    const espectro = new Float32Array(N_FFT / 2 + 1);
    const acum = new Float32Array(N_FFT / 2 + 1);
    const db = new Float32Array(columnas * filas);
    const tramo = largo / columnas;
    const porColumna = Math.max(1, Math.min(4, Math.floor(tramo / N_FFT)));
    let tope = -Infinity;

    for (let c = 0; c < columnas; c++) {
      acum.fill(0);
      for (let m = 0; m < porColumna; m++) {
        const centro = Math.floor(c * tramo + (m + 0.5) * tramo / porColumna);
        const ini = Math.max(0, Math.min(largo - N_FFT, centro - N_FFT / 2));
        for (let i = 0; i < N_FFT; i++) {
          let v = 0;
          for (const d of canales) v += d[ini + i] || 0;
          marco[i] = v / canales.length * hann[i];
        }
        potencia(marco, espectro);
        for (let i = 0; i < acum.length; i++) acum[i] += espectro[i];
      }
      for (let r = 0; r < filas; r++) {
        let s = 0;
        for (let b = desde[r]; b <= hasta[r]; b++) s += acum[b];
        const v = 10 * Math.log10(s / ((hasta[r] - desde[r] + 1) * porColumna) + 1e-12);
        db[c * filas + r] = v;
        if (v > tope) tope = v;
      }
      if (c % 150 === 149) await pausa();      // no congela la interfaz
    }

    // Imagen fuera de pantalla: fila 0 arriba = frecuencia mas alta.
    const lienzo = document.createElement("canvas");
    lienzo.width = columnas; lienzo.height = filas;
    const ctx = lienzo.getContext("2d");
    const img = ctx.createImageData(columnas, filas);
    for (let c = 0; c < columnas; c++) {
      for (let r = 0; r < filas; r++) {
        const t = Math.max(0, Math.min(1, (db[c * filas + r] - (tope - RANGO_DB)) / RANGO_DB));
        const k = Math.round(t * 255) * 3;
        const p = ((filas - 1 - r) * columnas + c) * 4;
        img.data[p] = PALETA[k]; img.data[p + 1] = PALETA[k + 1];
        img.data[p + 2] = PALETA[k + 2]; img.data[p + 3] = 255;
      }
    }
    ctx.putImageData(img, 0, 0);
    return { lienzo, fMin: F_MIN, fMax };
  }

  /* Todo lo que la interfaz necesita del archivo; el AudioBuffer se
     descarta al terminar, para no retener cientos de MB. */
  async function preparar(archivo) {
    const buf = await decodificar(archivo);
    const resultado = {
      duracion: buf.duration,
      frecuencia: buf.sampleRate,
      picos: picos(buf),
      espectro: null,
    };
    resultado.listoEspectro = espectrograma(buf).then((e) => (resultado.espectro = e));
    return resultado;
  }

  /* ───────────────────────────────────────────────────── dibujo ── */

  function ajustar(lienzo) {
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const w = Math.max(1, Math.round(lienzo.clientWidth * dpr));
    const h = Math.max(1, Math.round(lienzo.clientHeight * dpr));
    if (lienzo.width !== w || lienzo.height !== h) { lienzo.width = w; lienzo.height = h; }
    return { ctx: lienzo.getContext("2d"), w, h, dpr };
  }

  /* Barras finas y simetricas. Amplitud absoluta: 1.0 es el fondo de
     escala, asi un recorte se ve como tal. */
  function dibujarOnda(lienzo, p, color) {
    const { ctx, w, h, dpr } = ajustar(lienzo);
    ctx.clearRect(0, 0, w, h);
    if (!p) return;
    ctx.fillStyle = color;
    const paso = 3 * dpr, ancho = Math.max(1, 2 * dpr);
    const mitad = h / 2, escala = mitad * 0.94;
    const n = p.min.length;
    for (let x = 0; x < w; x += paso) {
      const a = Math.floor(x / w * n), b = Math.max(a + 1, Math.floor((x + paso) / w * n));
      let lo = 0, hi = 0;
      for (let i = a; i < b && i < n; i++) {
        if (p.min[i] < lo) lo = p.min[i];
        if (p.max[i] > hi) hi = p.max[i];
      }
      const y0 = mitad - Math.min(1, hi) * escala;
      const y1 = mitad - Math.max(-1, lo) * escala;
      ctx.fillRect(x, y0, ancho, Math.max(dpr, y1 - y0));
    }
  }

  function dibujarEspectro(lienzo, e) {
    const { ctx, w, h } = ajustar(lienzo);
    ctx.clearRect(0, 0, w, h);
    if (!e) return;
    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = "high";
    ctx.drawImage(e.lienzo, 0, 0, w, h);
  }

  /* Posicion vertical (0 arriba, 1 abajo) de una frecuencia en el eje. */
  function alturaDe(f, e) {
    return 1 - Math.log(f / e.fMin) / Math.log(e.fMax / e.fMin);
  }

  return { preparar, dibujarOnda, dibujarEspectro, alturaDe };
})();

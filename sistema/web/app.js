/* ════════════════════════════════════════════════════════════════
   sonar · aplicacion web

   Se ejecuta en el navegador y habla con el servicio de analisis a
   traves del servidor web, que es el unico punto de entrada del
   sistema (RNF-07). Nunca llama al servicio directamente.

   El analisis es asincrono: se envia la pista, se recibe un
   identificador y se consulta el avance hasta que el reporte esta
   listo (RF-07).

   El audio se reproduce y se dibuja LOCALMENTE, desde el archivo que
   el usuario eligio, sin volver a pedirlo al servidor (senal.js).
   ════════════════════════════════════════════════════════════════ */

const API = "/api";
const SONDEO_MS = 1000;          // RF-07 pide al menos uno cada 5 s

// Orden fijo de los cuatro defectos en toda la interfaz.
const DEFECTOS = ["distorsion", "hiss", "clic", "hum"];

const ROTULO = {
  distorsion: "Distorsión",
  hiss: "Ruido de fondo",
  clic: "Clics",
  hum: "Zumbido",
};

// Lo que significa cada clase del modelo; no describe una medicion.
const DESCRIPCION = {
  distorsion: "Fragmentos compatibles con saturación o recorte: armónicos añadidos "
            + "y picos aplanados. Revisa ganancias, limitadores y plugins en este tramo.",
  hiss: "Ruido de banda ancha, más presente en los agudos, compatible con soplido "
      + "de fondo. Suele notarse más en pasajes suaves o en colas de reverberación.",
  clic: "Transitorio breve y abrupto, compatible con un clic de edición o un "
      + "chasquido. Puede durar milisegundos: escucha el tramo con atención.",
  hum: "Componente tonal sostenido en graves, compatible con zumbido eléctrico "
     + "de 50 o 60 Hz y sus armónicos.",
};

const $ = (id) => document.getElementById(id);
const PANTALLAS = { "pantalla-carga": "carga", "pantalla-progreso": "progreso",
                    "pantalla-reporte": "reporte" };

let reporte = null;        // reporte en pantalla
let idActual = null;       // sesion abierta en el servicio
let sondeo = null;         // temporizador de consulta de estado
let urlAudio = null;       // object URL del archivo local
let ordenCol = "inicio_s";
let ordenAsc = true;

let senal = null;          // forma de onda y espectrograma del archivo local
let senalFallo = false;
let turno = 0;             // descarta lecturas de un archivo anterior
let seleccion = -1;        // hallazgo en el panel de detalle
let tramoSonando = null;   // hallazgo que se escucha como intervalo

/* ─────────────────────────────────────────────────────── utiles ── */

function mostrar(id) {
  Object.keys(PANTALLAS).forEach((p) => { $(p).hidden = (p !== id); });
  document.body.dataset.pantalla = PANTALLAS[id];
  window.scrollTo({ top: 0, behavior: "smooth" });
}

function tiempo(s) {
  if (!isFinite(s)) return "0:00";
  const m = Math.floor(s / 60);
  const r = Math.floor(s % 60);
  return `${m}:${String(r).padStart(2, "0")}`;
}

// 00:01:12.4 → "01:12.4": decimas, como en una estacion de trabajo
function tiempoPreciso(s) {
  const t = Math.round(s * 10) / 10;
  const m = Math.floor(t / 60);
  return `${String(m).padStart(2, "0")}:${(t - 60 * m).toFixed(1).padStart(4, "0")}`;
}

function duracionCorta(s) {
  return s < 60 ? `${s.toFixed(1)} s` : tiempo(s);
}

function megas(bytes) {
  return bytes < 1e6 ? `${Math.max(1, Math.round(bytes / 1e3))} KB` : `${(bytes / 1e6).toFixed(bytes < 1e7 ? 1 : 0)} MB`;
}

function nivelConfianza(c) {
  if (c >= 0.85) return { barras: 3, palabra: "Alta" };
  if (c >= 0.70) return { barras: 2, palabra: "Media" };
  return { barras: 1, palabra: "Baja" };
}

function porcentaje(c) { return `${Math.round(c * 100)} %`; }

function colorVar(nombre) {
  return getComputedStyle(document.documentElement).getPropertyValue(nombre).trim();
}

function error(msg) {
  $("error-texto").textContent = msg;
  $("error-carga").hidden = false;
  mostrar("pantalla-carga");
  // en pantallas bajas el mensaje queda debajo de la zona de carga
  setTimeout(() => $("error-carga").scrollIntoView({ block: "nearest", behavior: "smooth" }), 60);
}

function limpiarError() { $("error-carga").hidden = true; }

/* ───────────────────────────────────────── salud del servicio ─── */

async function revisarSalud() {
  const s = $("estado-servicio");
  try {
    const r = await fetch(`${API}/salud`);
    const d = await r.json();
    if (d.estado !== "ok") {
      s.textContent = "El servicio está en pie pero no tiene modelo cargado, "
                    + "así que todavía no puede analizar pistas.";
      s.hidden = false;
    }
  } catch {
    s.textContent = "No se pudo contactar al servicio de análisis.";
    s.hidden = false;
  }
}

/* ─────────────────────────────────────────── carga y analisis ─── */

/* Revision local previa: evita subir 200 MB para que el servidor
   los rechace. La validacion que vale es la del servicio (RF-02). */
function revisarLocal(f) {
  if (!/\.wav$/i.test(f.name) && !/wav/i.test(f.type)) {
    return "Solo se admiten archivos WAV sin comprimir.";
  }
  if (f.size > 200 * 1024 * 1024) {
    return `El archivo pesa ${(f.size / 1e6).toFixed(0)} MB y el máximo son 200 MB.`;
  }
  if (f.size < 1024) return "El archivo está vacío o es demasiado pequeño.";
  return null;
}

async function analizar(f) {
  const fallo = revisarLocal(f);
  if (fallo) return error(fallo);

  limpiarError();
  $("nombre-pista").textContent = `${f.name} · ${megas(f.size)}`;
  $("nombre-pista").dataset.nombre = f.name;
  $("cabecera-archivo").textContent = f.name;
  $("texto-detalle").textContent = "";
  fijarEtapa("enviando");
  fijarAnillo(null, "Enviando…");
  mostrar("pantalla-progreso");

  // el audio local queda listo para el reproductor del reporte
  if (urlAudio) URL.revokeObjectURL(urlAudio);
  urlAudio = URL.createObjectURL(f);
  leerSenal(f);

  const cuerpo = new FormData();
  cuerpo.append("archivo", f, f.name);

  let r;
  try {
    r = await fetch(`${API}/analisis`, { method: "POST", body: cuerpo });
  } catch {
    return error("No se pudo contactar al servicio de análisis.");
  }

  let datos;
  try { datos = await r.json(); } catch { datos = {}; }

  if (!r.ok) {
    return error(datos.detail || "El servicio rechazó el archivo.");
  }

  idActual = datos.id;
  $("texto-detalle").textContent =
    `${tiempo(datos.duracion_s)} de audio · ${datos.frecuencia_original_hz} Hz · `
    + (datos.canales_original === 1 ? "mono" : datos.canales_original === 2
       ? "estéreo" : `${datos.canales_original} canales`);
  fijarEtapa("en_cola");
  fijarAnillo(0, "En cola");

  seguir(datos.id);
}

/* `fraccion` null deja el anillo en modo indeterminado: no se inventa
   un avance que el servicio todavia no informo. */
function fijarAnillo(fraccion, rotulo) {
  const anillo = $("anillo");
  const circunferencia = 2 * Math.PI * 72;
  if (rotulo) $("texto-etapa").textContent = rotulo;
  if (fraccion === null) {
    anillo.dataset.modo = "indeterminado";
    anillo.removeAttribute("aria-valuenow");
    return;
  }
  anillo.dataset.modo = "real";
  $("anillo-avance").style.strokeDashoffset = circunferencia * (1 - fraccion);
  $("porcentaje").textContent = Math.round(fraccion * 100);
  anillo.setAttribute("aria-valuenow", Math.round(fraccion * 100));
}

/* Las etapas son las del servicio (main.py · ETAPAS); el primer paso
   es el envio, que ocurre antes de que exista la tarea. */
const ORDEN_ETAPAS = ["carga", "preparando", "clasificando", "consolidando"];

function fijarEtapa(etapa) {
  const activa = etapa === "enviando" ? 0
               : etapa === "en_cola" ? 1
               : Math.max(1, ORDEN_ETAPAS.indexOf(etapa));
  document.querySelectorAll("#etapas li").forEach((li, k) => {
    li.classList.toggle("hecha", k < activa);
    li.classList.toggle("activa", k === activa);
    if (k === activa) li.setAttribute("aria-current", "step");
    else li.removeAttribute("aria-current");
  });
  $("etapa-carga-titulo").textContent = etapa === "enviando" ? "Enviando archivo" : "Archivo cargado";
  $("etapa-preparando-detalle").textContent = etapa === "en_cola"
    ? "En espera: hay otro análisis en curso" : "Fragmentos de un segundo";
}

function seguir(id) {
  clearInterval(sondeo);
  sondeo = setInterval(async () => {
    let r, d;
    try {
      r = await fetch(`${API}/analisis/${id}`);
      d = await r.json();
    } catch {
      return;                       // un fallo puntual no aborta el seguimiento
    }

    if (!r.ok) {
      clearInterval(sondeo);
      idActual = null;
      return error(d.detail || "El análisis falló.");
    }

    if (d.estado === "en_proceso") {
      fijarEtapa(d.etapa);
      fijarAnillo(d.progreso || 0, d.etapa_texto || "Procesando…");
      return;
    }

    clearInterval(sondeo);
    fijarAnillo(1, "Listo");
    pintarReporte(d);
  }, SONDEO_MS);
}

/* ─────────────────────────────────────── senal del archivo local ── */

function leerSenal(f) {
  const mio = ++turno;
  senal = null;
  senalFallo = false;
  $("onda-progreso").hidden = true;

  Senal.preparar(f).then((s) => {
    if (mio !== turno) return;
    senal = s;
    pintarOndaProgreso();
    if (reporte) pintarVisor();
    s.listoEspectro.then(() => { if (mio === turno && reporte) pintarVisor(); })
                   .catch(() => {});
  }).catch(() => {
    if (mio !== turno) return;
    senalFallo = true;
    if (reporte) pintarVisor();
  });
}

function pintarOndaProgreso() {
  if (!senal || $("pantalla-progreso").hidden) return;
  $("onda-progreso").hidden = false;
  Senal.dibujarOnda($("lienzo-onda-progreso"), senal.picos, colorVar("--onda-tenue"));
  pintarRegla($("regla-progreso"), senal.duracion);
}

/* Marcas de tiempo a intervalos redondos; la ultima es la duracion. */
function pintarRegla(regla, dur) {
  const ancho = regla.clientWidth || 600;
  const maximo = Math.max(2, Math.floor(ancho / 72));
  const pasos = [1, 2, 5, 10, 15, 20, 30, 60, 120, 180, 300, 600];
  const paso = pasos.find((p) => dur / p <= maximo) || 900;
  // la ultima marca redonda cede si quedaria encima de la duracion
  const holgura = 64 / ancho * dur;
  const marcas = [];
  for (let t = 0; t < dur - holgura; t += paso) marcas.push(t);
  marcas.push(dur);
  regla.innerHTML = marcas.map((t, i) => {
    const borde = i === 0 ? " primera" : i === marcas.length - 1 ? " ultima" : "";
    return `<span class="marca-tiempo${borde}" style="left:${100 * t / dur}%">${tiempo(t)}</span>`;
  }).join("");
}

/* ──────────────────────────────────────────────────── reporte ─── */

function pintarReporte(rep) {
  reporte = rep;
  ordenCol = "inicio_s";
  ordenAsc = true;
  tramoSonando = null;

  const n = rep.hallazgos.length;
  $("cabecera-archivo").textContent = $("nombre-pista").dataset.nombre || "Pista analizada";
  $("resumen-reporte").textContent = n === 0
    ? `No se detectaron hallazgos en ${tiempo(rep.duracion_s)} de audio.`
    : `Se ${n === 1 ? "detectó 1 hallazgo" : `detectaron ${n} hallazgos`} con posibles `
      + `defectos en ${tiempo(rep.duracion_s)} de audio.`;
  if (rep.advertencia) $("texto-aviso").textContent = rep.advertencia;

  // ---- un indicador por defecto, con su numero real de hallazgos
  $("cifras").innerHTML = DEFECTOS.map((c) => {
    const propios = rep.hallazgos.filter((h) => h.clase === c);
    const seg = propios.reduce((a, h) => a + h.duracion_s, 0);
    return `<li class="cifra" data-clase="${c}"${propios.length ? "" : " data-vacia"}>
        <span class="cifra-valor">${propios.length}</span>
        <span class="cifra-texto">
          <span class="cifra-rotulo">${ROTULO[c]}</span>
          <span class="cifra-extra">${propios.length ? `${duracionCorta(seg)} en total` : "Sin hallazgos"}</span>
        </span>
      </li>`;
  }).join("");

  document.querySelectorAll("#tabla th").forEach((o) => o.removeAttribute("aria-sort"));
  document.querySelector('#tabla th[data-col="inicio_s"]').setAttribute("aria-sort", "ascending");
  $("orden-ayuda").textContent = "Ordenados por posición en la pista";

  pintarTramos(rep);
  pintarTabla();
  prepararAudio(rep);

  $("sin-hallazgos").hidden = n > 0;
  $("tabla").closest(".tabla-envoltura").hidden = n === 0;
  $("pie-sesion").textContent =
    `Analizado en ${rep.segundos_de_analisis} s · ${rep.n_segmentos} fragmentos de un segundo · `
    + "esta pista y su reporte se eliminan al cerrar la sesión.";

  mostrar("pantalla-reporte");
  seleccionar(n > 0 ? 0 : -1);
  pintarVisor();
}

function pintarTramos(rep) {
  const onda = $("tramos-onda"), espectro = $("tramos-espectro");
  onda.innerHTML = "";
  espectro.innerHTML = "";

  rep.hallazgos.forEach((h, i) => {
    const left = `${100 * h.inicio_s / rep.duracion_s}%`;
    const width = `${100 * h.duracion_s / rep.duracion_s}%`;

    const b = document.createElement("button");
    b.className = "tramo";
    b.type = "button";
    b.dataset.clase = h.clase;
    b.dataset.i = i;
    b.style.left = left;
    b.style.width = width;
    b.title = `${h.nombre} · ${tiempoPreciso(h.inicio_s)}–${tiempoPreciso(h.fin_s)}`;
    b.setAttribute("aria-label",
      `${h.nombre} desde ${tiempo(h.inicio_s)} hasta ${tiempo(h.fin_s)}`);
    b.addEventListener("click", (e) => { e.stopPropagation(); irA(i); });
    onda.appendChild(b);

    const caja = document.createElement("div");
    caja.className = "tramo";
    caja.dataset.clase = h.clase;
    caja.dataset.i = i;
    caja.style.left = left;
    caja.style.width = width;
    caja.addEventListener("click", (e) => { e.stopPropagation(); irA(i); });
    espectro.appendChild(caja);
  });
}

/* Dibuja lo que haya del archivo local; se vuelve a llamar cuando llega
   el espectrograma y cuando cambia el tamano de la ventana. */
function pintarVisor() {
  if (!reporte || $("pantalla-reporte").hidden) return;
  const visor = document.querySelector(".visor");
  const dur = reporte.duracion_s;

  pintarRegla($("regla"), dur);

  if (senalFallo) {
    visor.dataset.senal = "sin";
    Senal.dibujarOnda($("lienzo-onda"), null);
    $("visor-nota").textContent = "El navegador no pudo decodificar este WAV, así que la línea "
      + "de tiempo muestra solo los hallazgos. El análisis del servicio no se ve afectado.";
    return;
  }
  if (!senal) {
    visor.dataset.senal = "leyendo";
    $("visor-nota").textContent = "";
    return;
  }

  visor.dataset.senal = senal.espectro ? "completa" : "onda";
  Senal.dibujarOnda($("lienzo-onda"), senal.picos, colorVar("--onda"));
  if (senal.espectro) {
    Senal.dibujarEspectro($("lienzo-espectro"), senal.espectro);
    pintarEjeFrecuencia(senal.espectro);
  }
  $("visor-nota").textContent = "Forma de onda y espectrograma calculados en tu navegador a partir "
    + `del archivo original (${(senal.frecuencia / 1000).toFixed(0)} kHz, escala de frecuencia logarítmica). `
    + "Los intervalos coloreados son los hallazgos del modelo.";
}

function pintarEjeFrecuencia(e) {
  const marcas = [100, 500, 1000, 2000, 4000, 8000, 16000]
    .filter((f) => f > e.fMin * 1.6 && f < e.fMax * 0.97);
  $("eje-frec").innerHTML = marcas.map((f) =>
    `<span style="top:${100 * Senal.alturaDe(f, e)}%">${f >= 1000 ? `${f / 1000}k` : f}</span>`
  ).join("");
}

function pintarTabla() {
  const filas = [...reporte.hallazgos].sort((a, b) => {
    const x = a[ordenCol], y = b[ordenCol];
    const d = typeof x === "string" ? x.localeCompare(y) : x - y;
    return ordenAsc ? d : -d;
  });

  $("cuerpo-tabla").innerHTML = filas.map((h) => {
    const idx = reporte.hallazgos.indexOf(h);
    const { barras, palabra } = nivelConfianza(h.confianza);
    const barritas = [1, 2, 3].map((k) =>
      `<i class="conf-barra${k <= barras ? " activa" : ""}"></i>`).join("");
    return `<tr data-i="${idx}" data-clase="${h.clase}">
      <td><span class="pastilla"><i class="punto"></i>${h.nombre}</span></td>
      <td class="num">${tiempoPreciso(h.inicio_s)}</td>
      <td class="num">${tiempoPreciso(h.fin_s)}</td>
      <td class="num">${duracionCorta(h.duracion_s)}</td>
      <td><span class="conf" title="Confianza ${palabra.toLowerCase()}"><span class="conf-barras" aria-hidden="true">${barritas}</span>
          <span class="conf-texto"><span class="visualmente-oculto">${palabra}, </span>${porcentaje(h.confianza)}</span></span></td>
      <td><button class="btn-fila" type="button" data-i="${idx}"
                  aria-label="Escuchar ${h.nombre} desde ${tiempo(h.inicio_s)}">
          <svg class="icono-play" aria-hidden="true"><use href="#i-play"/></svg>
          <svg class="icono-pausa" aria-hidden="true" hidden><use href="#i-pausa"/></svg></button></td>
    </tr>`;
  }).join("");

  $("cuerpo-tabla").querySelectorAll(".btn-fila").forEach((b) => {
    b.addEventListener("click", (e) => { e.stopPropagation(); alternarHallazgo(Number(b.dataset.i)); });
  });
  $("cuerpo-tabla").querySelectorAll("tr").forEach((tr) => {
    tr.addEventListener("click", () => seleccionar(Number(tr.dataset.i)));
  });
  resaltar(seleccion);
  fijarIconos();
}

function resaltar(i) {
  document.querySelectorAll(".tramo").forEach((t) =>
    t.classList.toggle("activo", Number(t.dataset.i) === i));
  document.querySelectorAll("#cuerpo-tabla tr").forEach((tr) =>
    tr.classList.toggle("activa", Number(tr.dataset.i) === i));
  document.querySelector(".visor").classList.toggle("con-seleccion", i >= 0);
}

/* ───────────────────────────────────────────── panel de detalle ─── */

function seleccionar(i) {
  const n = reporte ? reporte.hallazgos.length : 0;
  seleccion = n ? Math.max(0, Math.min(n - 1, i)) : -1;
  resaltar(seleccion);

  $("detalle-cuerpo").hidden = seleccion < 0;
  $("detalle-nav").hidden = seleccion < 0;
  $("detalle-vacio").hidden = seleccion >= 0;
  if (seleccion < 0) return;

  const h = reporte.hallazgos[seleccion];
  const { palabra } = nivelConfianza(h.confianza);
  $("detalle").dataset.clase = h.clase;
  $("det-pos").textContent = `${seleccion + 1} de ${n}`;
  $("det-ant").disabled = seleccion === 0;
  $("det-sig").disabled = seleccion === n - 1;
  $("det-tipo").querySelector("span").textContent = h.nombre;
  $("det-inicio").textContent = tiempoPreciso(h.inicio_s);
  $("det-fin").textContent = tiempoPreciso(h.fin_s);
  $("det-duracion").textContent = duracionCorta(h.duracion_s);
  $("det-confianza").textContent = `${porcentaje(h.confianza)} · ${palabra.toLowerCase()}`;
  $("det-segmentos").textContent = h.n_segmentos;
  $("det-descripcion").textContent = DESCRIPCION[h.clase] || "";
  $("det-play").disabled = !urlAudio;
  actualizarTiempoDetalle();
  fijarIconos();
}

function actualizarTiempoDetalle() {
  if (!reporte || seleccion < 0) return;
  const h = reporte.hallazgos[seleccion];
  const audio = $("audio");
  const dentro = tramoSonando === seleccion
    ? Math.max(0, Math.min(h.duracion_s, audio.currentTime - h.inicio_s)) : 0;
  $("det-tiempo").textContent = `${tiempo(dentro)} / ${tiempo(h.duracion_s)}`;
}

/* ──────────────────────────────────────────────── reproductor ─── */

function prepararAudio(rep) {
  const audio = $("audio");
  const play = $("btn-play");

  $("tiempo-actual").textContent = "0:00";
  $("playhead").hidden = true;

  if (!urlAudio) { play.disabled = true; return; }
  play.disabled = false;
  audio.src = urlAudio;

  audio.ontimeupdate = () => {
    if (reporte !== rep) return;
    // un hallazgo se escucha solo hasta su fin
    if (tramoSonando !== null) {
      const h = rep.hallazgos[tramoSonando];
      if (h && audio.currentTime >= h.fin_s) {
        audio.pause();
        audio.currentTime = h.fin_s;
      }
    }
    const f = audio.currentTime / (audio.duration || rep.duracion_s);
    const ph = $("playhead");
    ph.hidden = false;
    ph.style.left = `${100 * Math.min(1, f)}%`;
    $("tiempo-actual").textContent = tiempo(audio.currentTime);
    actualizarTiempoDetalle();
  };
  audio.onplay = fijarIconos;
  audio.onpause = fijarIconos;
  audio.onended = () => { tramoSonando = null; fijarIconos(); };
}

function fijarIcono(boton, sonando) {
  boton.querySelector(".icono-play").hidden = sonando;
  boton.querySelector(".icono-pausa").hidden = !sonando;
}

function fijarIconos() {
  const sonando = !$("audio").paused;
  fijarIcono($("btn-play"), sonando && tramoSonando === null);
  $("btn-play").setAttribute("aria-label",
    sonando && tramoSonando === null ? "Pausar la pista" : "Reproducir la pista");
  fijarIcono($("det-play"), sonando && tramoSonando === seleccion);
  $("det-play").setAttribute("aria-label",
    sonando && tramoSonando === seleccion ? "Pausar este hallazgo" : "Escuchar este hallazgo");
  document.querySelectorAll(".btn-fila").forEach((b) =>
    fijarIcono(b, sonando && tramoSonando === Number(b.dataset.i)));
}

function reproducir() {
  $("audio").play().catch(() => {});
}

/* Escucha un hallazgo de principio a fin. */
function irA(i) {
  if (!reporte) return;
  seleccionar(i);
  if (!urlAudio) return;
  const h = reporte.hallazgos[i];
  tramoSonando = i;
  $("audio").currentTime = h.inicio_s;
  reproducir();
}

function alternarHallazgo(i) {
  const audio = $("audio");
  if (!audio.paused && tramoSonando === i) { audio.pause(); return; }
  irA(i);
}

/* ───────────────────────────────────────────────── cerrar sesion ─ */

function cerrarSesion() {
  if (!idActual) return;
  const id = idActual;
  idActual = null;
  // keepalive: tambien sale si la pestana se esta cerrando
  fetch(`${API}/analisis/${id}`, { method: "DELETE", keepalive: true })
    .catch(() => {});
}

function volverAlInicio() {
  clearInterval(sondeo);
  $("audio").pause();
  tramoSonando = null;
  cerrarSesion();
  reporte = null;
  seleccion = -1;
  turno++;                         // ignora lecturas pendientes
  senal = null;
  limpiarError();
  mostrar("pantalla-carga");
}

/* ──────────────────────────────────────────────────── eventos ─── */

const zona = $("zona");

["dragenter", "dragover"].forEach((ev) =>
  zona.addEventListener(ev, (e) => {
    e.preventDefault();
    zona.classList.add("encima");
  }));
["dragleave", "drop"].forEach((ev) =>
  zona.addEventListener(ev, (e) => {
    e.preventDefault();
    zona.classList.remove("encima");
  }));

zona.addEventListener("drop", (e) => {
  const f = e.dataTransfer.files[0];
  if (f) analizar(f);
});
zona.addEventListener("click", () => $("archivo").click());
zona.addEventListener("keydown", (e) => {
  if (e.key === "Enter" || e.key === " ") { e.preventDefault(); $("archivo").click(); }
});

// Soltar un archivo fuera de la zona no debe abrirlo en la pestana.
["dragover", "drop"].forEach((ev) =>
  window.addEventListener(ev, (e) => { if (!zona.contains(e.target)) e.preventDefault(); }));

$("archivo").addEventListener("change", (e) => {
  const f = e.target.files[0];
  if (f) analizar(f);
  e.target.value = "";             // permite reelegir el mismo archivo
});

document.querySelectorAll("#tabla th[data-col]").forEach((th) => {
  th.addEventListener("click", () => {
    const col = th.dataset.col;
    ordenAsc = (col === ordenCol) ? !ordenAsc : true;
    ordenCol = col;
    document.querySelectorAll("#tabla th").forEach((o) => o.removeAttribute("aria-sort"));
    th.setAttribute("aria-sort", ordenAsc ? "ascending" : "descending");
    $("orden-ayuda").textContent = "Ordenados por " + th.textContent.trim().toLowerCase();
    pintarTabla();
  });
});

$("btn-play").addEventListener("click", () => {
  const audio = $("audio");
  if (!audio.paused && tramoSonando === null) { audio.pause(); return; }
  tramoSonando = null;
  if (audio.ended) audio.currentTime = 0;
  reproducir();
  fijarIconos();
});

$("linea").addEventListener("click", (e) => {
  if (!urlAudio || !reporte) return;
  const caja = $("linea").getBoundingClientRect();
  const f = (e.clientX - caja.left) / caja.width;
  const audio = $("audio");
  tramoSonando = null;
  audio.currentTime = Math.max(0, f * (audio.duration || reporte.duracion_s));
  reproducir();
});

$("det-play").addEventListener("click", () => { if (seleccion >= 0) alternarHallazgo(seleccion); });
$("det-ant").addEventListener("click", () => seleccionar(seleccion - 1));
$("det-sig").addEventListener("click", () => seleccionar(seleccion + 1));
$("volumen").addEventListener("input", (e) => { $("audio").volume = Number(e.target.value); });

$("btn-cancelar").addEventListener("click", volverAlInicio);
$("btn-otra").addEventListener("click", volverAlInicio);

$("nav-analizar").addEventListener("click", () => {
  const actual = document.body.dataset.pantalla;
  if (actual === "reporte") volverAlInicio();
  else if (actual === "carga") zona.focus();
  // durante el analisis no hace nada: cancelar es una accion explicita
});

const dialogo = $("dialogo-ayuda");
$("nav-ayuda").addEventListener("click", () => dialogo.showModal());
$("ayuda-cerrar").addEventListener("click", () => dialogo.close());
dialogo.addEventListener("click", (e) => { if (e.target === dialogo) dialogo.close(); });

/* RF-09 · el reporte se lleva en un archivo estructurado */
$("btn-exportar").addEventListener("click", () => {
  if (!reporte) return;
  const blob = new Blob([JSON.stringify(reporte, null, 2)],
                        { type: "application/json" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = `reporte-${reporte.id.slice(0, 8)}.json`;
  a.click();
  URL.revokeObjectURL(a.href);
});

// Los lienzos se redibujan al cambiar de tamano.
let redibujo = 0;
new ResizeObserver(() => {
  cancelAnimationFrame(redibujo);
  redibujo = requestAnimationFrame(() => { pintarVisor(); pintarOndaProgreso(); });
}).observe(document.querySelector(".hoja"));

/* RF-10 · al abandonar la pagina la sesion se cierra */
window.addEventListener("pagehide", cerrarSesion);

revisarSalud();

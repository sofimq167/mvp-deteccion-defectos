/* Aplicación web: carga, seguimiento y presentación del reporte.
   Se ejecuta en el navegador y habla con el servicio a través del
   servidor web, que es el único punto de entrada del sistema. */

const API = "/api";

const COLOR = {
  sin_defecto: "var(--sin-defecto)",
  hum: "var(--hum)",
  hiss: "var(--hiss)",
  distorsion: "var(--distorsion)",
  clic: "var(--clic)",
};

const $ = (id) => document.getElementById(id);
const pantallas = ["pantalla-carga", "pantalla-progreso", "pantalla-reporte"];

let reporteActual = null;
let ordenCol = "inicio_s";
let ordenAsc = true;

function mostrar(id) {
  pantallas.forEach((p) => { $(p).hidden = (p !== id); });
}

function tiempo(s) {
  const m = Math.floor(s / 60);
  const r = (s % 60).toFixed(1).padStart(4, "0");
  return `${m}:${r}`;
}

/* ------------------------------------------------------- validación previa */
/* Se comprueba en el navegador antes de enviar, para que un rechazo por
   formato o tamaño ocurra de inmediato y no tras subir el archivo. */
function revisarLocal(f) {
  if (!/\.wav$/i.test(f.name)) {
    return "El archivo debe ser un WAV. Exportá la pista en ese formato desde tu estación de trabajo.";
  }
  if (f.size > 200 * 1024 * 1024) {
    return `El archivo pesa ${(f.size / 1e6).toFixed(0)} MB y el máximo admitido es 200 MB.`;
  }
  if (f.size < 1024) {
    return "El archivo está vacío o es demasiado pequeño para contener audio.";
  }
  return null;
}

/* ------------------------------------------------------------------ envío */
async function analizar(f) {
  const problema = revisarLocal(f);
  if (problema) { error(problema); return; }

  $("error-carga").hidden = true;
  $("nombre-pista").textContent = f.name;
  mostrar("pantalla-progreso");

  // El servicio responde cuando termina; el avance se estima a partir de
  // la duración típica de análisis, y se completa al llegar la respuesta.
  let pct = 0;
  $("barra-avance").style.width = "0%";
  $("texto-progreso").textContent = "Enviando la pista…";
  const reloj = setInterval(() => {
    pct = Math.min(pct + 2, 92);
    $("barra-avance").style.width = pct + "%";
    if (pct > 10) $("texto-progreso").textContent = "Analizando los segmentos…";
  }, 700);

  const cuerpo = new FormData();
  cuerpo.append("archivo", f);

  try {
    const r = await fetch(`${API}/analisis`, { method: "POST", body: cuerpo });
    clearInterval(reloj);
    if (!r.ok) {
      const d = await r.json().catch(() => ({}));
      error(d.detail || `El servicio respondió con un error (${r.status}).`);
      mostrar("pantalla-carga");
      return;
    }
    $("barra-avance").style.width = "100%";
    reporteActual = await r.json();
    pintarReporte(reporteActual);
    mostrar("pantalla-reporte");
  } catch (e) {
    clearInterval(reloj);
    error("No se pudo contactar con el servicio. Revisá que esté en ejecución.");
    mostrar("pantalla-carga");
  }
}

function error(msg) {
  const e = $("error-carga");
  e.textContent = msg;
  e.hidden = false;
}

/* ---------------------------------------------------------------- reporte */
function pintarReporte(rep) {
  $("titulo-reporte").textContent =
    `Reporte · ${tiempo(rep.duracion_s)} de audio`;

  const n = rep.hallazgos.length;
  $("resumen").textContent = n === 0
    ? `Se analizaron ${rep.n_segmentos} segmentos y no se señaló ninguno.`
    : `Se analizaron ${rep.n_segmentos} segmentos; ${n} ${n === 1 ? "hallazgo" : "hallazgos"} en ${rep.segundos_de_analisis} s de cómputo.`;

  // línea de tiempo
  const linea = $("linea");
  linea.innerHTML = "";
  rep.hallazgos.forEach((h, i) => {
    const d = document.createElement("div");
    d.className = "tramo";
    d.style.left = (100 * h.inicio_s / rep.duracion_s) + "%";
    d.style.width = Math.max(0.6, 100 * h.duracion_s / rep.duracion_s) + "%";
    d.style.background = COLOR[h.clase];
    d.tabIndex = 0;
    d.title = `${h.nombre} · ${tiempo(h.inicio_s)} a ${tiempo(h.fin_s)}`;
    d.addEventListener("click", () => resaltarFila(i));
    linea.appendChild(d);
  });

  $("marcas").innerHTML = `<span>0:00.0</span><span>${tiempo(rep.duracion_s)}</span>`;

  const presentes = [...new Set(rep.hallazgos.map((h) => h.clase))];
  $("leyenda").innerHTML = presentes.length === 0 ? "" :
    presentes.map((c) => {
      const nom = rep.hallazgos.find((h) => h.clase === c).nombre;
      return `<span><i class="muestra" style="background:${COLOR[c]}"></i>${nom}</span>`;
    }).join("");

  $("sin-hallazgos").hidden = n > 0;
  pintarTabla();
}

function barrasConfianza(c) {
  // Graduada en cuatro niveles: una cifra con decimales sugeriría una
  // precisión que el modelo no tiene.
  const nivel = c >= 0.9 ? 4 : c >= 0.75 ? 3 : c >= 0.55 ? 2 : 1;
  const etiqueta = ["baja", "media", "alta", "muy alta"][nivel - 1];
  const barras = [1, 2, 3, 4]
    .map((i) => `<i class="conf-barra ${i <= nivel ? "activa" : ""}"></i>`).join("");
  return `<span class="conf"><span class="conf-barras">${barras}</span>${etiqueta}</span>`;
}

function pintarTabla() {
  if (!reporteActual) return;
  const filas = [...reporteActual.hallazgos].sort((a, b) => {
    const x = a[ordenCol], y = b[ordenCol];
    const c = typeof x === "string" ? x.localeCompare(y) : x - y;
    return ordenAsc ? c : -c;
  });

  $("cuerpo-tabla").innerHTML = filas.map((h, i) => `
    <tr id="fila-${i}">
      <td>${tiempo(h.inicio_s)}</td>
      <td>${tiempo(h.fin_s)}</td>
      <td><span class="pastilla"><i class="muestra" style="background:${COLOR[h.clase]}"></i>${h.nombre}</span></td>
      <td>${h.duracion_s.toFixed(1)} s</td>
      <td>${barrasConfianza(h.confianza)}</td>
    </tr>`).join("");
}

function resaltarFila(i) {
  const f = $(`fila-${i}`);
  if (!f) return;
  f.scrollIntoView({ behavior: "smooth", block: "center" });
  f.style.background = "#fff3cd";
  setTimeout(() => { f.style.background = ""; }, 1200);
}

/* ------------------------------------------------------------- eventos */
const zona = $("zona");
["dragenter", "dragover"].forEach((ev) =>
  zona.addEventListener(ev, (e) => { e.preventDefault(); zona.classList.add("encima"); }));
["dragleave", "drop"].forEach((ev) =>
  zona.addEventListener(ev, (e) => { e.preventDefault(); zona.classList.remove("encima"); }));
zona.addEventListener("drop", (e) => {
  if (e.dataTransfer.files.length) analizar(e.dataTransfer.files[0]);
});
$("archivo").addEventListener("change", (e) => {
  if (e.target.files.length) analizar(e.target.files[0]);
});

document.querySelectorAll("#tabla th").forEach((th) => {
  th.addEventListener("click", () => {
    const c = th.dataset.col;
    ordenAsc = (c === ordenCol) ? !ordenAsc : true;
    ordenCol = c;
    pintarTabla();
  });
});

$("btn-otra").addEventListener("click", async () => {
  // Cierre explícito de la sesión: el material se elimina del servidor.
  if (reporteActual) {
    fetch(`${API}/analisis/${reporteActual.id}`, { method: "DELETE" }).catch(() => {});
  }
  reporteActual = null;
  $("archivo").value = "";
  mostrar("pantalla-carga");
});

$("btn-exportar").addEventListener("click", () => {
  if (!reporteActual) return;
  const blob = new Blob([JSON.stringify(reporteActual, null, 2)],
                        { type: "application/json" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = `reporte-${reporteActual.id.slice(0, 8)}.json`;
  a.click();
  URL.revokeObjectURL(a.href);
});

mostrar("pantalla-carga");

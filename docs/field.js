"use strict";

function setupCopy() {
  const button = document.querySelector("#copy");
  const command = document.querySelector("#install-cmd");
  const status = document.querySelector("#copy-status");
  let reset;
  button.hidden = false;
  button.addEventListener("click", async () => {
    let message = "Copied";
    try {
      await navigator.clipboard.writeText(command.textContent);
    } catch {
      const range = document.createRange();
      range.selectNodeContents(command);
      const selection = window.getSelection();
      selection.removeAllRanges();
      selection.addRange(range);
      message = "Selected";
    }
    clearTimeout(reset);
    status.textContent = message === "Copied" ? "Install command copied" : "Install command selected for copying";
    button.textContent = message;
    reset = setTimeout(() => {
      button.textContent = "Copy";
      status.textContent = "";
    }, 1800);
  });
}

const TAU = 2 * Math.PI;
const smoothstep = t => t <= 0 ? 0 : t >= 1 ? 1 : t * t * (3 - 2 * t);

// Lay one schema sheet out as a card of "key: value" bars, so the middle
// shape reads as a list of typed fields rather than a blank plane.
// The three sheets fan open around their left edge like a hand of cards.
function fieldRow(sheet, x, y) {
  const bands = 6;
  const band = Math.min(bands - 1, Math.floor((y + 1) / 2 * bands));
  const across = (y + 1) / 2 * bands - band - .5;
  const keyLength = .42;
  const valueLength = .5 + .8 * ((sheet * 7 + band * 5) % 9) / 8;
  const gap = .1;
  const along = (x + 1) / 2 * (keyLength + valueLength);
  const reach = along + (along > keyLength ? gap : 0);
  const angle = (sheet - 1) * .62;
  return {
    position: [
      -.85 + reach * Math.cos(angle),
      ((bands - 1) / 2 - band) * .26 - across * .1,
      reach * Math.sin(angle),
    ],
    key: along <= keyLength,
  };
}

// Each sample has a corresponding position on the source ribbon, schema sheets,
// and output cube so that scrolling moves the same material through all three forms.
function makeField() {
  const points = [];
  const rows = 360;
  const columns = 60;
  let seed = 137;
  function noise() {
    seed = (Math.imul(1664525, seed) + 1013904223) >>> 0;
    return seed / 4294967296;
  }
  for (let row = 0; row < rows; row++) {
    const u = row / rows * TAU;
    // Sweep a tube around a trefoil. Its radial and binormal vectors keep
    // the particle rows continuous as they wind around the crossings.
    const radius = .81 + .36 * Math.cos(3 * u);
    const cu = Math.cos(2 * u), su = Math.sin(2 * u);
    const dx = -1.08 * Math.sin(3 * u) * cu - 2 * radius * su;
    const dy = -1.08 * Math.sin(3 * u) * su + 2 * radius * cu;
    const dz = 1.53 * Math.cos(3 * u);
    const length = Math.hypot(dx, dy, dz);
    const horizontal = Math.hypot(dx, dy);
    const nx = dy / horizontal, ny = -dx / horizontal;
    const bx = -dz * ny / length, by = dz * nx / length;
    const bz = -horizontal / length;
    for (let col = 0; col < columns; col++) {
      const v = col / columns * TAU;
      const cv = Math.cos(v), sv = Math.sin(v);
      const debris = noise() < .035 ? noise() * .35 : 0;
      const tube = .265 + debris;
      const x = radius * cu + tube * (nx * cv + bx * sv);
      const y = radius * su + tube * (ny * cv + by * sv);
      const z = .51 * Math.sin(3 * u) + tube * bz * sv;

      const sheet = Math.floor(row / (rows / 3));
      const sheetX = row % (rows / 3) / (rows / 3 - 1) * 2 - 1;
      const sheetY = col / (columns - 1) * 2 - 1;
      const field = fieldRow(sheet, sheetX, sheetY);

      const face = Math.floor(row / (rows / 6));
      const a = row % (rows / 6) / (rows / 6 - 1) * 2 - 1;
      const side = face % 2 ? 1 : -1;
      const cube = (face < 2 ? [side, a, sheetY] : face < 4 ? [a, side, sheetY] : [a, sheetY, side])
        .map(c => c * .82);

      points.push({
        source: [x, y, z], schema: field.position, cube,
        light: .50 + .26 * (nx * cv + bx * sv) - .25 * (ny * cv + by * sv) + .18 * bz * sv + .12 * noise(),
        stripe: col % 5 === 0, debris,
        // Accents that only show in their own shape: field keys and cube edges.
        key: field.key, edge: Math.max(Math.abs(a), Math.abs(sheetY)) > .96,
        // Where along the ribbon the sample sits: drives the flowing light
        // pulse and the order in which samples leave for the next shape.
        phase: row / rows, delay: .55 * (row / rows) + .1 * noise(), swirl: noise() * 2 - 1,
      });
    }
  }
  return points;
}

function setupField() {
  const canvas = document.querySelector("#field");
  const ctx = canvas.getContext("2d");
  if (!ctx) return;

  const scroller = document.querySelector("main");
  const story = document.querySelector("#story");
  const quickstart = document.querySelector("#quickstart");
  const rail = document.querySelector(".rail");
  const buttons = [...rail.querySelectorAll(".phases button")];
  const meter = document.querySelector("#field-progress");
  const title = document.querySelector("#title");
  const lede = document.querySelector(".lede");
  const specimen = document.querySelector("#specimen-name");
  const value = document.querySelector("#specimen-value");
  const quickstartLink = document.querySelector('nav a[href="#quickstart"]');
  const scenes = [
    ["Files ", "in.", lede.textContent, specimen.textContent, value.textContent],
    ["Give it ", "shape.", "Define the fields you need and their types in a Pydantic model.", "Schema · Invoice", "total: Decimal"],
    ["Structure ", "out.", "Read the extracted total directly from your validated Invoice object.", "Object · invoice.total", 'Decimal("120.00")'],
  ];
  const lastScene = scenes.length;
  const reduce = matchMedia("(prefers-reduced-motion: reduce)");
  const points = makeField();
  // Particles ramp from a deep shade of the page accent, through the accent, to near white.
  const accent = getComputedStyle(canvas).getPropertyValue("--accent").trim().match(/\w\w/g).map(hex => parseInt(hex, 16));
  const palette = Array.from({length: 64}, (_, i) => {
    const t = i / 63;
    const toAccent = Math.min(1, t / .7) ** 1.3;
    const toWhite = Math.max(0, (t - .7) / .3) ** 1.4;
    return "rgb(" + accent.map(c => {
      const deep = c * .12;
      return Math.round(deep + (c - deep) * toAccent + (246 - c) * toWhite);
    }) + ")";
  });
  const glow = document.createElement("canvas");
  const glowCtx = glow.getContext("2d");
  let width, height, stageNumber;
  let progress = 0, shown = 0, pointerX = 0, pointerY = 0, viewX = 0, viewY = 0, frame = 0;

  function schedule() {
    if (!frame) frame = requestAnimationFrame(draw);
  }
  // Eases the drawn state toward the scroll position and pointer. Returns
  // whether anything is still settling, so idle frames can stop when motion is reduced.
  function settle() {
    const target = reduce.matches ? stageNumber : progress;
    shown += (target - shown) * (reduce.matches ? 1 : .14);
    viewX += (pointerX - viewX) * .08;
    viewY += (pointerY - viewY) * .08;
    if (Math.abs(target - shown) < .0005) shown = target;
    return shown !== target || Math.abs(pointerX - viewX) + Math.abs(pointerY - viewY) > .001;
  }
  function updateStage() {
    meter.value = progress;
    const next = Math.round(progress);
    if (next === stageNumber) return;
    const initial = stageNumber === undefined;
    stageNumber = next;
    const showExample = next === lastScene;
    const outgoing = showExample ? story : quickstart;
    if (outgoing.contains(document.activeElement)) scroller.focus({preventScroll: true});
    story.hidden = showExample;
    quickstart.hidden = !showExample;
    canvas.parentElement.classList.toggle("resting", showExample);
    if (!showExample) {
      const [opening, emphasis, description, source, sample] = scenes[next];
      const word = document.createElement("em");
      word.textContent = emphasis;
      title.replaceChildren(opening, word);
      lede.textContent = description;
      specimen.textContent = source;
      value.textContent = sample;
    }
    if (showExample) quickstartLink.setAttribute("aria-current", "location");
    else quickstartLink.removeAttribute("aria-current");
    buttons.forEach((button, i) => {
      if (i === next) button.setAttribute("aria-current", "step");
      else button.removeAttribute("aria-current");
    });
    if (!initial && !reduce.matches) {
      const entering = showExample ? [quickstart] : [title.parentElement, specimen.parentElement];
      for (const item of entering) {
        item.animate([{opacity: 0, transform: "translateY(10px)"}, {opacity: 1, transform: "none"}], {duration: 230});
      }
    }
  }
  function onScroll() {
    const distance = scroller.scrollHeight - scroller.clientHeight;
    progress = distance ? Math.max(0, Math.min(lastScene, scroller.scrollTop / distance * lastScene)) : 0;
    updateStage();
    schedule();
  }
  function resizeCanvas() {
    width = canvas.clientWidth;
    height = canvas.clientHeight;
    const ratio = Math.min(devicePixelRatio || 1, 2);
    canvas.width = Math.round(width * ratio);
    canvas.height = Math.round(height * ratio);
    ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
    glow.width = Math.ceil(canvas.width / 4);
    glow.height = Math.ceil(canvas.height / 4);
    schedule();
  }
  // Blur a quarter-size copy of the particles and add it back for a soft bloom.
  function bloom() {
    glowCtx.clearRect(0, 0, glow.width, glow.height);
    glowCtx.filter = "blur(3px)";
    glowCtx.drawImage(canvas, 0, 0, glow.width, glow.height);
    ctx.save();
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.globalCompositeOperation = "lighter";
    ctx.globalAlpha = .7;
    ctx.drawImage(glow, 0, 0, canvas.width, canvas.height);
    ctx.restore();
  }
  function draw(now) {
    frame = 0;
    if (!width || !height) return;
    const settling = settle();
    const time = reduce.matches ? 0 : now / 1000;
    ctx.clearRect(0, 0, width, height);
    ctx.globalCompositeOperation = "lighter";
    const shape = Math.min(2, shown);
    const part = Math.min(1, Math.floor(shape));
    const fraction = shape - part;
    const yaw = -.34 + shape * .54 + viewX * .24 + Math.sin(time * .17) * .22;
    const pitch = -.33 - shape * .09 + viewY * .21 + Math.sin(time * .13) * .05;
    const roll = -.46 + Math.min(shape, 1) * .54;
    const cy = Math.cos(yaw), sy = Math.sin(yaw);
    const cp = Math.cos(pitch), sp = Math.sin(pitch);
    const cr = Math.cos(roll), sr = Math.sin(roll);
    const scale = Math.min(width / 3.5, height / 3.7);
    const dot = Math.max(.72, Math.min(1.55, scale * .009));

    for (const p of points) {
      const from = part ? p.schema : p.source;
      const to = part ? p.cube : p.schema;
      // Samples leave in ribbon order and spiral around the vertical axis in transit.
      const mix = smoothstep((fraction - p.delay) / (1 - .65));
      const transit = Math.sin(Math.PI * mix);
      let x = from[0] + (to[0] - from[0]) * mix;
      let y = from[1] + (to[1] - from[1]) * mix;
      let z = from[2] + (to[2] - from[2]) * mix;
      if (transit > .001) {
        const turn = transit * (1.2 + .3 * p.swirl);
        const c = Math.cos(turn), s = Math.sin(turn);
        [x, z] = [x * c - z * s, x * s + z * c];
        y += transit * p.swirl * .14;
      }
      const rx = x * cy + z * sy;
      const rz = z * cy - x * sy;
      const ry = y * cp - rz * sp;
      const depth = y * sp + rz * cp;
      const perspective = 5.5 / (5.5 - depth);
      const px = width / 2 + (rx * cr - ry * sr) * perspective * scale;
      const py = height * .46 + (rx * sr + ry * cr) * perspective * scale;
      const pulse = Math.max(0, Math.sin((p.phase * 2 - time * .16) * TAU)) ** 12;
      const schemaWeight = part ? 1 - mix : mix;
      const cubeWeight = part ? mix : 0;
      const accent = schemaWeight * (p.key ? .14 : -.04) + cubeWeight * (p.edge ? .5 : -.16);
      const light = (.1 + p.light * .42 + depth * .2 + (p.stripe ? .16 : 0) - p.debris) *
        (1 + .25 * transit) + accent + pulse * .45;
      const size = dot * perspective * (p.stripe ? 1.13 : .86);
      ctx.fillStyle = palette[Math.round(Math.max(0, Math.min(1, light)) * 63)];
      ctx.fillRect(px, py, size, size);
    }
    bloom();
    if (settling || !reduce.matches) schedule();
  }

  function goToStage(i, instant = false) {
    const scene = Math.max(0, Math.min(lastScene, i));
    const distance = scroller.scrollHeight - scroller.clientHeight;
    scroller.scrollTo({top: distance * scene / lastScene, behavior: instant || reduce.matches ? "instant" : "smooth"});
  }
  buttons.forEach((button, i) => button.addEventListener("click", () => goToStage(i)));
  function followHash() {
    goToStage(location.hash === "#quickstart" ? lastScene : 0, true);
  }
  for (const link of document.querySelectorAll('a[href="#main"], a[href="#quickstart"]')) {
    link.addEventListener("click", event => {
      if (event.metaKey || event.ctrlKey || event.shiftKey || event.altKey) return;
      event.preventDefault();
      if (location.hash !== link.hash) history.pushState(null, "", link.hash);
      followHash();
      scroller.focus({preventScroll: true});
    });
  }
  window.addEventListener("hashchange", followHash);
  window.addEventListener("keydown", event => {
    if (event.altKey || event.ctrlKey || event.metaKey || event.defaultPrevented) return;
    // Keep space activation and local scrolling inside the code sample native.
    if (event.target.closest("pre") || (event.key === " " && event.target.closest("button, a"))) return;
    // On short screens the example has its own reading area. Let its
    // keyboard scrolling reach either edge before changing scenes.
    if (quickstart.contains(event.target)) {
      const goingUp = ["ArrowUp", "PageUp", "Home"].includes(event.key) || (event.key === " " && event.shiftKey);
      const goingDown = ["ArrowDown", "PageDown", "End"].includes(event.key) || (event.key === " " && !event.shiftKey);
      if (goingUp && quickstart.scrollTop > 0) return;
      if (goingDown && quickstart.scrollTop + quickstart.clientHeight < quickstart.scrollHeight - 1) return;
    }
    const destinations = {
      ArrowDown: stageNumber + 1, ArrowUp: stageNumber - 1,
      PageDown: stageNumber + 1, PageUp: stageNumber - 1,
      " ": stageNumber + (event.shiftKey ? -1 : 1), Home: 0, End: lastScene,
    };
    if (!Object.hasOwn(destinations, event.key)) return;
    event.preventDefault();
    goToStage(destinations[event.key]);
  });
  scroller.addEventListener("pointermove", event => {
    if (reduce.matches || event.pointerType === "touch") return;
    const rect = canvas.getBoundingClientRect();
    pointerX = (event.clientX - rect.left) / width * 2 - 1;
    pointerY = (event.clientY - rect.top) / height * 2 - 1;
    schedule();
  });
  function centerView() {
    pointerX = pointerY = 0;
    schedule();
  }
  scroller.addEventListener("pointerleave", centerView);
  reduce.addEventListener("change", centerView);
  scroller.addEventListener("scroll", onScroll, {passive: true});
  document.documentElement.classList.add("exhibit");
  rail.hidden = false;
  quickstart.hidden = true;
  new ResizeObserver(resizeCanvas).observe(canvas);
  new ResizeObserver(onScroll).observe(scroller);
  followHash();
  onScroll();
  resizeCanvas();
}

setupCopy();
setupField();

// fx.js: visual layer for the portal. The WebGL "neural sphere" background and the interaction effects.
// Progressive enhancement only: every page works without it. CSP-safe (no eval, no inline code, no external
// resources). Styles are applied through CSSOM only. app.js (form logic) is independent of this file.
(function () {
  "use strict";

  const doc = document;
  const root = doc.documentElement;
  const win = window;

  // Runs first so `.js`-scoped CSS (hide-before-reveal) applies as early as possible.
  root.classList.remove("no-js");
  root.classList.add("js");

  /* ------------------------------------------------------------------ environment */

  const PASSIVE = { passive: true };
  const now = () => (win.performance && performance.now ? performance.now() : Date.now());
  const raf = win.requestAnimationFrame ? win.requestAnimationFrame.bind(win) : (fn) => win.setTimeout(() => fn(now()), 16);
  const caf = win.cancelAnimationFrame ? win.cancelAnimationFrame.bind(win) : win.clearTimeout.bind(win);

  function media(q) {
    try { return win.matchMedia ? win.matchMedia(q) : null; } catch (e) { return null; }
  }
  function onMediaChange(m, fn) {
    if (!m) return;
    if (m.addEventListener) m.addEventListener("change", fn);
    else if (m.addListener) m.addListener(fn);
  }
  const reduceMQ = media("(prefers-reduced-motion: reduce)");
  const fineMQ = media("(hover: hover) and (pointer: fine)");
  const reduced = () => !!(reduceMQ && reduceMQ.matches);
  const finePointer = () => !!(fineMQ && fineMQ.matches);
  const isRefresh = () => !!(doc.body && doc.body.hasAttribute("data-refresh"));
  const clamp = (v, a, b) => (v < a ? a : v > b ? b : v);
  const easeOutCubic = (t) => 1 - Math.pow(1 - t, 3);
  const easeInOutCubic = (t) => (t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2);
  const css = (el, prop, val) => el.style.setProperty(prop, val);
  const isStatic = (el) => { try { return getComputedStyle(el).position === "static"; } catch (e) { return false; } };
  function safely(fn) {
    return function () { try { return fn.apply(this, arguments); } catch (e) { /* never break the page */ } };
  }

  /* ------------------------------------------------------------------ mat4 (column-major) */

  const M4 = {
    create() { const m = new Float32Array(16); m[0] = m[5] = m[10] = m[15] = 1; return m; },
    identity(m) { m.fill(0); m[0] = m[5] = m[10] = m[15] = 1; return m; },
    _a: new Float32Array(16),
    mul(out, a, b) { // out = a * b; out may alias a or b
      const A = M4._a; A.set(a);
      for (let c = 0; c < 4; c++) {
        const i = c * 4, b0 = b[i], b1 = b[i + 1], b2 = b[i + 2], b3 = b[i + 3];
        out[i] = A[0] * b0 + A[4] * b1 + A[8] * b2 + A[12] * b3;
        out[i + 1] = A[1] * b0 + A[5] * b1 + A[9] * b2 + A[13] * b3;
        out[i + 2] = A[2] * b0 + A[6] * b1 + A[10] * b2 + A[14] * b3;
        out[i + 3] = A[3] * b0 + A[7] * b1 + A[11] * b2 + A[15] * b3;
      }
      return out;
    },
    rotX(m, a) { M4.identity(m); const c = Math.cos(a), s = Math.sin(a); m[5] = c; m[6] = s; m[9] = -s; m[10] = c; return m; },
    rotY(m, a) { M4.identity(m); const c = Math.cos(a), s = Math.sin(a); m[0] = c; m[2] = -s; m[8] = s; m[10] = c; return m; },
    rotZ(m, a) { M4.identity(m); const c = Math.cos(a), s = Math.sin(a); m[0] = c; m[1] = s; m[4] = -s; m[5] = c; return m; },
    translate(m, x, y, z) { M4.identity(m); m[12] = x; m[13] = y; m[14] = z; return m; },
    perspective(m, fovy, aspect, near, far) {
      const f = 1 / Math.tan(fovy / 2), nf = 1 / (near - far);
      m.fill(0);
      m[0] = f / aspect; m[5] = f; m[10] = (far + near) * nf; m[11] = -1; m[14] = 2 * far * near * nf;
      return m;
    },
  };

  /* ------------------------------------------------------------------ geometry */

  const PALETTE = [[139, 92, 246], [99, 102, 241], [34, 211, 238]]; // violet -> indigo -> cyan
  const PINK = [244, 114, 182];
  const RINGS = [
    { R: 1.4, rx: 1.22, ry: 0.0, rz: 0.38, speed: 0.11, t0: 0.78, share: 0.38 },
    { R: 1.62, rx: -0.92, ry: 0.35, rz: -0.55, speed: -0.075, t0: 0.12, share: 0.36 },
    { R: 1.24, rx: 0.35, ry: 1.05, rz: 0.9, speed: 0.16, t0: 0.55, share: 0.26 },
  ];
  const FLOATS_PER_POINT = 8; // x y z r g b size seed

  function rng(seed) { // mulberry32: deterministic so every page load draws the same network
    let s = seed >>> 0;
    return function () {
      s = (s + 0x6d2b79f5) >>> 0;
      let t = s;
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  function paletteColor(t, out, o) {
    t = clamp(t, 0, 1) * (PALETTE.length - 1);
    const i = Math.min(Math.floor(t), PALETTE.length - 2), f = t - i, a = PALETTE[i], b = PALETTE[i + 1];
    out[o] = (a[0] + (b[0] - a[0]) * f) / 255;
    out[o + 1] = (a[1] + (b[1] - a[1]) * f) / 255;
    out[o + 2] = (a[2] + (b[2] - a[2]) * f) / 255;
  }

  function quality(low) {
    if (low) return { outer: 380, inner: 110, ring: 60, k: 2 };
    const w = win.innerWidth || 1280, h = win.innerHeight || 800;
    const cores = navigator.hardwareConcurrency || 4, mem = navigator.deviceMemory || 4;
    const small = Math.min(w, h) < 560 || w * h < 520000;
    let n = 1320;
    if (cores <= 4 || mem <= 2) n = 960;
    if (small) n = Math.min(n, 700);
    return { outer: n, inner: Math.round(n * 0.28), ring: Math.round(n * 0.12), k: small ? 2 : 3 };
  }

  // k-nearest-neighbour pairs inside [start, start+n) of the point buffer, deduplicated, distance-capped.
  // Fibonacci-sphere indices are sorted by height, so only an index window around i can be within maxD.
  function neighbourPairs(P, start, n, k, maxD, radius, out) {
    const span = Math.ceil(((maxD / radius) * 1.1 + 0.08) * n / 2) + 2;
    const seen = new Set(), bestD = new Float32Array(k), bestJ = new Int32Array(k), max2 = maxD * maxD;
    for (let i = 0; i < n; i++) {
      const oi = (start + i) * FLOATS_PER_POINT, x = P[oi], y = P[oi + 1], z = P[oi + 2];
      bestD.fill(Infinity); bestJ.fill(-1);
      for (let j = Math.max(0, i - span), jEnd = Math.min(n, i + span + 1); j < jEnd; j++) {
        if (j === i) continue;
        const oj = (start + j) * FLOATS_PER_POINT, dx = P[oj] - x, dy = P[oj + 1] - y, dz = P[oj + 2] - z;
        const d = dx * dx + dy * dy + dz * dz;
        if (d > max2 || d >= bestD[k - 1]) continue;
        let p = k - 1;
        while (p > 0 && bestD[p - 1] > d) { bestD[p] = bestD[p - 1]; bestJ[p] = bestJ[p - 1]; p--; }
        bestD[p] = d; bestJ[p] = j;
      }
      for (let q = 0; q < k; q++) {
        const j = bestJ[q];
        if (j < 0) continue;
        const a = Math.min(i, j), b = Math.max(i, j), key = a * n + b;
        if (seen.has(key)) continue;
        seen.add(key);
        out.push(start + a, start + b);
      }
    }
  }

  function buildGeometry(q) {
    const rand = rng(0x5eed1);
    const ringCounts = RINGS.map((r) => Math.max(12, Math.round(q.ring * r.share * 3)));
    const ringTotal = ringCounts.reduce((a, b) => a + b, 0);
    const total = q.outer + q.inner + ringTotal;
    const P = new Float32Array(total * FLOATS_PER_POINT);
    const tone = new Uint8Array(total); // 0 violet, 1 indigo, 2 cyan, 3 pink (Canvas2D sprite buckets)
    const GA = Math.PI * (3 - Math.sqrt(5));
    let idx = 0;

    function put(x, y, z, t, size, spark) {
      const o = idx * FLOATS_PER_POINT;
      P[o] = x; P[o + 1] = y; P[o + 2] = z;
      if (spark) { P[o + 3] = PINK[0] / 255; P[o + 4] = PINK[1] / 255; P[o + 5] = PINK[2] / 255; tone[idx] = 3; }
      else { paletteColor(t, P, o + 3); tone[idx] = t < 0.3 ? 0 : t < 0.68 ? 1 : 2; }
      P[o + 6] = size; P[o + 7] = rand();
      idx++;
    }
    function fibSphere(n, radius, jitter, tBase, tSpan, size, sparkRate) {
      for (let i = 0; i < n; i++) {
        const y = 1 - ((i + 0.5) / n) * 2, r = Math.sqrt(1 - y * y), th = i * GA;
        const rr = radius * (1 + (rand() - 0.5) * jitter);
        const x = Math.cos(th) * r, z = Math.sin(th) * r;
        const t = tBase + tSpan * (0.5 + 0.38 * y + 0.22 * x) + (rand() - 0.5) * 0.25;
        const spark = rand() < sparkRate;
        put(x * rr, y * rr, z * rr, t, spark ? size * 1.7 : size * (0.75 + rand() * 0.6), spark);
      }
    }

    fibSphere(q.outer, 1, 0.05, 0, 1, 0.034, 0.035);
    fibSphere(q.inner, 0.56, 0.08, 0, 0.5, 0.027, 0.02);
    const ringStart = [];
    RINGS.forEach((ring, ri) => {
      ringStart.push(idx);
      const n = ringCounts[ri];
      for (let j = 0; j < n; j++) {
        const a = (j / n) * Math.PI * 2 + rand() * 0.03, R = ring.R * (1 + (rand() - 0.5) * 0.035);
        const spark = rand() < 0.04;
        put(Math.cos(a) * R, Math.sin(a) * R, (rand() - 0.5) * 0.03, ring.t0 + (rand() - 0.5) * 0.3,
          spark ? 0.042 : 0.018 + rand() * 0.016, spark);
      }
    });

    const pairs = [];
    neighbourPairs(P, 0, q.outer, q.k, 2.7 * Math.sqrt((4 * Math.PI) / q.outer), 1, pairs);
    const outerPairCount = pairs.length / 2;
    neighbourPairs(P, q.outer, q.inner, q.k + 1, 2.9 * 0.56 * Math.sqrt((4 * Math.PI) / q.inner), 0.56, pairs);

    return {
      P, tone, total, pairs: new Uint16Array(pairs),
      outer: [0, q.outer], inner: [q.outer, q.inner],
      rings: RINGS.map((r, i) => [ringStart[i], ringCounts[i]]),
      outerPairs: outerPairCount, innerPairs: pairs.length / 2 - outerPairCount,
    };
  }

  /* ------------------------------------------------------------------ scene state (shared by both renderers) */

  const FOV = (45 * Math.PI) / 180;
  const SCENES = {
    hero: { intensity: 1, speed: 1, lineA: 0.2, dist: 4.05, shiftX: 0.4, narrowDim: 0.6, fit: 2.7 },
    app: { intensity: 0.45, speed: 0.55, lineA: 0.17, dist: 6.0, shiftX: 0.46, narrowDim: 0.8, fit: 3.3 },
  };

  function createSceneState() {
    const s = {
      cfg: SCENES.app, W: 1, H: 1, dpr: 1, wide: true,
      time: 0, spin: 0.6, intro: 0,
      mx: 0, my: 0, tmx: 0, tmy: 0, scroll: 0, tscroll: 0,
      shiftX: 0, shiftY: 0, cam: 6, I: 0, lineA: 0, px: 1,
      proj: M4.create(), view: M4.create(), base: M4.create(), outer: M4.create(), inner: M4.create(),
      rings: RINGS.map(() => M4.create()), ringTilt: RINGS.map(() => M4.create()),
      _t: M4.create(),
    };
    RINGS.forEach((r, i) => {
      const m = s.ringTilt[i];
      M4.rotX(m, r.rx);
      M4.mul(m, m, M4.rotY(s._t, r.ry));
      M4.mul(m, m, M4.rotZ(s._t, r.rz));
    });
    return s;
  }

  function sceneName() {
    const n = doc.body && doc.body.dataset ? doc.body.dataset.scene : "";
    return SCENES[n] ? n : "app";
  }

  // Advances time/easing (dt in seconds; 0 for a static frame) and rebuilds every matrix. No allocations.
  function updateScene(s, dt) {
    const c = s.cfg, aspect = s.W / s.H, k = 1 - Math.exp(-dt * 3.2);
    s.time += dt * c.speed;
    s.spin += dt * 0.07 * c.speed;
    if (s.intro < 1) s.intro = Math.min(1, s.intro + dt / 1.6);
    s.mx += (s.tmx - s.mx) * k;
    s.my += (s.tmy - s.my) * k;
    s.scroll += (s.tscroll - s.scroll) * (dt ? 1 - Math.exp(-dt * 5) : 1);

    const sp = clamp(s.scroll / Math.max(s.H / s.dpr, 1), 0, 2.5);
    let dist = c.dist, shiftX = 0, dim = 1;
    if (s.wide) shiftX = c.shiftX;
    else { dist = Math.max(c.dist, c.fit / Math.max(aspect, 0.42)); dim = c.narrowDim; }
    s.cam = dist + sp * 0.45;
    s.shiftX = shiftX + s.mx * 0.02;
    s.shiftY = -s.my * 0.02 + sp * 0.22;
    s.I = c.intensity * dim * easeOutCubic(s.intro);
    s.lineA = c.lineA;
    s.px = s.H / (2 * Math.tan(FOV / 2));

    M4.perspective(s.proj, FOV, aspect, 0.1, 50);
    M4.translate(s.view, 0, 0, -s.cam);
    const t = s._t, b = s.base;
    M4.rotZ(b, 0.2);
    M4.mul(b, b, M4.rotX(t, 0.32 + s.my * 0.2 + sp * 0.18));
    M4.mul(b, b, M4.rotY(t, s.spin + s.mx * 0.35 + sp * 0.5));
    s.outer.set(b);
    M4.mul(s.inner, b, M4.rotY(t, -s.spin * 1.8));
    M4.mul(s.inner, s.inner, M4.rotX(t, s.spin * 0.9));
    for (let i = 0; i < RINGS.length; i++) {
      M4.mul(s.rings[i], b, s.ringTilt[i]);
      M4.mul(s.rings[i], s.rings[i], M4.rotZ(t, s.time * RINGS[i].speed * 2.5));
    }
  }

  /* ------------------------------------------------------------------ WebGL renderer */

  const SHADER_COMMON = [
    "precision highp float;",
    "uniform mat4 uProj; uniform mat4 uView; uniform mat4 uModel;",
    "uniform float uTime; uniform vec2 uShift; uniform float uI; uniform float uCam;",
    "float wave(vec3 p) {",
    "  float a = sin(dot(p, vec3(0.83, 0.47, 0.29)) * 3.4 - uTime * 1.15);",
    "  float b = sin(dot(p, vec3(-0.31, 0.88, -0.36)) * 2.6 - uTime * 0.7 + 1.7);",
    "  return pow(max(a, 0.0), 10.0) + 0.6 * pow(max(b, 0.0), 14.0);",
    "}",
    "vec4 place(vec3 p, out float depth) {",
    "  vec4 vp = uView * uModel * vec4(p, 1.0);",
    "  vec4 cp = uProj * vp;",
    "  cp.xy += uShift * cp.w;",
    "  depth = -vp.z;",
    "  return cp;",
    "}",
    "float fade(float z) { return clamp(1.0 - (z - uCam + 1.0) * 0.38, 0.18, 1.0); }",
  ].join("\n");

  const POINT_VS = SHADER_COMMON + "\n" + [
    "attribute vec3 aPos; attribute vec3 aCol; attribute float aSize; attribute float aSeed;",
    "uniform float uPx; uniform float uMaxPt;",
    "varying vec3 vCol; varying float vA;",
    "void main() {",
    "  float z;",
    "  gl_Position = place(aPos, z);",
    "  float w = wave(aPos);",
    "  float tw = 0.62 + 0.38 * sin(uTime * (1.3 + aSeed * 2.2) + aSeed * 40.0);",
    "  gl_PointSize = clamp(aSize * uPx / z * (1.0 + 0.55 * w), 1.0, uMaxPt);",
    "  vCol = min(aCol * (1.0 + 0.9 * w), vec3(1.0));",
    "  vA = uI * fade(z) * (tw + 0.9 * w);",
    "}",
  ].join("\n");

  const POINT_FS = [
    "precision mediump float;",
    "varying vec3 vCol; varying float vA;",
    "void main() {",
    "  vec2 c = gl_PointCoord * 2.0 - 1.0;",
    "  float d = dot(c, c);",
    "  if (d > 1.0) discard;",
    "  float glow = exp(-d * 5.0) * (1.0 - d);",
    "  float core = 1.0 - smoothstep(0.0, 0.16, d);",
    "  float a = clamp((glow * 0.55 + core) * vA, 0.0, 1.0);",
    "  vec3 col = mix(vCol, vec3(1.0), core * 0.45);",
    "  gl_FragColor = vec4(col * a, a);",
    "}",
  ].join("\n");

  const LINE_VS = SHADER_COMMON + "\n" + [
    "attribute vec3 aPos; attribute vec3 aCol;",
    "uniform float uLineA;",
    "varying vec3 vCol; varying float vA;",
    "void main() {",
    "  float z;",
    "  gl_Position = place(aPos, z);",
    "  float w = wave(aPos);",
    "  float f = fade(z);",
    "  vCol = min(aCol * (1.0 + 1.2 * w), vec3(1.0));",
    "  vA = uI * uLineA * f * f * (0.55 + 1.6 * w);",
    "}",
  ].join("\n");

  const LINE_FS = [
    "precision mediump float;",
    "varying vec3 vCol; varying float vA;",
    "void main() { float a = clamp(vA, 0.0, 1.0); gl_FragColor = vec4(vCol * a, a); }",
  ].join("\n");

  const UNIFORMS = ["uProj", "uView", "uModel", "uTime", "uShift", "uI", "uCam", "uPx", "uMaxPt", "uLineA"];

  function createGLRenderer(canvas, makeGeometry) {
    const attrs = { alpha: true, antialias: true, depth: false, stencil: false, premultipliedAlpha: true,
      preserveDrawingBuffer: false, failIfMajorPerformanceCaveat: true, powerPreference: "default" };
    const gl = canvas.getContext("webgl", attrs) || canvas.getContext("experimental-webgl", attrs);
    if (!gl) return null;
    const geo = makeGeometry();

    let pointProg = null, lineProg = null, pointBuf = null, lineBuf = null, maxPt = 64;
    const lineVerts = geo.pairs.length;

    function shader(type, src) {
      const sh = gl.createShader(type);
      gl.shaderSource(sh, src);
      gl.compileShader(sh);
      if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS) && !gl.isContextLost()) throw new Error("fx: shader");
      return sh;
    }
    function program(vs, fs, names) {
      const p = gl.createProgram();
      gl.attachShader(p, shader(gl.VERTEX_SHADER, vs));
      gl.attachShader(p, shader(gl.FRAGMENT_SHADER, fs));
      names.forEach((n, i) => gl.bindAttribLocation(p, i, n));
      gl.linkProgram(p);
      if (!gl.getProgramParameter(p, gl.LINK_STATUS) && !gl.isContextLost()) throw new Error("fx: link");
      const u = {};
      UNIFORMS.forEach((n) => { u[n] = gl.getUniformLocation(p, n); });
      return { p, u };
    }

    function init() {
      pointProg = program(POINT_VS, POINT_FS, ["aPos", "aCol", "aSize", "aSeed"]);
      lineProg = program(LINE_VS, LINE_FS, ["aPos", "aCol"]);

      pointBuf = gl.createBuffer();
      gl.bindBuffer(gl.ARRAY_BUFFER, pointBuf);
      gl.bufferData(gl.ARRAY_BUFFER, geo.P, gl.STATIC_DRAW);

      // Expand index pairs into a position+colour line buffer (6 floats per vertex).
      const L = new Float32Array(lineVerts * 6);
      for (let i = 0; i < lineVerts; i++) {
        const o = geo.pairs[i] * FLOATS_PER_POINT;
        for (let c = 0; c < 6; c++) L[i * 6 + c] = geo.P[o + c];
      }
      lineBuf = gl.createBuffer();
      gl.bindBuffer(gl.ARRAY_BUFFER, lineBuf);
      gl.bufferData(gl.ARRAY_BUFFER, L, gl.STATIC_DRAW);

      const range = gl.getParameter(gl.ALIASED_POINT_SIZE_RANGE);
      maxPt = range && range[1] ? range[1] : 64;
      gl.disable(gl.DEPTH_TEST);
      gl.enable(gl.BLEND);
      gl.blendFunc(gl.ONE, gl.ONE); // additive, premultiplied output keeps rgb <= alpha
    }

    function common(prog, s) {
      const u = prog.u;
      gl.uniformMatrix4fv(u.uProj, false, s.proj);
      gl.uniformMatrix4fv(u.uView, false, s.view);
      gl.uniform1f(u.uTime, s.time);
      gl.uniform2f(u.uShift, s.shiftX, s.shiftY);
      gl.uniform1f(u.uI, s.I);
      gl.uniform1f(u.uCam, s.cam);
    }

    function render(s) {
      if (gl.isContextLost()) return;
      gl.viewport(0, 0, s.W, s.H);
      gl.clearColor(0, 0, 0, 0);
      gl.clear(gl.COLOR_BUFFER_BIT);
      if (s.I <= 0.002) return;

      // Neighbour lines
      gl.useProgram(lineProg.p);
      common(lineProg, s);
      gl.uniform1f(lineProg.u.uLineA, s.lineA);
      gl.bindBuffer(gl.ARRAY_BUFFER, lineBuf);
      gl.enableVertexAttribArray(0); gl.enableVertexAttribArray(1);
      gl.disableVertexAttribArray(2); gl.disableVertexAttribArray(3);
      gl.vertexAttribPointer(0, 3, gl.FLOAT, false, 24, 0);
      gl.vertexAttribPointer(1, 3, gl.FLOAT, false, 24, 12);
      gl.uniformMatrix4fv(lineProg.u.uModel, false, s.outer);
      gl.drawArrays(gl.LINES, 0, geo.outerPairs * 2);
      gl.uniformMatrix4fv(lineProg.u.uModel, false, s.inner);
      gl.drawArrays(gl.LINES, geo.outerPairs * 2, geo.innerPairs * 2);

      // Glowing point sprites
      gl.useProgram(pointProg.p);
      common(pointProg, s);
      gl.uniform1f(pointProg.u.uPx, s.px);
      gl.uniform1f(pointProg.u.uMaxPt, Math.min(maxPt, 48 * s.dpr));
      gl.bindBuffer(gl.ARRAY_BUFFER, pointBuf);
      for (let i = 0; i < 4; i++) gl.enableVertexAttribArray(i);
      gl.vertexAttribPointer(0, 3, gl.FLOAT, false, 32, 0);
      gl.vertexAttribPointer(1, 3, gl.FLOAT, false, 32, 12);
      gl.vertexAttribPointer(2, 1, gl.FLOAT, false, 32, 24);
      gl.vertexAttribPointer(3, 1, gl.FLOAT, false, 32, 28);
      const u = pointProg.u;
      gl.uniformMatrix4fv(u.uModel, false, s.outer);
      gl.drawArrays(gl.POINTS, geo.outer[0], geo.outer[1]);
      gl.uniformMatrix4fv(u.uModel, false, s.inner);
      gl.uniform1f(u.uI, s.I * 0.85);
      gl.drawArrays(gl.POINTS, geo.inner[0], geo.inner[1]);
      gl.uniform1f(u.uI, s.I * 0.9);
      for (let i = 0; i < geo.rings.length; i++) {
        gl.uniformMatrix4fv(u.uModel, false, s.rings[i]);
        gl.drawArrays(gl.POINTS, geo.rings[i][0], geo.rings[i][1]);
      }
    }

    init();
    return { kind: "webgl", render, restore: init, maxDpr: 1.75, isLost: () => gl.isContextLost() };
  }

  /* ------------------------------------------------------------------ Canvas2D fallback renderer */

  function createCanvas2DRenderer(canvas, geo) {
    const ctx = canvas.getContext("2d");
    if (!ctx) return null;
    const n = geo.total;
    const SX = new Float32Array(n), SY = new Float32Array(n), SA = new Float32Array(n), SS = new Float32Array(n);
    const mvp = M4.create();
    const sprites = [PALETTE[0], PALETTE[1], PALETTE[2], PINK].map((c) => {
      const cv = doc.createElement("canvas");
      cv.width = cv.height = 64;
      const g = cv.getContext("2d"), grad = g.createRadialGradient(32, 32, 0, 32, 32, 32);
      grad.addColorStop(0, "rgba(255,255,255,1)");
      grad.addColorStop(0.12, "rgba(" + c[0] + "," + c[1] + "," + c[2] + ",0.95)");
      grad.addColorStop(0.4, "rgba(" + c[0] + "," + c[1] + "," + c[2] + ",0.28)");
      grad.addColorStop(1, "rgba(" + c[0] + "," + c[1] + "," + c[2] + ",0)");
      g.fillStyle = grad;
      g.fillRect(0, 0, 64, 64);
      return cv;
    });

    function wave(x, y, z, t) {
      const a = Math.sin((x * 0.83 + y * 0.47 + z * 0.29) * 3.4 - t * 1.15);
      const b = Math.sin((x * -0.31 + y * 0.88 - z * 0.36) * 2.6 - t * 0.7 + 1.7);
      return Math.pow(Math.max(a, 0), 10) + 0.6 * Math.pow(Math.max(b, 0), 14);
    }

    function project(s, model, start, count, gain) {
      M4.mul(mvp, s.view, model);
      M4.mul(mvp, s.proj, mvp);
      const P = geo.P, m = mvp;
      for (let i = start; i < start + count; i++) {
        const o = i * FLOATS_PER_POINT, x = P[o], y = P[o + 1], z = P[o + 2];
        const w = m[3] * x + m[7] * y + m[11] * z + m[15];
        const cx = (m[0] * x + m[4] * y + m[8] * z + m[12]) / w + s.shiftX;
        const cy = (m[1] * x + m[5] * y + m[9] * z + m[13]) / w + s.shiftY;
        SX[i] = (cx * 0.5 + 0.5) * s.W;
        SY[i] = (0.5 - cy * 0.5) * s.H;
        const f = clamp(1 - (w - s.cam + 1) * 0.38, 0.18, 1), wv = wave(x, y, z, s.time);
        const tw = 0.62 + 0.38 * Math.sin(s.time * (1.3 + P[o + 7] * 2.2) + P[o + 7] * 40);
        SA[i] = clamp(s.I * gain * f * (tw + 0.9 * wv), 0, 1);
        SS[i] = P[o + 6] * s.px / w * (1 + 0.55 * wv) * 2.2;
      }
    }

    function strokePairs(from, to, alpha) {
      ctx.strokeStyle = "rgba(129,140,248," + alpha.toFixed(3) + ")";
      ctx.beginPath();
      for (let p = from; p < to; p++) {
        const a = geo.pairs[p * 2], b = geo.pairs[p * 2 + 1];
        ctx.moveTo(SX[a], SY[a]);
        ctx.lineTo(SX[b], SY[b]);
      }
      ctx.stroke();
    }

    function render(s) {
      ctx.setTransform(1, 0, 0, 1, 0, 0);
      ctx.globalCompositeOperation = "source-over";
      ctx.globalAlpha = 1;
      ctx.clearRect(0, 0, s.W, s.H);
      if (s.I <= 0.002) return;
      project(s, s.outer, geo.outer[0], geo.outer[1], 1);
      project(s, s.inner, geo.inner[0], geo.inner[1], 0.85);
      for (let i = 0; i < geo.rings.length; i++) project(s, s.rings[i], geo.rings[i][0], geo.rings[i][1], 0.9);

      ctx.globalCompositeOperation = "lighter";
      ctx.lineWidth = Math.max(1, s.dpr * 0.8);
      strokePairs(0, geo.outerPairs, s.lineA * s.I * 0.55);
      strokePairs(geo.outerPairs, geo.outerPairs + geo.innerPairs, s.lineA * s.I * 0.45);
      for (let i = 0; i < n; i++) {
        const a = SA[i], sz = SS[i];
        if (a < 0.01 || sz < 0.5) continue;
        ctx.globalAlpha = a;
        ctx.drawImage(sprites[geo.tone[i]], SX[i] - sz / 2, SY[i] - sz / 2, sz, sz);
      }
      ctx.globalAlpha = 1;
    }

    return { kind: "2d", render, restore() {}, maxDpr: 1.25, isLost: () => false };
  }

  /* ------------------------------------------------------------------ background controller */

  function initBackground() {
    let canvas = doc.getElementById("fx-bg");
    if (!canvas || !canvas.getContext) return;
    if (isStatic(canvas)) { // SHELL normally styles it; make sure it never takes layout space
      css(canvas, "position", "fixed"); css(canvas, "inset", "0"); css(canvas, "width", "100%");
      css(canvas, "height", "100%"); css(canvas, "pointer-events", "none"); css(canvas, "z-index", "0");
    }

    const s = createSceneState();
    s.cfg = SCENES[sceneName()];
    let renderer = null;
    try { renderer = createGLRenderer(canvas, () => buildGeometry(quality(false))); } catch (e) { renderer = null; }
    if (!renderer) {
      // A canvas that already holds a (broken) WebGL context cannot give a 2D one: swap in a fresh clone.
      const fresh = canvas.cloneNode(false);
      if (canvas.parentNode) canvas.parentNode.replaceChild(fresh, canvas);
      canvas = fresh;
      try { renderer = createCanvas2DRenderer(canvas, buildGeometry(quality(true))); } catch (e) { renderer = null; }
    }
    if (!renderer) return;

    if (isRefresh() || reduced()) s.intro = 1; // no fade-in replay on meta-refresh pages
    let rafId = 0, last = 0, lost = false;

    function resize() {
      const dpr = Math.min(win.devicePixelRatio || 1, renderer.maxDpr);
      const cw = win.innerWidth || canvas.clientWidth || 1, ch = win.innerHeight || canvas.clientHeight || 1;
      const W = Math.max(1, Math.round(cw * dpr)), H = Math.max(1, Math.round(ch * dpr));
      if (canvas.width !== W) canvas.width = W;
      if (canvas.height !== H) canvas.height = H;
      s.W = W; s.H = H; s.dpr = dpr; s.wide = cw >= 960;
    }

    const draw = safely((dt) => { if (!lost) { updateScene(s, dt); renderer.render(s); } });

    function frame(t) {
      rafId = 0;
      const dt = Math.min(Math.max((t - last) / 1000, 0), 0.05);
      last = t;
      draw(dt);
      if (animating()) rafId = raf(frame);
    }
    const animating = () => !reduced() && !doc.hidden && !lost;
    function start() {
      if (rafId || !animating()) return;
      last = now();
      rafId = raf(frame);
    }
    function stop() { if (rafId) caf(rafId); rafId = 0; }
    function staticFrame() { s.tmx = s.mx = s.tmy = s.my = s.tscroll = s.scroll = 0; s.intro = 1; draw(0); }
    function refresh() { stop(); if (reduced()) staticFrame(); else start(); }

    let resizeQueued = false;
    win.addEventListener("resize", () => {
      if (resizeQueued) return;
      resizeQueued = true;
      raf(() => { resizeQueued = false; resize(); if (!rafId) draw(0); });
    }, PASSIVE);
    doc.addEventListener("visibilitychange", () => { if (doc.hidden) stop(); else start(); });
    onMediaChange(reduceMQ, refresh);

    win.addEventListener("pointermove", (e) => {
      if (e.pointerType === "touch" || reduced()) return;
      s.tmx = clamp((e.clientX / (win.innerWidth || 1)) * 2 - 1, -1, 1);
      s.tmy = clamp((e.clientY / (win.innerHeight || 1)) * 2 - 1, -1, 1);
    }, PASSIVE);
    win.addEventListener("scroll", () => {
      if (!reduced()) s.tscroll = win.pageYOffset || root.scrollTop || 0;
    }, PASSIVE);

    canvas.addEventListener("webglcontextlost", (e) => { e.preventDefault(); lost = true; stop(); }, false);
    canvas.addEventListener("webglcontextrestored", safely(() => {
      renderer.restore();
      lost = false;
      resize();
      refresh();
    }), false);

    s.tscroll = s.scroll = reduced() ? 0 : win.pageYOffset || 0;
    resize();
    refresh();
  }

  /* ------------------------------------------------------------------ interaction motion loop */

  // A single rAF loop that runs only while some item still animates. step(dt, t) returns true to keep going.
  const Motion = (function () {
    const items = new Set();
    let id = 0, last = 0;
    function tick(t) {
      id = 0;
      const dt = Math.min(Math.max((t - last) / 1000, 0), 0.05);
      last = t;
      items.forEach((it) => {
        let keep = false;
        try { keep = it.step(dt, t); } catch (e) { keep = false; }
        if (!keep) items.delete(it);
      });
      if (items.size) id = raf(tick);
    }
    return {
      add(it) {
        items.add(it);
        if (!id) { last = now(); id = raf(tick); }
      },
    };
  })();

  // Cached client rects: read once per pointerenter, invalidated by scroll/resize.
  let rectEpoch = 0;
  const rects = new WeakMap();
  function rectOf(el, fresh) {
    let r = rects.get(el);
    if (fresh || !r || r.epoch !== rectEpoch) {
      r = { box: el.getBoundingClientRect(), epoch: rectEpoch };
      rects.set(el, r);
    }
    return r.box;
  }
  win.addEventListener("scroll", () => { rectEpoch++; }, PASSIVE);
  win.addEventListener("resize", () => { rectEpoch++; }, PASSIVE);

  const ease = (dt, rate) => 1 - Math.exp(-dt * rate);
  const settled = (a, b, eps) => Math.abs(a - b) < eps;

  /* ------------------------------------------------------------------ reveals */

  function initReveals() {
    const els = Array.prototype.slice.call(doc.querySelectorAll(".reveal"));
    if (!els.length) return;
    const showAll = () => els.forEach((el) => { css(el, "--i", "0"); el.classList.add("in"); });
    if (isRefresh() || reduced() || !("IntersectionObserver" in win)) { showAll(); return; }
    const order = new Map();
    els.forEach((el, i) => { order.set(el, i); css(el, "--i", String(Math.min(i, 8))); });
    const io = new IntersectionObserver((entries) => {
      // Stagger within each batch that enters together, in DOM order.
      const entering = entries.filter((e) => e.isIntersecting).map((e) => e.target);
      entering.sort((a, b) => order.get(a) - order.get(b));
      entering.forEach((el, k) => {
        css(el, "--i", String(Math.min(k, 8)));
        el.classList.add("in");
        io.unobserve(el);
      });
    }, { rootMargin: "0px 0px -6% 0px", threshold: 0 });
    els.forEach((el) => io.observe(el));
    onMediaChange(reduceMQ, () => { if (reduced()) { io.disconnect(); showAll(); } });
  }

  /* ------------------------------------------------------------------ count-ups and gauges */

  // Calls fn(el) once the element is visible (immediately without IntersectionObserver).
  function whenVisible(els, fn) {
    if (!els.length) return;
    if (!("IntersectionObserver" in win)) { els.forEach(fn); return; }
    const io = new IntersectionObserver((entries) => {
      entries.forEach((e) => {
        if (!e.isIntersecting) return;
        io.unobserve(e.target);
        fn(e.target);
      });
    }, { threshold: 0.3 });
    els.forEach((el) => io.observe(el));
  }

  function initCountUps() {
    if (isRefresh() || reduced()) return;
    const NUM = /-?\d[\d,]*(?:\.\d+)?/;
    const ready = [], data = new WeakMap();
    Array.prototype.forEach.call(doc.querySelectorAll("[data-count]"), (el) => {
      if (el.children.length) return; // only plain-text nodes; never clobber markup
      const orig = el.textContent, to = parseFloat(el.getAttribute("data-count"));
      const m = orig.match(NUM);
      if (!m || !isFinite(to)) return;
      const from = parseFloat(el.getAttribute("data-count-from") || "0") || 0;
      const num = m[0], pre = orig.slice(0, m.index), post = orig.slice(m.index + num.length);
      const dot = num.indexOf("."), dec = dot < 0 ? 0 : num.length - dot - 1, group = num.indexOf(",") >= 0;
      const fmt = (v) => {
        let str = v.toFixed(dec);
        if (group) {
          const parts = str.split(".");
          parts[0] = parts[0].replace(/\B(?=(\d{3})+(?!\d))/g, ",");
          str = parts.join(".");
        }
        return pre + str + post;
      };
      el.textContent = fmt(from);
      ready.push(el);
      data.set(el, { orig, from, to, fmt });
    });
    whenVisible(ready, (el) => {
      const c = data.get(el);
      let t0 = 0;
      Motion.add({
        step(dt, t) {
          if (!t0) t0 = t;
          const p = clamp((t - t0) / 1100, 0, 1);
          if (p >= 1) { el.textContent = c.orig; return false; }
          el.textContent = c.fmt(c.from + (c.to - c.from) * easeOutCubic(p));
          return true;
        },
      });
    });
  }

  function gaugeHue(v) { // 0 red -> 45 amber (at 50) -> 145 green (at 100)
    v = clamp(v, 0, 100);
    return v < 50 ? v * 0.9 : 45 + (v - 50) * 2;
  }

  function initGauges() {
    const els = Array.prototype.filter.call(doc.querySelectorAll(".gauge[data-value]"),
      (el) => isFinite(parseFloat(el.getAttribute("data-value"))));
    if (!els.length) return;
    const setGauge = (el, v) => { css(el, "--p", (clamp(v, 0, 100) / 100).toFixed(4)); css(el, "--gauge-hue", gaugeHue(v).toFixed(1)); };
    const value = (el) => parseFloat(el.getAttribute("data-value"));
    const before = (el) => { const b = parseFloat(el.getAttribute("data-before")); return isFinite(b) ? b : 0; };
    if (isRefresh() || reduced()) { els.forEach((el) => setGauge(el, value(el))); return; }
    els.forEach((el) => setGauge(el, before(el)));
    whenVisible(els, (el) => {
      const from = before(el), to = value(el);
      let t0 = 0;
      Motion.add({
        step(dt, t) {
          if (!t0) t0 = t;
          const p = clamp((t - t0) / 1400, 0, 1);
          setGauge(el, from + (to - from) * easeInOutCubic(p));
          return p < 1;
        },
      });
    });
  }

  /* ------------------------------------------------------------------ tilt, spotlight, magnetic */

  const MAX_TILT = 8;
  const pointerOK = (e) => e.pointerType !== "touch" && finePointer() && !reduced();

  function initTilt() {
    const els = doc.querySelectorAll(".tilt");
    if (!els.length) return;
    const glares = [];
    Array.prototype.forEach.call(els, (el) => {
      const glare = doc.createElement("span");
      glare.className = "glare";
      glare.setAttribute("aria-hidden", "true");
      el.appendChild(glare);
      glares.push(glare);

      const st = { x: 0, y: 0, z: 0, g: 0, tx: 0, ty: 0, tz: 0, tg: 0, hover: false, active: false };
      st.step = (dt) => {
        const k = ease(dt, st.hover ? 14 : 9);
        st.x += (st.tx - st.x) * k; st.y += (st.ty - st.y) * k;
        st.z += (st.tz - st.z) * k; st.g += (st.tg - st.g) * k;
        const done = settled(st.x, st.tx, 0.02) && settled(st.y, st.ty, 0.02) && settled(st.z, st.tz, 0.05) && settled(st.g, st.tg, 0.01);
        if (done && !st.hover) {
          st.x = st.y = st.z = st.g = 0;
          el.style.removeProperty("transform");
          css(el, "--glare-o", "0");
          st.active = false;
          return false;
        }
        css(el, "transform", "perspective(900px) rotateX(" + st.x.toFixed(3) + "deg) rotateY(" + st.y.toFixed(3) +
          "deg) translateZ(" + st.z.toFixed(2) + "px)");
        css(el, "--glare-o", st.g.toFixed(3));
        if (done) { st.active = false; return false; }
        return true;
      };
      const kick = () => { if (!st.active) { st.active = true; Motion.add(st); } };

      el.addEventListener("pointerenter", (e) => {
        if (!pointerOK(e)) return;
        rectOf(el, true);
        st.hover = true; st.tz = 6; st.tg = 1;
        kick();
      }, PASSIVE);
      el.addEventListener("pointermove", (e) => {
        if (!st.hover) return;
        const r = rectOf(el);
        const px = clamp((e.clientX - r.left) / (r.width || 1), 0, 1), py = clamp((e.clientY - r.top) / (r.height || 1), 0, 1);
        st.ty = (px - 0.5) * 2 * MAX_TILT;
        st.tx = (0.5 - py) * 2 * MAX_TILT;
        css(el, "--gx", (px * 100).toFixed(1) + "%");
        css(el, "--gy", (py * 100).toFixed(1) + "%");
        kick();
      }, PASSIVE);
      el.addEventListener("pointerleave", () => {
        if (!st.hover) return;
        st.hover = false; st.tx = st.ty = st.tz = st.tg = 0;
        kick();
      }, PASSIVE);
    });

    // If SHELL does not style .glare, keep it out of layout and give it a functional default look.
    if (glares.length && isStatic(glares[0])) {
      glares.forEach((g) => {
        css(g, "position", "absolute"); css(g, "inset", "0"); css(g, "pointer-events", "none");
        css(g, "border-radius", "inherit"); css(g, "opacity", "var(--glare-o, 0)");
        css(g, "background", "radial-gradient(circle at var(--gx, 50%) var(--gy, 50%), rgba(255,255,255,.14), transparent 55%)");
        if (g.parentNode && isStatic(g.parentNode)) css(g.parentNode, "position", "relative");
      });
    }
  }

  function initSpotlight() {
    Array.prototype.forEach.call(doc.querySelectorAll(".card"), (el) => {
      const st = { cx: 0, cy: 0, queued: false };
      st.step = () => {
        st.queued = false;
        const r = rectOf(el);
        css(el, "--mx", (st.cx - r.left).toFixed(1) + "px");
        css(el, "--my", (st.cy - r.top).toFixed(1) + "px");
        return false;
      };
      el.addEventListener("pointerenter", (e) => { if (e.pointerType !== "touch") rectOf(el, true); }, PASSIVE);
      el.addEventListener("pointermove", (e) => {
        if (e.pointerType === "touch") return;
        st.cx = e.clientX; st.cy = e.clientY;
        if (!st.queued) { st.queued = true; Motion.add(st); }
      }, PASSIVE);
    });
  }

  function initMagnetic() {
    const useTranslate = "translate" in root.style; // composes with CSS hover transforms
    const MAX = 6;
    Array.prototype.forEach.call(doc.querySelectorAll(".magnetic"), (el) => {
      const st = { x: 0, y: 0, tx: 0, ty: 0, hover: false, active: false };
      const apply = (x, y) => {
        if (useTranslate) css(el, "translate", x.toFixed(2) + "px " + y.toFixed(2) + "px");
        else css(el, "transform", "translate3d(" + x.toFixed(2) + "px," + y.toFixed(2) + "px,0)");
      };
      st.step = (dt) => {
        const k = ease(dt, 12);
        st.x += (st.tx - st.x) * k; st.y += (st.ty - st.y) * k;
        if (settled(st.x, st.tx, 0.03) && settled(st.y, st.ty, 0.03)) {
          st.active = false;
          if (!st.hover) { el.style.removeProperty(useTranslate ? "translate" : "transform"); return false; }
          apply(st.tx, st.ty);
          return false;
        }
        apply(st.x, st.y);
        return true;
      };
      const kick = () => { if (!st.active) { st.active = true; Motion.add(st); } };
      el.addEventListener("pointerenter", (e) => {
        if (!pointerOK(e)) return;
        rectOf(el, true);
        st.hover = true;
      }, PASSIVE);
      el.addEventListener("pointermove", (e) => {
        if (!st.hover) return;
        const r = rectOf(el);
        st.tx = clamp((e.clientX - (r.left + r.width / 2)) / (r.width / 2 || 1), -1, 1) * MAX;
        st.ty = clamp((e.clientY - (r.top + r.height / 2)) / (r.height / 2 || 1), -1, 1) * MAX;
        kick();
      }, PASSIVE);
      el.addEventListener("pointerleave", () => {
        st.hover = false; st.tx = st.ty = 0;
        kick();
      }, PASSIVE);
    });
  }

  /* ------------------------------------------------------------------ ripple */

  let rippleStyled = null; // null = unknown, then true/false after the first ripple
  function initRipple() {
    doc.addEventListener("click", safely((e) => {
      if (reduced() || !e.target || !e.target.closest) return;
      const btn = e.target.closest(".btn");
      if (!btn || btn.disabled) return;
      const r = btn.getBoundingClientRect();
      const keyboard = e.detail === 0 || (!e.clientX && !e.clientY);
      const x = keyboard ? r.width / 2 : e.clientX - r.left, y = keyboard ? r.height / 2 : e.clientY - r.top;
      const span = doc.createElement("span");
      span.className = "ripple";
      span.setAttribute("aria-hidden", "true");
      css(span, "--rx", x.toFixed(1) + "px");
      css(span, "--ry", y.toFixed(1) + "px");
      btn.appendChild(span);
      if (rippleStyled === null) rippleStyled = !isStatic(span);
      let done = false;
      const remove = () => { if (done) return; done = true; if (span.parentNode) span.parentNode.removeChild(span); };
      if (!rippleStyled) { // fallback look so an unstyled ripple is still harmless (and pretty)
        const size = Math.max(r.width, r.height) * 2;
        css(span, "position", "absolute"); css(span, "pointer-events", "none"); css(span, "border-radius", "50%");
        css(span, "left", (x - size / 2) + "px"); css(span, "top", (y - size / 2) + "px");
        css(span, "width", size + "px"); css(span, "height", size + "px");
        css(span, "background", "radial-gradient(circle, rgba(255,255,255,.35), rgba(255,255,255,0) 60%)");
        if (isStatic(btn)) css(btn, "position", "relative");
        css(btn, "overflow", "hidden");
        if (span.animate) {
          const a = span.animate([{ transform: "scale(0)", opacity: 1 }, { transform: "scale(1)", opacity: 0 }],
            { duration: 600, easing: "cubic-bezier(.2,.7,.3,1)" });
          a.onfinish = remove;
        }
      }
      span.addEventListener("animationend", remove);
      win.setTimeout(remove, 700);
    }), PASSIVE);
  }

  /* ------------------------------------------------------------------ page-exit transition */

  function shouldTransition(e, a) {
    if (e.defaultPrevented || e.button !== 0 || e.metaKey || e.ctrlKey || e.shiftKey || e.altKey) return false;
    if (typeof a.href !== "string") return false; // SVG <a>
    if (reduced() || a.hasAttribute("download") || a.hasAttribute("data-no-transition")) return false;
    const target = (a.getAttribute("target") || "").toLowerCase();
    if (target && target !== "_self") return false;
    let url;
    try { url = new URL(a.href, location.href); } catch (err) { return false; }
    if (url.origin !== location.origin || (url.protocol !== "http:" && url.protocol !== "https:")) return false;
    if (url.pathname === location.pathname && url.search === location.search) return false; // hash-only / same page
    const p = url.pathname;
    if (/^\/runs\/[^/]+\/report\./.test(p)) return false;
    if (/^\/(docs|redoc|openapi\.json|api\/|static\/)/.test(p)) return false;
    return true;
  }

  function initPageExit() {
    let fallback = 0;
    doc.addEventListener("click", (e) => {
      try {
        const a = e.target && e.target.closest ? e.target.closest("a[href]") : null;
        if (!a || !shouldTransition(e, a)) return;
        e.preventDefault();
        const href = a.href;
        root.classList.add("leaving");
        win.setTimeout(() => { location.href = href; }, 180);
        win.clearTimeout(fallback);
        fallback = win.setTimeout(() => root.classList.remove("leaving"), 4000); // e.g. 204 / download responses
      } catch (err) { /* let the browser navigate normally */ }
    });
    win.addEventListener("pageshow", () => { root.classList.remove("leaving"); });
  }

  /* ------------------------------------------------------------------ cursor glow */

  function initCursorGlow() {
    if (!finePointer() || reduced() || !doc.body) return;
    const el = doc.createElement("div");
    el.className = "cursor-glow";
    el.setAttribute("aria-hidden", "true");
    const SIZE = 440;
    [["position", "fixed"], ["left", "0"], ["top", "0"], ["width", SIZE + "px"], ["height", SIZE + "px"],
      ["margin", (-SIZE / 2) + "px 0 0 " + (-SIZE / 2) + "px"], ["border-radius", "50%"], ["pointer-events", "none"],
      ["z-index", "0"], ["mix-blend-mode", "screen"], ["opacity", "0"], ["transition", "opacity .4s ease"],
      ["will-change", "transform"],
      ["background", "radial-gradient(circle, rgba(139,92,246,.32) 0%, rgba(34,211,238,.12) 38%, rgba(5,7,13,0) 68%)"],
    ].forEach((p) => css(el, p[0], p[1]));
    const bg = doc.getElementById("fx-bg");
    if (bg && bg.parentNode === doc.body) doc.body.insertBefore(el, bg.nextSibling);
    else doc.body.insertBefore(el, doc.body.firstChild);

    const st = { x: -9999, y: -9999, tx: -9999, ty: -9999, active: false, shown: false };
    st.step = (dt) => {
      const k = ease(dt, 10);
      st.x += (st.tx - st.x) * k; st.y += (st.ty - st.y) * k;
      css(el, "transform", "translate3d(" + st.x.toFixed(1) + "px," + st.y.toFixed(1) + "px,0)");
      const keep = !(settled(st.x, st.tx, 0.3) && settled(st.y, st.ty, 0.3));
      st.active = keep;
      return keep;
    };
    win.addEventListener("pointermove", (e) => {
      if (e.pointerType === "touch" || reduced()) return;
      st.tx = e.clientX; st.ty = e.clientY;
      if (!st.shown) { st.shown = true; st.x = st.tx; st.y = st.ty; css(el, "opacity", ".35"); }
      if (!st.active) { st.active = true; Motion.add(st); }
    }, PASSIVE);
    doc.addEventListener("pointerout", (e) => {
      if (!e.relatedTarget && st.shown) { st.shown = false; css(el, "opacity", "0"); }
    }, PASSIVE);
    onMediaChange(reduceMQ, () => { if (reduced()) { st.shown = false; css(el, "opacity", "0"); } });
  }

  /* ------------------------------------------------------------------ boot */

  function boot() {
    [initReveals, initCountUps, initGauges, initTilt, initSpotlight, initMagnetic, initRipple, initPageExit,
      initCursorGlow, initBackground].forEach((fn) => { safely(fn)(); });
  }

  if (doc.readyState === "loading") doc.addEventListener("DOMContentLoaded", boot);
  else boot();
})();

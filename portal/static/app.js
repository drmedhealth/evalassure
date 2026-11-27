// Progressive enhancement only: every form works without JavaScript.
(function () {
  "use strict";

  // Confirm destructive actions.
  document.querySelectorAll("form[data-confirm]").forEach(function (f) {
    f.addEventListener("submit", function (e) {
      if (!window.confirm(f.getAttribute("data-confirm"))) e.preventDefault();
    });
  });

  // ML form: fill target / sensitive / drop choices from the chosen dataset's columns.
  var ds = document.getElementById("ml-dataset");
  if (ds) {
    var target = document.getElementById("ml-target");
    var sens = document.getElementById("ml-sensitive");
    var drop = document.getElementById("ml-drop");
    var task = document.getElementById("ml-task");
    var builtin = document.querySelector("select[name=builtin]");
    var guess = /^(target|label|y|class|outcome|approved|churn|fraud|default|price)$/i;
    var sensGuess = /^(gender|sex|race|ethnicity|age_group|age_band|religion|nationality|disability|marital_status)$/i;

    function cols() {
      var el = document.getElementById("cols-" + ds.value);
      return el ? JSON.parse(el.textContent) : [];
    }
    function checkbox(name, value, checked) {
      var l = document.createElement("label"); l.className = "chk";
      var i = document.createElement("input"); i.type = "checkbox"; i.name = name; i.value = value; i.checked = checked;
      l.appendChild(i); l.appendChild(document.createTextNode(" " + value));
      return l;
    }
    function fill() {
      var c = cols(), sel = target.getAttribute("data-selected") || "";
      target.innerHTML = "";
      var pick = sel && c.indexOf(sel) >= 0 ? sel : (c.filter(function (x) { return guess.test(x); })[0] || c[c.length - 1]);
      c.forEach(function (name) {
        var o = document.createElement("option"); o.value = name; o.textContent = name; o.selected = name === pick;
        target.appendChild(o);
      });
      [sens, drop].forEach(function (box) {
        box.innerHTML = "";
        c.forEach(function (name) {
          if (name === target.value) return;
          box.appendChild(checkbox(box.getAttribute("data-name"), name, box === sens && sensGuess.test(name)));
        });
      });
    }
    function filterModels() {
      Array.prototype.forEach.call(builtin.options, function (o) { o.hidden = o.getAttribute("data-task") !== task.value; });
      if (builtin.selectedOptions[0] && builtin.selectedOptions[0].hidden) {
        var first = Array.prototype.filter.call(builtin.options, function (o) { return !o.hidden; })[0];
        if (first) first.selected = true;
      }
    }
    ds.addEventListener("change", function () { target.setAttribute("data-selected", ""); fill(); });
    target.addEventListener("change", function () { target.setAttribute("data-selected", target.value); fill(); });
    task.addEventListener("change", filterModels);
    fill(); filterModels();
  }

  // LLM form: keep provider in sync with the key, and show the request estimate.
  var key = document.getElementById("llm-key");
  if (key) {
    var provider = document.getElementById("llm-provider");
    var cases = document.getElementById("llm-cases");
    var cons = document.getElementById("llm-consistency");
    var autofix = document.getElementById("llm-autofix");
    var est = document.querySelector("[data-est]");
    var model = document.querySelector("input[name=llm_model]");
    function sync() {
      var p = key.selectedOptions[0].getAttribute("data-provider");
      provider.value = p;
      if (p !== "anthropic" && /^claude-/.test(model.value)) model.value = "";
    }
    function estimate() {
      var n = parseInt(cases.selectedOptions[0].getAttribute("data-n"), 10) || 0;
      var k = parseInt(cons.value, 10) || 0;
      var perPass = n + k * 2;
      est.textContent = String(perPass * (autofix.checked ? 8 : 1));
    }
    key.addEventListener("change", function () { sync(); estimate(); });
    [cases, cons, autofix].forEach(function (el) { el.addEventListener("change", estimate); el.addEventListener("input", estimate); });
    sync(); estimate();
  }

  // Keys form: base URL only for OpenAI-compatible endpoints.
  var kp = document.getElementById("key-provider");
  if (kp) {
    var urlField = document.querySelector("[data-for=openai_compatible]");
    function toggle() { urlField.hidden = kp.value !== "openai_compatible"; }
    kp.addEventListener("change", toggle); toggle();
  }
})();

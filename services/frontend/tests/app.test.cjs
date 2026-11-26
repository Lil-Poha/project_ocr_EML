// Запуск: node --test services/frontend/tests/app.test.cjs
const test = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");

function page() {
  const elements = new Map();
  const pending = [];
  const element = (id) => {
    if (!elements.has(id)) elements.set(id, {
      hidden: false, textContent: "", disabled: false, files: [], events: {},
      addEventListener(type, handler) { this.events[type] = handler; },
      setAttribute() {}, removeAttribute() {}, focus() {},
      classList: { add() {}, remove() {} },
    });
    return elements.get(id);
  };
  const context = vm.createContext({
    document: { getElementById: element }, window: { addEventListener() {} },
    URL: { createObjectURL: () => "blob:test", revokeObjectURL() {} },
    AbortController, FormData: class { append() {} }, setTimeout,
    fetch: (url, options) => new Promise((resolve) => pending.push({ resolve, options })),
  });
  vm.runInContext(fs.readFileSync(path.join(__dirname, "../public/app.js"), "utf8"), context);
  function select(name) {
    element("file-input").files = [{ name, type: "image/png", size: 100 }];
    element("file-input").events.change();
  }
  const respond = (index, text) => pending[index].resolve({
    ok: true, json: async () => ({ text, model: "stub", stub: true }),
  });
  return { element, pending, select, respond };
}

test("показывает ответ заглушки", async () => {
  const p = page();
  p.select("a.png");
  const request = p.element("recognize-btn").events.click();
  p.respond(0, "Тестовый текст");
  await request;
  assert.equal(p.element("result-text").textContent, "Тестовый текст");
  assert.equal(p.element("demo-badge").hidden, false);
  assert.equal(p.element("recognize-btn").disabled, false);
});

test("смена фото отменяет старый запрос и сохраняет результат нового", async () => {
  const p = page();
  p.select("a.png");
  const old = p.element("recognize-btn").events.click();
  p.element("reset-btn").events.click();
  assert.equal(p.pending[0].options.signal.aborted, true);
  p.select("b.png");
  const current = p.element("recognize-btn").events.click();
  p.respond(1, "Новое фото");
  await current;
  p.respond(0, "Старое фото");
  await old;
  assert.equal(p.element("result-text").textContent, "Новое фото");
  assert.equal(p.element("file-meta").textContent, "b.png · 1 КБ");
});

test("сброс во время запроса не показывает запоздавший ответ", async () => {
  const p = page();
  p.select("a.png");
  const request = p.element("recognize-btn").events.click();
  p.element("reset-btn").events.click();
  p.respond(0, "Старое фото");
  await request;
  assert.equal(p.element("result-text").textContent, "");
  assert.equal(p.element("workspace").hidden, true);
  assert.equal(p.element("recognize-btn").disabled, false);
});

/* SalesGod console: navigation shell, onboarding, customers & leads, call types, campaigns, follow-ups, settings, "Add with AI".
   Loaded by console.html before its own script; console.html calls window.SGP(SG) and gets back the router/shell.
   Every API path used here is listed in PLATFORM_SPEC.md (tests/test_ui_static.py checks that). A 404 from any of them shows
   a "not available yet" state instead of breaking the page. No external libraries. */
window.SGP = function (SG) {
'use strict';
const { h, sv, ic, ICONS, api, store, S, modal, notice, head, ago } = SG;
const LANGS = [['en', 'English'], ['hi', 'Hindi'], ['hinglish', 'Hinglish']];
const STAGES = ['new', 'contacted', 'interested', 'qualified', 'demo_booked', 'negotiating', 'won', 'lost', 'do_not_call'];
const STAGE_LABEL = { new: 'New', contacted: 'Contacted', interested: 'Interested', qualified: 'Qualified', demo_booked: 'Demo booked', negotiating: 'Negotiating', won: 'Won', lost: 'Lost', do_not_call: 'Do not call' };
const STAGE_TONE = { new: '', contacted: 'blue', interested: 'blue', qualified: 'green', demo_booked: 'green', negotiating: 'amber', won: 'green', lost: 'red', do_not_call: 'red' };
const ACTIONS = ['send_link', 'book_callback', 'book_demo', 'transfer_human', 'opt_out', 'send_sms', 'take_payment_link'];
const ANSWER_TYPES = ['text', 'yes_no', 'number', 'choice', 'date', 'none'];
const BUILTIN_CT = [['sales_pitch', 'Sales pitch'], ['lead_qualification', 'Lead qualification'], ['follow_up', 'Follow-up'], ['payment_reminder', 'Payment reminder'], ['feedback_survey', 'Feedback survey'], ['appointment_booking', 'Appointment booking'], ['custom', 'Custom']];

Object.assign(ICONS, {
  home: 'M3 11l9-8 9 8M5 10v10h5v-6h4v6h5V10',
  box: 'M21 8l-9-5-9 5v8l9 5 9-5zM3 8l9 5 9-5M12 13v9',
  list: 'M8 6h13M8 12h13M8 18h13M3 6h.01M3 12h.01M3 18h.01',
  users: 'M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2M9 11a4 4 0 1 0 0-8 4 4 0 0 0 0 8zM23 21v-2a4 4 0 0 0-3-3.9M16 3.1a4 4 0 0 1 0 7.8',
  mega: 'M3 11v2a1 1 0 0 0 1 1h3l8 5V5L7 10H4a1 1 0 0 0-1 1zM19 8a5 5 0 0 1 0 8',
  clock: 'M12 22a10 10 0 1 0 0-20 10 10 0 0 0 0 20zM12 6v6l4 2',
  cog: 'M12 15a3 3 0 1 0 0-6 3 3 0 0 0 0 6zM19.4 15a1.7 1.7 0 0 0 .3 1.8l.1.1a2 2 0 1 1-2.8 2.8l-.1-.1a1.7 1.7 0 0 0-1.8-.3 1.7 1.7 0 0 0-1 1.5V21a2 2 0 1 1-4 0v-.1a1.7 1.7 0 0 0-1.1-1.5 1.7 1.7 0 0 0-1.8.3l-.1.1a2 2 0 1 1-2.8-2.8l.1-.1a1.7 1.7 0 0 0 .3-1.8 1.7 1.7 0 0 0-1.5-1H3a2 2 0 1 1 0-4h.1a1.7 1.7 0 0 0 1.5-1.1 1.7 1.7 0 0 0-.3-1.8l-.1-.1a2 2 0 1 1 2.8-2.8l.1.1a1.7 1.7 0 0 0 1.8.3H9a1.7 1.7 0 0 0 1-1.5V3a2 2 0 1 1 4 0v.1a1.7 1.7 0 0 0 1 1.5 1.7 1.7 0 0 0 1.8-.3l.1-.1a2 2 0 1 1 2.8 2.8l-.1.1a1.7 1.7 0 0 0-.3 1.8V9a1.7 1.7 0 0 0 1.5 1H21a2 2 0 1 1 0 4h-.1a1.7 1.7 0 0 0-1.5 1z',
  search: 'M11 19a8 8 0 1 0 0-16 8 8 0 0 0 0 16zM21 21l-4.3-4.3',
  menu: 'M3 6h18M3 12h18M3 18h18',
  x: 'M18 6L6 18M6 6l12 12',
  check: 'M20 6L9 17l-5-5',
  upload: 'M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4M17 8l-5-5-5 5M12 3v12',
  grip: 'M9 5h.01M9 12h.01M9 19h.01M15 5h.01M15 12h.01M15 19h.01',
  link: 'M10 13a5 5 0 0 0 7.5.5l3-3a5 5 0 0 0-7-7l-1.7 1.7M14 11a5 5 0 0 0-7.5-.5l-3 3a5 5 0 0 0 7 7l1.7-1.7',
  copy: 'M9 9h11v11H9zM5 15H4a1 1 0 0 1-1-1V4a1 1 0 0 1 1-1h10a1 1 0 0 1 1 1v1',
  tag: 'M20.6 13.4l-7.2 7.2a2 2 0 0 1-2.8 0L2 12V2h10l8.6 8.6a2 2 0 0 1 0 2.8zM7 7h.01',
  board: 'M3 3h6v18H3zM9 3h6v10H9zM15 3h6v14h-6z',
  table: 'M3 3h18v18H3zM3 9h18M3 15h18M9 3v18',
  ban: 'M12 22a10 10 0 1 0 0-20 10 10 0 0 0 0 20zM4.9 4.9l14.2 14.2',
  arrow: 'M5 12h14M13 6l6 6-6 6',
  up: 'M12 19V5M5 12l7-7 7 7',
  down: 'M12 5v14M19 12l-7 7-7-7',
  dup: 'M8 8h12v12H8zM4 16V4h12',
  pause: 'M6 4h4v16H6zM14 4h4v16h-4z',
  user: 'M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2M12 11a4 4 0 1 0 0-8 4 4 0 0 0 0 8z',
});

/* ------------------------------------------------------------ small helpers */
const esc = encodeURIComponent;
const qs = (o) => new URLSearchParams(Object.entries(o).filter(([, v]) => v !== '' && v != null)).toString();
const arr = (j, k) => (Array.isArray(j) ? j : (j && j[k]) || []);
const slug = (t) => String(t || '').toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-|-$/g, '');
const clone = (o) => JSON.parse(JSON.stringify(o ?? null));
const lines = (v) => String(v || '').split('\n').map((x) => x.trim()).filter(Boolean);
const csv = (v) => String(v || '').split(',').map((x) => x.trim()).filter(Boolean);
const num = (v) => { const x = parseFloat(String(v).replace(/,/g, '')); return Number.isFinite(x) ? x : null; };
const when = (t) => { if (!t) return '—'; const d = typeof t === 'number' ? new Date(t * 1000) : new Date(t); return isNaN(d) ? String(t) : d.toLocaleString([], { dateStyle: 'medium', timeStyle: 'short' }); };
const toast = (m) => { const t = h('div', { class: 'toast', role: 'status' }, m); document.body.append(t); setTimeout(() => t.remove(), 2800); };
async function tryApi(path, opts) {
  try { return { ok: true, data: await api(path, opts) }; }
  catch (e) { return { ok: false, na: e.status === 404 || e.status === 405 || e.status === 501, status: e.status, err: e.message }; }
}
const post = (path, body, method) => tryApi(path, { method: method || 'POST', body: JSON.stringify(body || {}) });
/* create (no id) or update (id); a server that wants PUT for create gets a second try */
async function saveObj(base, id, obj) {
  if (id) return post(`${base}/${esc(id)}`, obj, 'PUT');
  if (!obj.id) { obj = { ...obj }; delete obj.id; }
  let r = await post(base, obj);
  if (!r.ok && r.status === 405 && obj.id) r = await post(`${base}/${esc(obj.id)}`, obj, 'PUT');
  return r;
}
const proj = () => S.tenants.find((t) => t.id === S.tenant) || { id: S.tenant, name: '—', color: '#888' };
const isDnc = (c) => !!(c && ((c.consent && c.consent.dnc) || c.dnc || c.stage === 'do_not_call'));
const hh = (n) => String(n).padStart(2, '0') + ':00';
const hoursOf = (p) => (p && Array.isArray(p.call_hours) && p.call_hours.length === 2 ? p.call_hours : [10, 20]);
const hourSel = (v, label, lo, hi) => h('select', { 'aria-label': label }, Array.from({ length: hi - lo + 1 }, (_, i) => lo + i).map((n) => h('option', { value: n, selected: n === v }, hh(n))));
const stageName = (s) => (proj().stages && proj().stages[s]) || STAGE_LABEL[s] || s;
function naBox(what, r, retry) {
  if (r && r.na) return h('div', { class: 'empty na' }, h('b', {}, what + ' is not available yet'), 'This server does not have that part of the API yet. Everything else keeps working, and this page fills in as soon as it does.');
  return h('div', { class: 'empty' }, h('b', {}, 'Could not load ' + what.toLowerCase()), (r && r.err) || 'Unknown error', retry ? h('div', { style: 'margin-top:10px' }, h('button', { class: 'btn sm', onclick: retry }, ic('refresh', 'sm'), 'Try again')) : null);
}
const fld = (label, el, hint, cls) => h('div', { class: cls || '' }, h('label', { class: 'lbl' }, label), el, hint ? h('div', { class: 'hint', style: 'margin-top:4px' }, hint) : null);
const TONES = [['professional', 'Professional', 'Sir, yeh hamari sabse achhi offer hai.'], ['mid', 'Friendly (mid)', 'Dekhiye, aapke liye 2,799 kar sakta hoon, aaj join kar lenge?'], ['casual', 'Casual (non-professional)', 'Bhai, 2,799 kar deta hoon, aaj hi le lo?']];
/* Tone dropdown (field tone_level, default mid) with a one-line example under the chosen option. */
function toneField(value) {
  const cur = TONES.some((t) => t[0] === value) ? value : 'mid';
  const ex = h('div', { class: 'hint tone-ex' });
  const sel = h('select', { 'aria-label': 'Tone', onchange: () => show() }, TONES.map(([k, n]) => h('option', { value: k, selected: k === cur }, n)));
  const show = () => { const t = TONES.find((x) => x[0] === sel.value); ex.textContent = 'e.g. \u201c' + t[2] + '\u201d'; };
  show();
  return { sel, el: h('div', {}, h('label', { class: 'lbl' }, 'Tone'), sel, ex, h('div', { class: 'hint tone-all' }, TONES.map(([k, n, e]) => h('div', {}, h('b', {}, n + ': '), '\u201c' + e + '\u201d')))) };
}
const langBlock = (obj, rows) => { const out = {}; const box = h('div', { class: 'langs' }, LANGS.map(([k, n]) => h('div', {}, h('span', { class: 'hint' }, n), out[k] = h('textarea', { rows: rows || 2, 'aria-label': n }, (obj || {})[k] || '')))); return { box, get: () => { const o = {}; LANGS.forEach(([k]) => { if (out[k].value.trim()) o[k] = out[k].value.trim(); }); return o; }, set: (v) => LANGS.forEach(([k]) => { out[k].value = (v || {})[k] || ''; }) }; };
const tabs = (items, cur, on) => h('div', { class: 'ptabs', role: 'tablist' }, items.map(([id, label]) => h('button', { role: 'tab', 'aria-selected': id === cur ? 'true' : 'false', class: id === cur ? 'on' : '', onclick: () => on(id) }, label)));
const confirmBox = (title, desc, okLabel, onOk, opts) => modal(title, desc, (b, f, close) => {
  let typed; const o = opts || {};
  if (o.type) b.append(h('label', { class: 'lbl' }, 'Type ' + o.type + ' to confirm'), typed = h('input', { autocomplete: 'off' }));
  const ok = h('button', { class: 'btn pri' + (o.danger ? ' danger' : ''), onclick: async () => { if (o.type && typed.value.trim().toUpperCase() !== o.type) return; ok.disabled = true; await onOk(); close(); } }, okLabel);
  f.append(ok, h('button', { class: 'btn ghost', onclick: close }, 'Cancel'));
});

/* ------------------------------------------------------------ Add with AI */
function fmtVal(v) { if (v == null) return ''; if (typeof v === 'string') return v; if (Array.isArray(v) && v.every((x) => typeof x === 'string')) return v.join('\n'); return JSON.stringify(v, null, 1); }
/* drop empty strings / null / [] / {} recursively (the server's draft carries every model default) */
function prune(v) {
  if (Array.isArray(v)) { const a = v.map(prune).filter((x) => x !== undefined); return a.length ? a : undefined; }
  if (v && typeof v === 'object') { const o = {}; for (const [k, x] of Object.entries(v)) { const p = prune(x); if (p !== undefined) o[k] = p; } return Object.keys(o).length ? o : undefined; }
  return v === '' || v == null ? undefined : v;
}
const same = (a, b) => JSON.stringify(a ?? null) === JSON.stringify(b ?? null);
/* aiBox: a prompt box for any form. It never saves; "Apply" only fills the form so the operator can edit and press the form's own Save. */
function aiBox(o) {
  const ta = h('textarea', { rows: 2, placeholder: o.placeholder || 'Describe what you want, in your own words', 'aria-label': 'Describe what you want the AI to draft' });
  const out = h('div', { class: 'ai-out' });
  let btn;
  const run = async () => {
    if (!ta.value.trim()) { ta.focus(); return; }
    btn.disabled = true; btn.lastChild.textContent = 'Drafting…'; out.replaceChildren();
    const r = await post('/api/ai/assist', { kind: o.kind, prompt: ta.value.trim(), context: o.context ? o.context() : {} });
    btn.disabled = false; btn.lastChild.textContent = 'Draft with AI';
    if (!r.ok) { out.append(r.na ? h('div', { class: 'hint' }, 'AI drafting is not available on this server yet. You can still fill the form by hand.') : h('div', { class: 'err' }, r.err)); return; }
    const d = r.data.draft || r.data.result || r.data;
    showDraft(d && typeof d === 'object' ? d : {});
  };
  function showDraft(d) {
    const cur = (o.current && o.current()) || {};
    d = prune(d) || {};
    const keys = Object.keys(d).filter((k) => !k.startsWith('_') && (!o.keys || o.keys.includes(k)));
    if (!keys.length) { out.append(h('div', { class: 'hint' }, 'The AI returned nothing usable. Try describing it with more detail.')); return; }
    const checks = {};
    const rows = keys.map((k) => {
      const changed = !same(cur[k], d[k]);
      return h('div', { class: 'drow' + (changed ? ' chg' : '') },
        h('label', { class: 'dk' }, checks[k] = h('input', { type: 'checkbox', checked: changed, style: 'width:auto;height:auto', 'aria-label': 'Use ' + k }), k.replaceAll('_', ' ')),
        h('div', { class: 'dv' }, fmtVal(cur[k]) ? h('div', { class: 'old' }, fmtVal(cur[k])) : null, h('div', { class: 'new' }, fmtVal(d[k])), changed ? null : h('span', { class: 'hint' }, 'same as now')));
    });
    out.append(h('div', { class: 'diff' }, h('div', { class: 'sec-lbl', style: 'margin-bottom:6px' }, 'Draft · green is new, struck-through is what is there now'), rows,
      h('div', { class: 'row', style: 'margin-top:10px' }, h('button', { class: 'btn pri sm', type: 'button', onclick: () => {
        const pick = {}; keys.forEach((k) => { if (checks[k].checked) pick[k] = d[k]; });
        o.apply(pick); out.replaceChildren(h('div', { class: 'hint' }, 'Applied to the form. Edit anything you like, then press Save: nothing is saved until you do.'));
      } }, ic('check', 'sm'), 'Apply to the form'), h('button', { class: 'btn ghost sm', type: 'button', onclick: () => out.replaceChildren() }, 'Discard'))));
  }
  return h('div', { class: 'ai' }, h('div', { class: 'ai-h' }, ic('spark', 'sm'), h('b', {}, o.title || 'Add with AI'), h('span', { class: 'hint' }, 'Writes a draft for you to review. Nothing is saved until you save.')),
    ta, h('div', { class: 'row', style: 'margin-top:8px' }, btn = h('button', { class: 'btn sm', type: 'button', onclick: run }, ic('spark', 'sm'), h('span', {}, 'Draft with AI'))), out);
}

/* ------------------------------------------------------------ router + shell */
const NAV = [['dashboard', 'Dashboard', 'home'], ['products', 'Products', 'box'], ['calltypes', 'Call types', 'list'], ['customers', 'Customers & Leads', 'users'], ['campaigns', 'Campaigns', 'mega'], ['followups', 'Follow-ups', 'clock'], ['calls', 'Calls', 'phone'], ['settings', 'Settings', 'cog']];
const SECTION_OF = { tree: 'products', sim: 'calls', progress: 'dashboard', onboarding: 'dashboard' };
const TITLE = { dashboard: 'Dashboard', products: 'Products', calltypes: 'Call types', customers: 'Customers & Leads', campaigns: 'Campaigns', followups: 'Follow-ups', calls: 'Calls', settings: 'Settings', tree: 'Decision tree', sim: 'Simulator', progress: 'Progress', onboarding: 'Set up a project' };
const PS = { route: null, tok: 0, navOpen: false, menu: false, crumb: '', ob: null };
const INTERNAL = { calls: 'live' };
function parseHash() {
  const x = (location.hash || '').replace(/^#\/?/, ''); const [path, query] = x.split('?');
  const seg = path.split('/').filter(Boolean).map(decodeURIComponent);
  return { name: TITLE[seg[0]] ? seg[0] : 'dashboard', id: seg[1] || null, q: new URLSearchParams(query || '') };
}
function go(p, a) {
  const name = p === 'live' ? 'calls' : p;
  let hash = '#/' + name; if (a && a.id) hash += '/' + esc(a.id); if (a && a.query) hash += '?' + new URLSearchParams(a.query);
  if (location.hash === hash) refresh(); else location.hash = hash;
}
function readRoute() {
  const r = PS.route = parseHash(); S.page = INTERNAL[r.name] || r.name; PS.crumb = '';
  if (r.name === 'calls' && r.id) { S.sel = r.id; S.rtab = 'map'; }
  store.set('sg_page', r.name);
}
function refresh() { if (S.authed) render($('#app')); }
window.addEventListener('hashchange', () => { if (S.authed) { PS.navOpen = false; refresh(); } });
const $ = (q) => document.querySelector(q);
document.addEventListener('keydown', (e) => { if (e.key === '/' && !/INPUT|TEXTAREA|SELECT/.test((document.activeElement || {}).tagName || '')) { const s = $('#gsearch'); if (s) { e.preventDefault(); s.focus(); } } });
document.addEventListener('click', (e) => { if (PS.menu && !e.target.closest('.switcher')) { PS.menu = false; const m = $('#pmenu'); if (m) m.remove(); } if (!e.target.closest('.searchbox')) { const r = $('#sres'); if (r) r.hidden = true; } });

function render(app) {
  readRoute();
  const t = proj(), dark = !document.documentElement.classList.contains('light'), cur = SECTION_OF[PS.route.name] || PS.route.name;
  app.innerHTML = '';
  app.append(h('div', { class: 'gs' + (PS.navOpen ? ' navopen' : '') },
    h('aside', { class: 'side', id: 'side', 'aria-label': 'Main navigation' },
      h('div', { class: 'brand' }, h('span', { class: 'mk' }, h('i')), h('b', {}, 'SalesGod'), h('small', {}, 'Voice')),
      h('div', { class: 'switcher' },
        h('button', { class: 'org', 'aria-haspopup': 'listbox', 'aria-expanded': String(PS.menu), onclick: () => { PS.menu = !PS.menu; const m = $('#pmenu'); if (m) { m.remove(); return; } $('.switcher').append(projectMenu()); } },
          h('span', { class: 'dot', style: `background:${t.color || '#888'}` }), h('span', { style: 'min-width:0;flex:1' }, h('small', {}, 'Project'), h('b', {}, t.name)), ic('chev', 'sm'))),
      h('nav', { class: 'nav' }, NAV.map(([id, label, icon]) => h('a', { class: 'nav-item' + (cur === id ? ' on' : ''), href: '#/' + id, 'aria-current': cur === id ? 'page' : null }, ic(icon), h('span', {}, label), id === 'calls' ? h('span', { class: 'cnt', id: 'livecnt' }) : null))),
      h('div', { class: 'side-foot' }, h('a', { class: 'nav-item', href: '#/onboarding' }, ic('spark'), h('span', {}, 'Guided setup')))),
    h('div', { class: 'scrim', onclick: () => { PS.navOpen = false; refresh(); } }),
    h('div', { class: 'main' },
      h('header', { class: 'topbar' },
        h('button', { class: 'icon-btn burger', title: 'Menu', 'aria-label': 'Open menu', onclick: () => { PS.navOpen = !PS.navOpen; $('.gs').classList.toggle('navopen', PS.navOpen); } }, ic('menu')),
        h('nav', { class: 'crumbs', 'aria-label': 'Breadcrumb' }, h('a', { href: '#/dashboard' }, t.name), h('i', {}, '/'), crumbs()),
        searchBox(),
        h('span', { class: 'conn ' + S.conn, id: 'conn', title: 'Live event stream' }, h('i'), h('span', {}, S.conn === 'ok' ? 'LIVE' : S.conn === 'bad' ? 'RECONNECTING' : 'CONNECTING')),
        h('a', { class: 'btn ghost sm hide-sm', href: '/classic' }, 'Classic view'),
        h('button', { class: 'icon-btn', title: dark ? 'Light theme' : 'Dark theme', 'aria-label': 'Toggle theme', onclick: () => { const l = document.documentElement.classList.toggle('light'); store.set('sg_theme', l ? 'light' : 'dark'); refresh(); } }, ic(dark ? 'sun' : 'moon'))),
      h('main', { class: 'content', id: 'page' }))));
  renderPage();
}
function crumbs() {
  const r = PS.route, sec = SECTION_OF[r.name] || r.name, out = [];
  if (SECTION_OF[r.name] && sec !== r.name) out.push(h('a', { href: '#/' + sec }, TITLE[sec]), h('i', {}, '/'), h('b', {}, TITLE[r.name]));
  else if (r.id) out.push(h('a', { href: '#/' + r.name }, TITLE[r.name]), h('i', {}, '/'), h('b', { id: 'crumb' }, PS.crumb || (r.id === 'new' ? 'New' : r.id)));
  else out.push(h('b', {}, TITLE[r.name]));
  return out;
}
const setCrumb = (t) => { PS.crumb = t; const e = $('#crumb'); if (e) e.textContent = t; };
function projectMenu() {
  return h('div', { class: 'menu pm', id: 'pmenu', role: 'listbox' }, h('div', { class: 'group', style: 'padding:6px 9px' }, 'Switch project'),
    S.tenants.map((x) => h('button', { role: 'option', onclick: () => { S.tenant = x.id; store.set('sg_tenant', x.id); PS.menu = false; S.sel = null; S.tree.pid = null; S.sim.sid = null; S.sim.turns = []; PS.ob = null; if (PS.route.id) location.hash = '#/' + PS.route.name; refresh(); } },
      h('span', { style: `width:8px;height:8px;border-radius:2px;flex:none;background:${x.color || '#888'}` }), x.name, x.id === S.tenant ? h('span', { class: 'hint', style: 'margin-left:auto' }, 'current') : null)),
    h('div', { class: 'sep' }), h('button', { onclick: () => { PS.menu = false; go('onboarding', { query: { new: 1 } }); } }, ic('plus', 'sm'), 'New project (guided setup)'));
}
function renderPage() {
  const el = $('#page'); if (!el) return;
  const name = PS.route.name, tok = ++PS.tok, ctx = { el, tok, alive: () => tok === PS.tok, id: PS.route.id, q: PS.route.q };
  const fn = PAGES[name] || ((e) => (SG.pages[S.page] || SG.pages.live)(e));
  try { const p = fn(el, ctx); if (p && p.catch) p.catch((e) => { if (ctx.alive()) el.replaceChildren(naBox('This page', { err: e.message })); }); }
  catch (e) { el.replaceChildren(naBox('This page', { err: e.message })); }
}

/* global search: customers (server), products and campaigns (local) */
function searchBox() {
  const res = h('div', { class: 'sres', id: 'sres', hidden: true, role: 'listbox' });
  let timer, seq = 0;
  const inp = h('input', { id: 'gsearch', type: 'search', placeholder: 'Search customers, products, campaigns  ( / )', 'aria-label': 'Search', autocomplete: 'off',
    onkeydown: (e) => { if (e.key === 'Escape') { inp.value = ''; res.hidden = true; } if (e.key === 'Enter') { const a = res.querySelector('a'); if (a) a.click(); } },
    oninput: () => { clearTimeout(timer); timer = setTimeout(() => doSearch(inp.value.trim(), res, ++seq, () => seq), 220); } });
  return h('div', { class: 'searchbox' }, ic('search', 'sm'), inp, res);
}
async function doSearch(q, res, mine, cur) {
  if (q.length < 2) { res.hidden = true; return; }
  const ql = q.toLowerCase();
  const prods = SG.tenantProducts().filter((p) => (p.product_name || '').toLowerCase().includes(ql)).slice(0, 4);
  const [cu, ca] = await Promise.all([tryApi('/api/customers?' + qs({ project: S.tenant, q, page_size: 6 })), campaigns()]);
  if (mine !== cur()) return;
  const cust = cu.ok ? arr(cu.data, 'customers').slice(0, 6) : [], camps = (ca || []).filter((c) => (c.name || '').toLowerCase().includes(ql)).slice(0, 4);
  const grp = (t, items) => items.length ? [h('div', { class: 'group', style: 'padding:6px 12px 2px' }, t), items] : null;
  res.replaceChildren(...[
    grp('Customers', cust.map((c) => h('a', { href: '#/customers/' + esc(c.id), onclick: () => { res.hidden = true; } }, h('b', {}, c.name || c.phone), h('span', { class: 'hint' }, ' ' + (c.phone || ''))))),
    grp('Products', prods.map((p) => h('a', { href: '#/products', onclick: () => { res.hidden = true; } }, h('b', {}, p.product_name)))),
    grp('Campaigns', camps.map((c) => h('a', { href: '#/campaigns/' + esc(c.id), onclick: () => { res.hidden = true; } }, h('b', {}, c.name)))),
  ].flat(Infinity).filter(Boolean));
  if (!res.childNodes.length) res.append(h('div', { class: 'hint', style: 'padding:10px 12px' }, cu.ok ? 'No matches.' : 'No matching products or campaigns' + (cu.na ? ' (customer search is not available yet).' : '.')));
  res.hidden = false;
}
let _camps = null;
async function campaigns(force) {
  if (_camps && !force && _camps.p === S.tenant) return _camps.list;
  const r = await tryApi('/api/campaigns?' + qs({ project: S.tenant })); const list = r.ok ? arr(r.data, 'campaigns') : [];
  _camps = { p: S.tenant, list, ok: r.ok, na: r.na }; return list;
}
let _cts = null;
async function calltypes(force) {
  if (_cts && !force && _cts.p === S.tenant) return _cts.res;
  const r = await tryApi('/api/calltypes?' + qs({ project: S.tenant })); const res = { ok: r.ok, na: r.na, err: r.err, list: r.ok ? arr(r.data, 'calltypes') : [] };
  if (r.ok && !res.list.length) res.list = []; _cts = { p: S.tenant, res }; return res;
}
const ctOptions = (res) => (res.list.length ? res.list.map((c) => [c.id, c.name || c.id]) : BUILTIN_CT);
const fieldDefs = async () => { const r = await tryApi(`/api/projects/${esc(S.tenant)}/fields`); return r.ok ? arr(r.data, 'fields') : (proj().fields || []); };

/* place a real call for one customer (always confirmed by the operator first) */
function callNow(c, o) {
  const label = o.callTypeName || o.callType || 'the default call type';
  confirmBox('Call ' + (c.name || c.phone) + ' now?', `This places a real call to ${c.phone} using ${label}. Calls are only made when you press this button or press Call next to a person in a campaign queue.`, 'Place the call', async () => {
    let r;
    if (o.campaignId) r = await post(`/api/campaigns/${esc(o.campaignId)}/call/${esc(c.id)}`, { call_type: o.callType || undefined });
    else {
      const pid = o.productId || SG.tenantProducts()[0]?.id;
      if (!pid) { notice('Add a product to this project first: the call needs something to talk about.'); return; }
      r = await post(`/api/customers/${esc(c.id)}/call`, { product_id: pid, call_type: o.callType || undefined });
    }
    if (!r.ok) { notice(r.na ? 'Calling is not available on this server yet.' : r.err); return; }
    const d = r.data, first = (d.calls || [])[0] || d;
    if (first.error) { notice(first.error); return; }
    await SG.loadCalls().catch(() => {}); go('calls', { id: first.call_id });
  });
}

/* ------------------------------------------------------------ onboarding state (project record, localStorage fallback) */
const obKey = (id) => 'sg_onb_' + (id || 'new');
function obLoad(id) {
  let st = null; try { st = JSON.parse(store.get(obKey(id), 'null')); } catch { st = null; }
  const p = S.tenants.find((t) => t.id === id);
  if (p && p.onboarding && (!st || (p.onboarding.step || 0) > (st.step || 0))) st = p.onboarding;
  return { step: 1, projectId: id || '', productId: '', calltypeId: '', campaignId: '', customers: 0, tested: false, done: false, basics: {}, ...(st || {}) };
}
async function obSave(st) {
  store.set(obKey(st.projectId), JSON.stringify(st));
  const p = S.tenants.find((t) => t.id === st.projectId);
  if (p && S.projectsApi) { const { sim, draftCT, ...keep } = st; p.onboarding = keep; await post(`/api/projects/${esc(p.id)}`, { onboarding: keep }, 'PUT'); }
}

/* ------------------------------------------------------------ Dashboard */
async function pageDashboard(el, ctx) {
  el.replaceChildren(h('div', { class: 'pad' }, head('Dashboard', 'Where this project stands today.', h('a', { class: 'btn sm', href: '#/progress' }, ic('activity', 'sm'), 'Build progress'), h('a', { class: 'btn pri sm', href: '#/onboarding' }, ic('spark', 'sm'), 'Guided setup')), h('div', { class: 'empty' }, 'Loading…')));
  const [cu, ca, fu, ct] = await Promise.all([tryApi('/api/customers?' + qs({ project: S.tenant, page_size: 1 })), campaigns(true), tryApi('/api/followups?' + qs({ project: S.tenant, due: 'today' })), calltypes()]);
  if (!ctx.alive()) return;
  const prods = SG.tenantProducts(), calls = SG.tenantCalls();
  const customers = cu.ok ? (cu.data.total ?? arr(cu.data, 'customers').length) : null, due = fu.ok ? arr(fu.data, 'followups').length : null;
  const running = _camps && _camps.ok ? ca.filter((c) => c.status === 'running').length : null;
  const tile = (v, k, href, na) => h('a', { class: 'tile', href }, h('div', { class: 'v' }, v == null ? '—' : String(v)), h('div', { class: 'k' }, k), na ? h('div', { class: 'hint' }, 'not available yet') : null);
  const steps = [['Add a product', prods.length > 0, '#/products'], ['Pick a call type', ct.ok && ct.list.length > 0, '#/calltypes'], ['Add customers', (customers || 0) > 0, '#/customers'], ['Create a campaign', ca.length > 0, '#/campaigns'], ['Run a test call', obLoad(S.tenant).tested, '#/onboarding']];
  const doneN = steps.filter((s) => s[1]).length;
  el.replaceChildren(h('div', { class: 'pad' },
    head('Dashboard', 'Where ' + proj().name + ' stands today.', h('a', { class: 'btn sm', href: '#/progress' }, ic('activity', 'sm'), 'Build progress'), h('a', { class: 'btn pri sm', href: '#/onboarding' }, ic('spark', 'sm'), 'Guided setup')),
    h('div', { class: 'tiles' }, tile(customers, 'Customers & leads', '#/customers', cu.na), tile(running, 'Campaigns running', '#/campaigns', _camps && _camps.na), tile(due, 'Follow-ups due today', '#/followups', fu.na), tile(calls.length, 'Calls', '#/calls'), tile(prods.length, 'Products', '#/products')),
    h('div', { class: 'cols2' },
      h('div', { class: 'panel' }, h('div', { class: 'panel-head' }, h('b', {}, 'Setup'), h('span', { class: 'tag ' + (doneN === steps.length ? 'green' : 'amber') }, doneN + ' of ' + steps.length)),
        steps.map(([t, ok, href]) => h('a', { class: 'item2', href }, h('span', { class: 'chk' + (ok ? ' ok' : '') }, ok ? ic('check', 'sm') : null), h('span', {}, t), ic('arrow', 'sm')))),
      h('div', { class: 'panel' }, h('div', { class: 'panel-head' }, h('b', {}, 'Recent calls'), h('a', { class: 'btn ghost sm', style: 'margin-left:auto', href: '#/calls' }, 'All calls')),
        calls.length ? calls.slice(0, 6).map((c) => h('a', { class: 'item2', href: '#/calls/' + esc(c.call_id) }, h('span', {}, c.customer_name || SG.mask(c.to), h('span', { class: 'hint' }, '  ' + SG.pname(c.campaign))), h('span', { class: 'st ' + c.status, style: 'margin-left:auto' }, c.status))) : h('div', { class: 'empty' }, 'No calls yet.')))));
}

/* ------------------------------------------------------------ Products */
function pageProducts(el) {
  const ps = SG.tenantProducts(), pg = (extra) => '/productgen?' + qs({ project: S.tenant, return: '/#/products', ...extra });
  el.replaceChildren(h('div', { class: 'pad' },
    head('Products', 'What the agent sells. Each product has its own offer, knowledge and decision tree.', h('a', { class: 'btn pri', href: pg() }, ic('spark', 'sm'), 'Add product with AI'), h('button', { class: 'btn', onclick: () => SG.editProduct(null) }, ic('plus', 'sm'), 'Add by hand')),
    ps.length ? h('div', { class: 'cards' }, ps.map((x) => h('div', { class: 'pcard' },
      h('div', { class: 'k' }, x.seller_name || 'Product'), h('b', {}, x.product_name),
      h('dl', {}, h('dt', {}, 'offer'), h('dd', {}, `${x.offer_price || '—'} ${x.discount || ''}`), h('dt', {}, 'normal'), h('dd', {}, x.original_price || '—'), h('dt', {}, 'agent'), h('dd', {}, x.agent_name || '—'),
        h('dt', {}, 'can call'), h('dd', {}, (x.allowed_numbers || []).length ? x.allowed_numbers.length + ' test numbers only' : 'any number')),
      h('div', { class: 'row', style: 'margin-top:12px;flex-wrap:wrap' }, h('button', { class: 'btn sm', onclick: () => SG.editProduct(x.id) }, 'Edit'), h('button', { class: 'btn sm', onclick: () => { S.tree.pid = x.id; S.tree.pb = null; go('tree'); } }, ic('tree', 'sm'), 'Decision tree'),
        h('a', { class: 'btn sm', href: '/productgen?' + qs({ product: S.tenant + '/' + x.id, return: '/#/products' }) }, 'Knowledge'))))) : h('div', { class: 'empty' }, h('b', {}, 'No products in this project yet'), 'Type just a name under Add product with AI and the knowledge, offer and decision tree are drafted for you.')));
}

/* ------------------------------------------------------------ Call types */
const blankCT = () => ({ id: '', name: '', objective: '', opening: {}, steps: [], allowed_actions: [], success_outcomes: [], closing: {}, tone: '', max_call_seconds: 360, selling_allowed: false, negotiation_allowed: false, compliance_lines: [] });
async function pageCallTypes(el, ctx) {
  if (ctx.id) return ctEditorPage(el, ctx);
  el.replaceChildren(h('div', { class: 'pad' }, h('div', { class: 'empty' }, 'Loading…')));
  const res = await calltypes(true); if (!ctx.alive()) return;
  const list = res.list.length ? res.list : (res.ok ? BUILTIN_CT.map(([id, name]) => ({ id, name, builtin: true })) : []);
  el.replaceChildren(h('div', { class: 'pad' }, head('Call types', 'A call type is a template for one kind of call: what it is for, what it says first, the steps it follows and what it may do.',
    h('button', { class: 'btn', onclick: () => go('calltypes', { id: 'new', query: { ai: 1 } }) }, ic('spark', 'sm'), 'Describe one with AI'), h('button', { class: 'btn pri', onclick: () => go('calltypes', { id: 'new' }) }, ic('plus', 'sm'), 'New call type')),
    !res.ok ? naBox('Call types', res, () => { _cts = null; pageCallTypes(el, ctx); }) : h('div', { class: 'cards' }, list.map((c) => h('div', { class: 'pcard' },
      h('div', { class: 'k' }, c.builtin ? 'Built in' : 'Custom'), h('b', {}, c.name || c.id), h('p', { class: 'hint', style: 'margin:6px 0 0;min-height:34px' }, c.objective || ''),
      h('div', { class: 'row', style: 'margin:8px 0;flex-wrap:wrap;gap:5px' }, h('span', { class: 'tag' }, (c.steps || []).length + ' steps'), c.selling_allowed ? h('span', { class: 'tag sig' }, 'selling') : null, c.negotiation_allowed ? h('span', { class: 'tag amber' }, 'negotiation') : null),
      h('div', { class: 'row', style: 'flex-wrap:wrap' }, h('button', { class: 'btn sm', onclick: () => go('calltypes', { id: c.id }) }, c.builtin ? 'View' : 'Edit'), h('button', { class: 'btn sm', onclick: () => dupCT(c) }, ic('dup', 'sm'), 'Duplicate')))))));
}
async function dupCT(c) {
  const r = await post(`/api/calltypes/${esc(c.id)}/duplicate`, { project: S.tenant, project_id: S.tenant });
  if (!r.ok) return notice(r.na ? 'Duplicating is not available on this server yet.' : r.err);
  _cts = null; toast('Duplicated. Opening the copy.'); const n = r.data.calltype || r.data; go('calltypes', { id: n.id || c.id });
}
async function ctEditorPage(el, ctx) {
  const isNew = ctx.id === 'new';
  const res = await calltypes(); if (!ctx.alive()) return;
  let base = isNew ? blankCT() : res.list.find((c) => c.id === ctx.id);
  if (!base && !isNew && !res.list.length) base = { ...blankCT(), id: ctx.id, name: (BUILTIN_CT.find((b) => b[0] === ctx.id) || [])[1] || ctx.id, builtin: true };
  if (!base) { el.replaceChildren(h('div', { class: 'pad' }, naBox('This call type', { err: 'Not found' }))); return; }
  const ro = !!base.builtin, d = { ...blankCT(), ...clone(base) };
  setCrumb(d.name || 'New call type');
  const dis = ro ? true : null, I = (v, a) => h('input', { value: v ?? '', disabled: dis, ...(a || {}) });
  const f = { name: I(d.name), objective: h('textarea', { rows: 2, disabled: dis }, d.objective || ''), tone: I(d.tone), max: I(d.max_call_seconds ?? '', { type: 'number', min: 30, step: 30 }) };
  const opening = langBlock(d.opening, 2), closing = langBlock(d.closing, 2);
  const outcomes = h('textarea', { rows: 2, disabled: dis, 'aria-label': 'Success outcomes', placeholder: 'one per line, e.g. demo_booked' }, (d.success_outcomes || []).join('\n'));
  const comp = h('textarea', { rows: 2, disabled: dis, placeholder: 'lines the agent must say or never say' }, (d.compliance_lines || []).join('\n'));
  const sell = h('input', { type: 'checkbox', checked: !!d.selling_allowed, disabled: dis, style: 'width:auto;height:auto' }), neg = h('input', { type: 'checkbox', checked: !!d.negotiation_allowed, disabled: dis, style: 'width:auto;height:auto' });
  const actions = new Set(d.allowed_actions || []);
  const actBox = h('div', { class: 'chips' }, [...new Set([...ACTIONS, ...actions])].map((a) => { const cb = h('input', { type: 'checkbox', checked: actions.has(a), disabled: dis, style: 'width:auto;height:auto', onchange: (e) => { e.target.checked ? actions.add(a) : actions.delete(a); } }); return h('label', { class: 'chip' }, cb, a.replaceAll('_', ' ')); }));
  const stepsBox = h('div', { class: 'steps' }); let dragI = null;
  const stepId = (s, i) => s.id || 'step_' + (i + 1);
  function drawSteps() {
    stepsBox.replaceChildren(...d.steps.map((s, i) => {
      const txt = langBlock(s.text, 2), id = I(s.id, { class: 'mono', placeholder: 'step_' + (i + 1), oninput: (e) => { s.id = e.target.value.trim(); } });
      const at = h('select', { disabled: dis, onchange: (e) => { s.answer_type = e.target.value; } }, ANSWER_TYPES.map((t) => h('option', { value: t, selected: (s.answer_type || 'text') === t }, t.replace('_', '/'))));
      const req = h('input', { type: 'checkbox', checked: !!s.required, disabled: dis, style: 'width:auto;height:auto', onchange: (e) => { s.required = e.target.checked; } });
      const sync = () => { s.text = txt.box.querySelectorAll('textarea').length ? txt.get() : {}; };
      txt.box.addEventListener('input', sync);
      const ai = ro ? null : h('details', { class: 'stepai' }, h('summary', {}, ic('spark', 'sm'), ' Add this step with AI'), aiBox({ kind: 'step', title: 'Write this step', placeholder: 'e.g. ask whether they can decide alone or need a partner', context: () => ({ project: S.tenant, calltype: { name: d.name, objective: d.objective }, step: s, index: i }),
        current: () => ({ id: s.id, text: s.text, answer_type: s.answer_type, required: s.required }), apply: (p) => { Object.assign(s, p); drawSteps(); } }));
      const card = h('div', { class: 'stepcard', draggable: 'false',
        ondragstart: (e) => { dragI = i; e.dataTransfer.effectAllowed = 'move'; try { e.dataTransfer.setData('text/plain', String(i)); } catch {} },
        ondragover: (e) => { if (dragI != null) { e.preventDefault(); card.classList.add('over'); } }, ondragleave: () => card.classList.remove('over'),
        ondrop: (e) => { e.preventDefault(); if (dragI == null || dragI === i) return; const [m] = d.steps.splice(dragI, 1); d.steps.splice(i, 0, m); dragI = null; drawSteps(); },
        ondragend: () => { dragI = null; card.draggable = false; card.classList.remove('over'); } },
        h('div', { class: 'sc-h' }, ro ? null : h('button', { type: 'button', class: 'grip', title: 'Drag to reorder', 'aria-label': 'Drag to reorder step ' + (i + 1), onmousedown: () => { card.draggable = true; }, ontouchstart: () => { card.draggable = true; } }, ic('grip', 'sm')),
          h('span', { class: 'sn' }, 'Step ' + (i + 1)), id, at, h('label', { class: 'chip' }, req, 'required'), h('span', { style: 'flex:1' }),
          ro ? null : [h('button', { type: 'button', class: 'btn ghost sm', title: 'Move up', 'aria-label': 'Move step up', disabled: i === 0, onclick: () => { [d.steps[i - 1], d.steps[i]] = [d.steps[i], d.steps[i - 1]]; drawSteps(); } }, ic('up', 'sm')),
            h('button', { type: 'button', class: 'btn ghost sm', title: 'Move down', 'aria-label': 'Move step down', disabled: i === d.steps.length - 1, onclick: () => { [d.steps[i + 1], d.steps[i]] = [d.steps[i], d.steps[i + 1]]; drawSteps(); } }, ic('down', 'sm')),
            h('button', { type: 'button', class: 'btn ghost sm danger', title: 'Remove step', 'aria-label': 'Remove step', onclick: () => { d.steps.splice(i, 1); drawSteps(); } }, ic('trash', 'sm'))]),
        txt.box, ai);
      if (ro) card.querySelectorAll('textarea').forEach((t) => { t.disabled = true; });
      return card;
    }));
    if (!d.steps.length) stepsBox.append(h('div', { class: 'hint', style: 'padding:8px 0' }, 'No steps yet: the call follows the objective freely.'));
  }
  drawSteps();
  const collect = () => ({ ...d, name: f.name.value.trim(), objective: f.objective.value.trim(), tone: f.tone.value.trim(), max_call_seconds: num(f.max.value) || undefined, opening: opening.get(), closing: closing.get(), success_outcomes: lines(outcomes.value), compliance_lines: lines(comp.value),
    selling_allowed: sell.checked, negotiation_allowed: sell.checked && neg.checked, allowed_actions: [...actions], steps: d.steps.map((s, i) => ({ ...s, id: stepId(s, i), text: s.text || {}, answer_type: s.answer_type || 'text', required: !!s.required })), project_id: S.tenant });
  const apply = (p) => {
    if ('name' in p) f.name.value = p.name; if ('objective' in p) f.objective.value = p.objective; if ('tone' in p) f.tone.value = p.tone; if ('max_call_seconds' in p) f.max.value = p.max_call_seconds;
    if (p.opening) opening.set(p.opening); if (p.closing) closing.set(p.closing); if (p.success_outcomes) outcomes.value = [].concat(p.success_outcomes).join('\n'); if (p.compliance_lines) comp.value = [].concat(p.compliance_lines).join('\n');
    if ('selling_allowed' in p) sell.checked = !!p.selling_allowed; if ('negotiation_allowed' in p) neg.checked = !!p.negotiation_allowed;
    if (p.allowed_actions) { p.allowed_actions.forEach((a) => actions.add(a)); [...actBox.querySelectorAll('input')].forEach((cb) => { cb.checked = actions.has(cb.parentNode.textContent.trim().replaceAll(' ', '_')); }); }
    if (Array.isArray(p.steps)) { d.steps = p.steps.map((s) => ({ id: s.id || '', text: typeof s.text === 'string' ? { en: s.text } : s.text || {}, answer_type: s.answer_type || 'text', required: !!s.required })); drawSteps(); }
  };
  const err = h('div', { class: 'err' });
  const save = async () => {
    const o = collect(); err.textContent = '';
    if (!o.name) { err.textContent = 'Give the call type a name.'; return; }
    const ids = o.steps.map((s) => s.id); if (new Set(ids).size !== ids.length) { err.textContent = 'Two steps share the same id. Ids must be unique.'; return; }
    if (isNew) delete o.id;  // the server derives the id from the name
    const r = await saveObj('/api/calltypes', isNew ? '' : d.id, o);
    if (!r.ok) { err.textContent = r.na ? 'Saving call types is not available on this server yet.' : r.err; return; }
    _cts = null; toast('Saved.');
    const saved = r.data.calltype || r.data, ret = ctx.q.get('return'); if (ret === 'onboarding') { const st = obLoad(S.tenant); st.calltypeId = saved.id || o.id; await obSave(st); go('onboarding'); } else go('calltypes');
  };
  const dupBtn = isNew ? null : h('button', { class: 'btn', onclick: () => dupCT(base) }, ic('dup', 'sm'), 'Duplicate');
  const section = (t, ...kids) => h('div', { class: 'panel', style: 'margin-bottom:16px' }, h('div', { class: 'panel-head' }, h('b', {}, t)), h('div', { style: 'padding:4px 18px 16px' }, kids));
  el.replaceChildren(h('div', { class: 'pad narrow' },
    head(isNew ? 'New call type' : d.name || 'Call type', ro ? 'This is a built-in call type. Duplicate it to change anything.' : 'Changes apply to the next call that uses it.', dupBtn,
      !isNew && !ro ? h('button', { class: 'btn danger', onclick: () => confirmBox('Delete this call type?', 'This cannot be undone. A call type that a campaign still uses cannot be deleted: change that campaign first.', 'Delete', async () => { const r = await tryApi(`/api/calltypes/${esc(d.id)}`, { method: 'DELETE' }); if (!r.ok) return notice(r.err); _cts = null; go('calltypes'); }, { danger: true }) }, ic('trash', 'sm'), 'Delete') : null,
      ro ? null : h('button', { class: 'btn pri', onclick: save }, ic('save', 'sm'), 'Save')),
    ro ? null : aiBox({ kind: 'calltype', keys: ['name', 'objective', 'opening', 'steps', 'allowed_actions', 'success_outcomes', 'closing', 'tone', 'max_call_seconds', 'selling_allowed', 'negotiation_allowed', 'compliance_lines'], placeholder: 'e.g. a follow-up call for people who got the payment link but did not pay', context: () => ({ project: S.tenant, calltype: collect() }), current: collect, apply }),
    ctx.q.get('ai') && isNew ? h('div', { class: 'hint', style: 'margin:-6px 0 14px' }, 'Describe the call above, press Draft with AI, apply it and adjust.') : null,
    section('Purpose', fld('Name', f.name), fld('Objective: what counts as success', f.objective), h('div', { class: 'grid2' }, fld('Tone', f.tone), fld('Maximum call length (seconds)', f.max))),
    section('Opening line · per language', opening.box),
    section('Steps and questions', h('p', { class: 'hint', style: 'margin:8px 0' }, 'Asked in this order. Drag a step by its handle (or use the arrows) to reorder. Answered questions are never asked again.'), stepsBox, ro ? null : h('button', { class: 'btn sm', style: 'margin-top:10px', onclick: () => { d.steps.push({ id: '', text: {}, answer_type: 'text', required: false }); drawSteps(); } }, ic('plus', 'sm'), 'Add step')),
    section('What it may do', h('label', { class: 'lbl' }, 'Allowed actions'), actBox, h('label', { class: 'lbl' }, 'Outcomes that count as success'), outcomes,
      h('div', { class: 'row', style: 'gap:22px;margin-top:12px;flex-wrap:wrap' }, h('label', { class: 'chip' }, sell, 'selling allowed'), h('label', { class: 'chip' }, neg, 'negotiation allowed (needs selling)'))),
    section('Closing lines · per language', closing.box), section('Compliance lines', comp), err,
    h('div', { class: 'row', style: 'margin:8px 0 40px' }, ro ? null : h('button', { class: 'btn pri', onclick: save }, ic('save', 'sm'), 'Save'), h('a', { class: 'btn ghost', href: '#/calltypes' }, 'Back to call types'))));
}

/* ------------------------------------------------------------ CSV import (column mapping, preview, server check, commit) */
function parseCSV(text) {
  const rows = []; let row = [], cell = '', q = false;
  const t = String(text).replace(/\r\n?/g, '\n');
  for (let i = 0; i < t.length; i++) {
    const c = t[i];
    if (q) { if (c === '"') { if (t[i + 1] === '"') { cell += '"'; i++; } else q = false; } else cell += c; }
    else if (c === '"') q = true; else if (c === ',' || c === '\t' || c === ';') { row.push(cell); cell = ''; }
    else if (c === '\n') { row.push(cell); rows.push(row); row = []; cell = ''; } else cell += c;
  }
  if (cell !== '' || row.length) { row.push(cell); rows.push(row); }
  return rows.filter((r) => r.some((x) => x.trim() !== ''));
}
const e164 = (v) => { const d = String(v || '').replace(/[^\d+]/g, ''); if (/^\+\d{8,15}$/.test(d)) return d; const n = d.replace(/\D/g, ''); if (n.length === 10) return '+91' + n; if (n.length === 12 && n.startsWith('91')) return '+' + n; return ''; };
const GUESS = { name: /^(full[_ ]?)?name$|customer/i, phone: /phone|mobile|number|contact|whatsapp/i, email: /mail/i, language: /lang/i, city: /city|town|location/i, tags: /tag|label/i };
function importer(onDone, startText) {
  const st = { rows: [], map: {}, fields: [], headers: [], useHeader: true };
  const box = h('div', { class: 'importer' }), body = h('div'), msg = h('div', { class: 'hint' });
  const file = h('input', { type: 'file', accept: '.csv,.tsv,.txt,text/csv', 'aria-label': 'Choose a CSV file', onchange: () => { const f = file.files[0]; if (!f) return; const rd = new FileReader(); rd.onload = () => load(String(rd.result)); rd.readAsText(f); } });
  const paste = h('textarea', { rows: 4, placeholder: 'Or paste a list, one per line:\nAsha Rao, 98765 43210\nRahul, +91 99999 11111', 'aria-label': 'Paste a customer list' }, startText || '');
  const useBtn = h('button', { class: 'btn sm', type: 'button', onclick: () => load(paste.value, true) }, 'Use pasted list');
  box.append(fld('CSV file', file, 'First row can be column names. Phone numbers are normalised to +91… and checked against Do-not-call.'), fld('Paste a list', paste), h('div', { class: 'row', style: 'margin-top:8px' }, useBtn), body, msg);
  fieldDefs().then((fs) => { st.fields = fs; });
  function load(text, pasted) {
    const rows = parseCSV(text); if (!rows.length) { msg.textContent = 'Nothing to read yet.'; return; }
    const first = rows[0], looksHeader = !first.some((c) => /\d{8,}/.test(c.replace(/\D/g, '')) && c.replace(/\D/g, '').length >= 8);
    st.useHeader = looksHeader; st.headers = looksHeader ? first.map((x) => x.trim()) : first.map((_, i) => 'Column ' + (i + 1)); st.rows = looksHeader ? rows.slice(1) : rows;
    st.map = {};
    st.headers.forEach((hd, i) => { for (const [k, re] of Object.entries(GUESS)) if (re.test(hd) && !Object.values(st.map).includes(i) && st.map[k] == null) st.map[k] = i; });
    if (st.map.phone == null) { const pi = st.headers.findIndex((_, i) => st.rows.slice(0, 5).every((r) => e164(r[i]))); if (pi >= 0) st.map.phone = pi; }
    if (st.map.name == null) { const ni = st.headers.findIndex((_, i) => i !== st.map.phone); if (ni >= 0 && st.headers.length > 1) st.map.name = ni; }
    drawMap();
  }
  const targets = () => [['', '(ignore)'], ['name', 'Name'], ['phone', 'Phone'], ['email', 'Email'], ['language', 'Language'], ['city', 'City'], ['tags', 'Tags'], ...st.fields.map((f) => ['cf:' + f.key, 'Field: ' + (f.label || f.key)])];
  const colTarget = (i) => Object.entries(st.map).find(([, v]) => v === i)?.[0] || '';
  function drawMap() {
    const ok = st.rows.filter((r) => st.map.phone != null && e164(r[st.map.phone]));
    body.replaceChildren(h('div', { class: 'sec-lbl', style: 'margin:16px 0 6px' }, 'Match your columns'),
      h('div', { class: 'tablewrap' }, h('table', { class: 'tbl' }, h('thead', {}, h('tr', {}, st.headers.map((hd, i) => h('th', {}, h('div', { class: 'hint' }, hd), h('select', { 'aria-label': 'Field for ' + hd, onchange: (e) => { for (const k of Object.keys(st.map)) if (st.map[k] === i) delete st.map[k]; if (e.target.value) { for (const k of Object.keys(st.map)) if (k === e.target.value) delete st.map[k]; st.map[e.target.value] = i; } drawMap(); } }, targets().map(([v, n]) => h('option', { value: v, selected: colTarget(i) === v }, n))))))),
        h('tbody', {}, st.rows.slice(0, 5).map((r) => h('tr', {}, st.headers.map((_, i) => h('td', { class: colTarget(i) === 'phone' && !e164(r[i]) ? 'bad' : '' }, colTarget(i) === 'phone' ? (e164(r[i]) || r[i] + ' ?') : r[i] || ''))))))),
      h('div', { class: 'hint', style: 'margin-top:6px' }, `${st.rows.length} rows · ${ok.length} with a usable phone number${ok.length < st.rows.length ? ' · the rest will be skipped' : ''}. Showing the first 5.`),
      h('div', { class: 'row', style: 'margin-top:10px' }, h('button', { class: 'btn', type: 'button', disabled: st.map.phone == null, onclick: () => check(false) }, ic('check', 'sm'), 'Check before importing'), st.map.phone == null ? h('span', { class: 'err', style: 'margin:0' }, 'Pick which column is the phone number.') : null));
    msg.replaceChildren();
  }
  /* The server takes the CSV as text plus a {field: column name} mapping; custom fields are "custom.<key>". */
  const toCSV = (rows) => rows.map((r) => r.map((c) => (/[",\n]/.test(c) ? '"' + String(c).replace(/"/g, '""') + '"' : String(c))).join(',')).join('\n');
  const heads = () => { const seen = {}; return st.headers.map((x, i) => { let n = (x || '').trim() || 'Column ' + (i + 1); if (seen[n]) n += ' (' + (i + 1) + ')'; seen[n] = 1; return n; }); };
  const payload = (commit) => {
    const hs = heads(), mapping = {};
    for (const [k, i] of Object.entries(st.map)) mapping[k.startsWith('cf:') ? 'custom.' + k.slice(3) : k] = hs[i];
    return { project: S.tenant, csv: toCSV([hs, ...st.rows.map((r) => hs.map((_, i) => r[i] ?? ''))]), mapping, commit, on_duplicate: 'skip', tags: [] };
  };
  async function check(commit) {
    msg.textContent = commit ? 'Importing…' : 'Checking…';
    const r = await post('/api/customers/import', payload(commit));
    if (!r.ok) { msg.replaceChildren(r.na ? h('div', { class: 'empty na' }, h('b', {}, 'Import is not available yet'), 'The server does not have the customers import API yet. Your list is not lost: pick it again once it is.') : h('div', { class: 'err' }, r.err)); return; }
    const d = r.data, n = d.counts || {};
    if (commit) { const added = (n.new || 0) + (n.updated || 0); msg.replaceChildren(h('div', { class: 'tag green' }, `Imported ${added} customers`)); onDone && onDone(added); return; }
    const bad = (d.rows || []).filter((x) => x.status === 'invalid');
    msg.replaceChildren(h('div', { class: 'panel', style: 'padding:12px 16px;margin-top:10px' }, h('div', { class: 'row', style: 'flex-wrap:wrap;gap:6px' },
      h('span', { class: 'tag green' }, (n.new || 0) + ' will be added'), h('span', { class: 'tag' }, (n.duplicate || 0) + ' duplicates (same phone) skipped'), h('span', { class: 'tag red' }, (n.dnc || 0) + ' on Do-not-call'), h('span', { class: 'tag amber' }, (n.invalid || 0) + ' invalid')),
      bad.length ? h('ul', { class: 'pts' }, bad.slice(0, 5).map((x) => h('li', {}, 'Row ' + x.row + ': ' + x.error))) : null,
      h('div', { class: 'row', style: 'margin-top:10px' }, h('button', { class: 'btn pri', type: 'button', onclick: () => check(true) }, ic('upload', 'sm'), 'Import now'))));
  }
  if (startText) load(startText, true);
  return box;
}
function importModal(done) { modal('Import customers', 'Add many people at once. Nothing is saved until you press Import now.', (b, f, close) => { b.append(importer((n) => { done && done(n); })); f.append(h('button', { class: 'btn ghost', onclick: close }, 'Close')); }); }

/* ------------------------------------------------------------ Customers & leads */
const C = { view: store.get('sg_cview', 'table'), q: '', stage: '', tag: '', page: 1, sel: new Set(), data: null };
function custForm(c, fields, done) {
  const base = clone(c) || { custom: {}, tags: [], language: 'hinglish', stage: 'new' };
  modal(c ? 'Edit ' + (c.name || c.phone) : 'Add a customer', 'Phone numbers are stored as +91…', (b, f, close) => {
    const I = (v, a) => h('input', { value: v ?? '', ...(a || {}) }), err = h('div', { class: 'err' });
    const n = I(base.name), p = I(base.phone, { placeholder: '+91 98765 43210', inputmode: 'tel' }), e = I(base.email, { type: 'email' }), ci = I(base.city), t = I((base.tags || []).join(', '), { placeholder: 'comma separated' });
    const l = h('select', {}, LANGS.map(([k, v]) => h('option', { value: k, selected: base.language === k }, v)));
    const cf = {}; const cfBox = fields.map((fd) => fld(fd.label || fd.key, cf[fd.key] = fd.type === 'choice' ? h('select', {}, h('option', { value: '' }, '—'), (fd.choices || []).map((o) => h('option', { value: o, selected: (base.custom || {})[fd.key] === o }, o))) : I((base.custom || {})[fd.key], { type: fd.type === 'number' ? 'number' : fd.type === 'date' ? 'date' : 'text' })));
    b.append(h('div', { class: 'grid2' }, fld('Name', n), fld('Phone', p)), h('div', { class: 'grid2' }, fld('Email', e), fld('City', ci)), h('div', { class: 'grid2' }, fld('Language', l), fld('Tags', t)), cfBox, err);
    f.append(h('button', { class: 'btn pri', onclick: async () => {
      const ph = e164(p.value); if (!ph) { err.textContent = 'That phone number does not look right. Use 10 digits or +country code.'; return; }
      const custom = {}; for (const [k, el] of Object.entries(cf)) if (String(el.value).trim() !== '') custom[k] = el.value;
      const o = { name: n.value.trim(), phone: ph, email: e.value.trim(), city: ci.value.trim(), language: l.value, tags: csv(t.value), custom, ...(c ? {} : { project_id: S.tenant }) };
      const r = await saveObj('/api/customers', c ? c.id : '', o);
      if (!r.ok) { err.textContent = r.na ? 'Saving customers is not available on this server yet.' : r.err; return; }
      close(); done && done(r.data.customer || r.data);
    } }, ic('save', 'sm'), 'Save'), h('button', { class: 'btn ghost', onclick: close }, 'Cancel'));
  });
}
async function pageCustomers(el, ctx) {
  if (ctx.id) return pageCustomer(el, ctx);
  const draw = async () => {
    const board = C.view === 'board';
    const r = await tryApi('/api/customers?' + qs({ project: S.tenant, q: C.q, stage: board ? '' : C.stage, tag: C.tag, page: board ? 1 : C.page, page_size: board ? 200 : 50 }));
    if (!ctx.alive()) return;
    C.data = r; const list = r.ok ? arr(r.data, 'customers') : [], total = r.ok ? (r.data.total ?? list.length) : 0;
    const pages = r.ok ? Math.max(1, Math.ceil(total / (r.data.page_size || 50))) : 1;
    const bar = h('div', { class: 'filters' },
      h('input', { type: 'search', placeholder: 'Search name or phone', value: C.q, 'aria-label': 'Search customers', style: 'max-width:240px', onkeydown: (e) => { if (e.key === 'Enter') { C.q = e.target.value.trim(); C.page = 1; draw(); } } }),
      board ? null : h('select', { class: 'inline', 'aria-label': 'Filter by stage', onchange: (e) => { C.stage = e.target.value; C.page = 1; draw(); } }, h('option', { value: '' }, 'All stages'), STAGES.map((s) => h('option', { value: s, selected: C.stage === s }, stageName(s)))),
      h('input', { placeholder: 'Tag', value: C.tag, 'aria-label': 'Filter by tag', style: 'max-width:130px', onkeydown: (e) => { if (e.key === 'Enter') { C.tag = e.target.value.trim(); C.page = 1; draw(); } } }),
      h('span', { class: 'hint' }, r.ok ? total + ' people' : ''), h('span', { style: 'flex:1' }),
      h('div', { class: 'seg', role: 'group', 'aria-label': 'View' }, [['table', 'Table', 'table'], ['board', 'Stages', 'board']].map(([v, n, i]) => h('button', { class: C.view === v ? 'on' : '', 'aria-pressed': String(C.view === v), onclick: () => { C.view = v; store.set('sg_cview', v); C.sel.clear(); draw(); } }, ic(i, 'sm'), n))));
    let body;
    if (!r.ok) body = naBox('Customers & Leads', r, draw);
    else if (!list.length) body = h('div', { class: 'empty' }, h('b', {}, C.q || C.stage || C.tag ? 'No one matches those filters' : 'No customers yet'), 'Add one by hand, import a CSV, or point your lead form at the webhook link (Guided setup, step 4).');
    else body = board ? kanban(list) : custTable(list, pages);
    el.replaceChildren(h('div', { class: 'pad' }, head('Customers & Leads', 'Everyone the agent may call, with where they are in the pipeline.',
      h('button', { class: 'btn', onclick: () => importModal(() => draw()) }, ic('upload', 'sm'), 'Import CSV'), h('button', { class: 'btn pri', onclick: async () => custForm(null, await fieldDefs(), () => draw()) }, ic('plus', 'sm'), 'Add customer')), bar, body));
  };
  const kanban = (list) => {
    const cols = STAGES.map((s) => [s, list.filter((c) => (c.stage || 'new') === s)]);
    const move = async (c, to) => {
      if ((c.stage || 'new') === to) return;
      if (isDnc(c)) { notice('This person is on the Do-not-call list and stays there: only they can ask to be called again.'); draw(); return; }
      const go2 = async () => { const r = await post(`/api/customers/${esc(c.id)}`, to === 'do_not_call' ? { consent: { dnc: true, reason: 'moved to Do not call by the operator' } } : { stage: to }, 'PUT'); if (!r.ok) notice(r.err); draw(); };
      if (to === 'do_not_call') confirmBox('Mark ' + (c.name || c.phone) + ' as Do not call?', 'They will never be called by any campaign. This cannot be undone from here.', 'Mark Do not call', go2, { danger: true });
      else await go2();
    };
    return h('div', { class: 'board' }, cols.map(([s, cs]) => h('section', { class: 'col', 'aria-label': stageName(s),
      ondragover: (e) => { e.preventDefault(); e.currentTarget.classList.add('over'); }, ondragleave: (e) => e.currentTarget.classList.remove('over'),
      ondrop: (e) => { e.preventDefault(); e.currentTarget.classList.remove('over'); const id = e.dataTransfer.getData('text/plain'); const c = list.find((x) => x.id === id); if (c) move(c, s); } },
      h('div', { class: 'col-h' }, h('span', { class: 'tag ' + (STAGE_TONE[s] || '') }, stageName(s)), h('span', { class: 'hint' }, cs.length)),
      cs.map((c) => h('div', { class: 'kcard' + (isDnc(c) ? ' locked' : ''), draggable: isDnc(c) ? 'false' : 'true', ondragstart: (e) => { e.dataTransfer.setData('text/plain', c.id); e.dataTransfer.effectAllowed = 'move'; } },
        h('a', { href: '#/customers/' + esc(c.id), class: 'kn' }, c.name || c.phone), h('div', { class: 'hint' }, c.phone + (c.score != null ? ' · score ' + c.score : '')),
        h('select', { class: 'inline mv', disabled: isDnc(c), 'aria-label': 'Move ' + (c.name || c.phone) + ' to stage', onchange: (e) => move(c, e.target.value) }, STAGES.map((x) => h('option', { value: x, selected: x === (c.stage || 'new') }, stageName(x)))))))));
  };
  const custTable = (list, pages) => {
    const all = h('input', { type: 'checkbox', 'aria-label': 'Select all', style: 'width:auto;height:auto', checked: list.every((c) => C.sel.has(c.id)), onchange: (e) => { list.forEach((c) => e.target.checked ? C.sel.add(c.id) : C.sel.delete(c.id)); draw(); } });
    return h('div', {}, C.sel.size ? bulkBar(list, draw) : null,
      h('div', { class: 'tablewrap' }, h('table', { class: 'tbl' }, h('thead', {}, h('tr', {}, h('th', { style: 'width:30px' }, all), ['Name', 'Phone', 'Stage', 'Score', 'Tags', 'Next action', 'Last call'].map((t) => h('th', {}, t)))),
        h('tbody', {}, list.map((c) => h('tr', {}, h('td', {}, h('input', { type: 'checkbox', style: 'width:auto;height:auto', 'aria-label': 'Select ' + (c.name || c.phone), checked: C.sel.has(c.id), onchange: (e) => { e.target.checked ? C.sel.add(c.id) : C.sel.delete(c.id); draw(); } })),
          h('td', {}, h('a', { href: '#/customers/' + esc(c.id), class: 'lnk' }, c.name || '(no name)'), isDnc(c) ? h('span', { class: 'tag red', style: 'margin-left:6px' }, 'DNC') : null), h('td', { class: 'mono' }, c.phone),
          h('td', {}, h('span', { class: 'tag ' + (STAGE_TONE[c.stage || 'new'] || '') }, stageName(c.stage || 'new'))), h('td', {}, c.score ?? '—'),
          h('td', {}, (c.tags || []).slice(0, 3).map((t) => h('span', { class: 'tag', style: 'margin-right:4px' }, t))), h('td', { class: 'hint' }, c.next_action ? `${(c.next_action.type || '').replaceAll('_', ' ')} ${c.next_action.due_at ? '· ' + when(c.next_action.due_at) : ''}` : '—'), h('td', { class: 'hint' }, c.last_contact_at ? when(c.last_contact_at) : '—'))))),
      pages > 1 ? h('div', { class: 'row', style: 'justify-content:center;margin-top:14px' }, h('button', { class: 'btn sm', disabled: C.page <= 1, onclick: () => { C.page--; draw(); } }, 'Previous'), h('span', { class: 'hint' }, `Page ${C.page} of ${pages}`), h('button', { class: 'btn sm', disabled: C.page >= pages, onclick: () => { C.page++; draw(); } }, 'Next')) : null));
  };
  el.replaceChildren(h('div', { class: 'pad' }, h('div', { class: 'empty' }, 'Loading…')));
  await draw();
}
function bulkBar(list, redraw) {
  const picked = () => list.filter((c) => C.sel.has(c.id));
  const each = async (fn, label) => { let ok = 0, bad = 0; for (const c of picked()) { const r = await fn(c); r.ok ? ok++ : bad++; } C.sel.clear(); toast(`${label}: ${ok} done${bad ? ', ' + bad + ' failed' : ''}`); redraw(); };
  return h('div', { class: 'bulk', role: 'toolbar', 'aria-label': 'Bulk actions' }, h('b', {}, C.sel.size + ' selected'),
    h('button', { class: 'btn sm', onclick: () => modal('Tag ' + C.sel.size + ' people', null, (b, f, close) => { const t = h('input', { placeholder: 'tag, another tag' }); b.append(fld('Tags to add', t)); f.append(h('button', { class: 'btn pri', onclick: async () => { const add = csv(t.value); close(); await each((c) => post(`/api/customers/${esc(c.id)}`, { tags: [...new Set([...(c.tags || []), ...add])] }, 'PUT'), 'Tagged'); } }, 'Add tags'), h('button', { class: 'btn ghost', onclick: close }, 'Cancel')); }) }, ic('tag', 'sm'), 'Tag'),
    h('button', { class: 'btn sm', onclick: async () => { const cs = await campaigns(); modal('Add to a campaign', null, (b, f, close) => { if (!cs.length) { b.append(h('div', { class: 'hint' }, _camps && _camps.na ? 'Campaigns are not available yet.' : 'Create a campaign first.')); f.append(h('button', { class: 'btn', onclick: close }, 'Close')); return; } const s = h('select', {}, cs.map((c) => h('option', { value: c.id }, c.name))); b.append(fld('Campaign', s)); f.append(h('button', { class: 'btn pri', onclick: async () => { const camp = cs.find((c) => c.id === s.value), ids = picked().map((c) => c.id); close(); const aud = { ...(camp.audience || {}), customer_ids: [...new Set([...((camp.audience || {}).customer_ids || []), ...ids])] }; const r = await post(`/api/campaigns/${esc(camp.id)}`, { audience: aud }, 'PUT'); C.sel.clear(); r.ok ? toast(ids.length + ' added to ' + camp.name) : notice(r.err); _camps = null; redraw(); } }, 'Add'), h('button', { class: 'btn ghost', onclick: close }, 'Cancel')); }); } }, ic('mega', 'sm'), 'Add to campaign'),
    h('button', { class: 'btn sm danger', onclick: () => confirmBox('Mark ' + C.sel.size + ' as Do not call?', 'They will never be called by any campaign. This cannot be undone from here.', 'Mark Do not call', () => each((c) => post(`/api/customers/${esc(c.id)}`, { consent: { dnc: true, reason: 'marked by the operator' } }, 'PUT'), 'Marked'), { danger: true }) }, ic('ban', 'sm'), 'Mark DNC'),
    h('button', { class: 'btn ghost sm', onclick: () => { C.sel.clear(); redraw(); } }, 'Clear'));
}

/* ------------------------------------------------------------ Customer detail */
async function pageCustomer(el, ctx) {
  const id = ctx.id;
  el.replaceChildren(h('div', { class: 'pad' }, h('div', { class: 'empty' }, 'Loading…')));
  const [r, tl, fields, cts, cams] = await Promise.all([tryApi(`/api/customers/${esc(id)}`), tryApi(`/api/customers/${esc(id)}/timeline`), fieldDefs(), calltypes(), campaigns()]);
  if (!ctx.alive()) return;
  if (!r.ok) { el.replaceChildren(h('div', { class: 'pad' }, naBox('This customer', r, () => pageCustomer(el, ctx)))); return; }
  let c = r.data.customer || r.data; setCrumb(c.name || c.phone);
  const events = tl.ok ? arr(tl.data, 'timeline') : [];
  const upd = async (patch) => { const x = await post(`/api/customers/${esc(id)}`, patch, 'PUT'); if (!x.ok) return notice(x.err); toast('Saved.'); pageCustomer(el, ctx); };
  const notes = h('textarea', { rows: 4, placeholder: 'Private notes for your team' }, c.notes || c.owner_notes || '');
  const ctSel = h('select', { class: 'inline', 'aria-label': 'Call type' }, h('option', { value: '' }, 'Default call type'), ctOptions(cts).map(([v, n]) => h('option', { value: v, selected: v === (c.next_action || {}).call_type }, n)));
  const camSel = h('select', { class: 'inline', 'aria-label': 'Campaign' }, h('option', { value: '' }, 'No campaign (single call)'), cams.map((x) => h('option', { value: x.id, selected: x.id === (c.next_action || {}).campaign_id }, x.name)));
  const prodSel = h('select', { class: 'inline', 'aria-label': 'Product' }, SG.tenantProducts().map((p) => h('option', { value: p.id }, p.product_name)));
  const na = c.next_action || {}, hist = c.history || {};
  const kv = (k, v) => [h('dt', {}, k), h('dd', {}, v == null || v === '' ? '—' : v)];
  el.replaceChildren(h('div', { class: 'pad' },
    head(c.name || c.phone, c.phone + (c.email ? ' · ' + c.email : ''), h('button', { class: 'btn', onclick: () => custForm(c, fields, () => pageCustomer(el, ctx)) }, 'Edit'),
      isDnc(c) ? h('span', { class: 'tag red' }, 'Do not call') : [ctSel, camSel, SG.tenantProducts().length > 1 ? prodSel : null, h('button', { class: 'btn pri', onclick: () => { const ct = ctSel.value; callNow(c, { callType: ct, callTypeName: ct ? ctSel.selectedOptions[0].textContent : '', campaignId: camSel.value, productId: prodSel.value || undefined }); } }, ic('phone', 'sm'), 'Call now')]),
    h('div', { class: 'cols3' },
      h('div', {},
        h('div', { class: 'panel' }, h('div', { class: 'panel-head' }, h('b', {}, 'Pipeline'), h('span', { class: 'tag ' + (STAGE_TONE[c.stage || 'new'] || ''), style: 'margin-left:auto' }, stageName(c.stage || 'new'))),
          h('div', { class: 'stagebar', role: 'group', 'aria-label': 'Stage' }, STAGES.filter((s) => s !== 'do_not_call').map((s) => h('button', { class: 'stg' + ((c.stage || 'new') === s ? ' on' : ''), disabled: isDnc(c), 'aria-pressed': String((c.stage || 'new') === s), onclick: () => upd({ stage: s }) }, stageName(s)))),
          h('dl', { class: 'kv' }, ...kv('score', c.score != null ? c.score + ' / 100' : null), ...kv('next action', na.type ? `${na.type.replaceAll('_', ' ')}${na.due_at ? ' · ' + when(na.due_at) : ''}${na.call_type ? ' · ' + na.call_type : ''}` : null),
            ...kv('agreed price', hist.agreed_price), ...kv('objections', (hist.objections || []).join(', ')), ...kv('likes', (hist.likes || []).join(', ')), ...kv('source', c.source))),
        h('div', { class: 'panel' }, h('div', { class: 'panel-head' }, h('b', {}, 'Profile')), h('dl', { class: 'kv' }, ...kv('language', c.language), ...kv('city', c.city), ...kv('tags', (c.tags || []).map((t) => h('span', { class: 'tag', style: 'margin-right:4px' }, t))),
          ...fields.map((fd) => kv(fd.label || fd.key, (c.custom || {})[fd.key])).flat())),
        h('div', { class: 'panel' }, h('div', { class: 'panel-head' }, h('b', {}, 'Notes')), h('div', { style: 'padding:12px 16px' }, notes, h('div', { class: 'row', style: 'margin-top:8px' }, h('button', { class: 'btn sm', onclick: () => upd({ notes: notes.value }) }, ic('save', 'sm'), 'Save notes'))),
          h('div', { style: 'padding:0 16px 14px' }, isDnc(c) ? h('div', { class: 'hint' }, 'On the Do-not-call list' + ((c.consent || {}).reason ? ': ' + c.consent.reason : '') + '. Only the person asking us again can lift this, so there is no button for it.') : h('button', { class: 'btn sm danger', onclick: () => confirmBox('Mark as Do not call?', 'They will never be called by any campaign. This cannot be undone from here.', 'Mark Do not call', () => upd({ consent: { dnc: true, reason: 'marked by the operator' } }), { danger: true }) }, ic('ban', 'sm'), 'Mark Do not call')))),
      h('div', { class: 'panel' }, h('div', { class: 'panel-head' }, h('b', {}, 'Timeline')),
        !tl.ok ? naBox('The timeline', tl) : events.length ? h('div', { class: 'tline' }, events.map((e) => {
          const t = e.type || (e.call_id ? 'call' : 'note');
          return h('div', { class: 'tl-item ' + t }, h('div', { class: 'tl-d' }, when(e.t || e.at)),
            h('div', {}, h('b', {}, t === 'call' ? 'Call' + (e.call_type ? ' · ' + e.call_type.replaceAll('_', ' ') : '') : t === 'call_started' ? 'Call started' + (e.call_type ? ' · ' + e.call_type.replaceAll('_', ' ') : '') : t === 'stage' ? 'Stage: ' + stageName(e.from || '') + ' to ' + stageName(e.to || '') : t === 'created' ? 'Added' + (e.source ? ' · ' + e.source : '') : t === 'lead_again' ? 'Came in again · ' + (e.source || '') : 'Note'),
              e.outcome ? h('span', { class: 'tag ' + (/won|book|accept|agree|paid|link/i.test(e.outcome) ? 'green' : /lost|declin|opt|dnc/i.test(e.outcome) ? 'red' : 'amber'), style: 'margin-left:8px' }, e.outcome.replaceAll('_', ' ')) : null,
              e.summary || e.text ? h('div', { class: 'hint', style: 'margin-top:3px' }, e.summary || e.text) : null,
              e.call_id ? h('a', { class: 'btn sm', style: 'margin-top:6px', href: '#/calls/' + esc(e.call_id) }, ic('tree', 'sm'), 'Open call and info map') : null)); })) : h('div', { class: 'empty' }, 'No history yet.')))));
}

/* ------------------------------------------------------------ Campaigns */
const DAYS = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'];
const blankCamp = () => { const [a, b] = hoursOf(proj()); return { id: '', name: '', product_id: '', call_type_id: '', language: 'hinglish', tone_level: 'mid', status: 'draft', dial_mode: 'manual', audience: { stages: [], tags: [], languages: [], min_score: null, customer_ids: [] },
  schedule: { window: [a, b], days: [0, 1, 2, 3, 4, 5], max_parallel: 1, pacing_seconds: 60, retry: { max_attempts: 3, retry_after_minutes: 1440, retry_on: ['no_answer', 'voicemail', 'failed'] } } }; };
const statusTone = { draft: '', running: 'green', paused: 'amber', done: 'blue' };
async function pageCampaigns(el, ctx) {
  if (ctx.id) return campaignPage(el, ctx);
  el.replaceChildren(h('div', { class: 'pad' }, h('div', { class: 'empty' }, 'Loading…')));
  const list = await campaigns(true); if (!ctx.alive()) return;
  const bad = _camps && !_camps.ok;
  el.replaceChildren(h('div', { class: 'pad' }, head('Campaigns', 'A campaign is who to call, with which call type and product, and when. Dial mode is manual by default: nothing is dialled until you press Call.', h('button', { class: 'btn pri', onclick: () => go('campaigns', { id: 'new' }) }, ic('plus', 'sm'), 'New campaign')),
    bad ? naBox('Campaigns', { na: _camps.na, err: 'The campaigns API did not answer.' }, () => pageCampaigns(el, ctx)) : list.length ? h('div', { class: 'tablewrap' }, h('table', { class: 'tbl' }, h('thead', {}, h('tr', {}, ['Campaign', 'Status', 'Product', 'Call type', 'Dial mode', 'Progress', ''].map((t) => h('th', {}, t)))),
      h('tbody', {}, list.map((c) => { const st = c.stats || {}; return h('tr', {}, h('td', {}, h('a', { class: 'lnk', href: '#/campaigns/' + esc(c.id) }, c.name)), h('td', {}, h('span', { class: 'tag ' + (statusTone[c.status] || '') }, c.status || 'draft')), h('td', {}, SG.pname(c.product_id)), h('td', {}, c.call_type_id || '—'),
        h('td', {}, h('span', { class: 'tag ' + (c.dial_mode === 'auto' ? 'amber' : '') }, c.dial_mode || 'manual')), h('td', { class: 'hint' }, `${st.dialed ?? 0} dialled · ${st.calls ?? 0} finished`),
        h('td', {}, h('button', { class: 'btn sm', onclick: () => toggleCampaign(c, () => pageCampaigns(el, ctx)) }, c.status === 'running' ? [ic('pause', 'sm'), 'Pause'] : [ic('play', 'sm'), 'Start']))); })))) : h('div', { class: 'empty' }, h('b', {}, 'No campaigns yet'), 'Create one to group people, a call type and a schedule.')));
}
function toggleCampaign(c, done) {
  const running = c.status === 'running', auto = c.dial_mode === 'auto';
  confirmBox(running ? 'Pause ' + c.name + '?' : 'Start ' + c.name + '?',
    running ? 'No new calls will be placed. Calls already in progress finish normally.' : auto ? `Dial mode is AUTO: once started, the system places calls by itself inside the calling window (${(c.schedule && c.schedule.window) ? c.schedule.window.map(hh).join(' to ') : hoursOf(proj()).map(hh).join(' to ')}), skipping anyone on Do-not-call. Be sure the audience and the call type are right.` : 'Dial mode is MANUAL: starting only opens the queue. Nothing is dialled until you press Call next to a person in the queue.',
    running ? 'Pause' : 'Start', async () => { const r = await post(running ? `/api/campaigns/${esc(c.id)}/pause` : `/api/campaigns/${esc(c.id)}/start`); if (!r.ok) notice(r.na ? 'Starting campaigns is not available on this server yet.' : r.err); else toast(running ? 'Paused.' : 'Started.'); _camps = null; done && done(); }, auto && !running ? { type: 'START' } : {});
}
function campaignForm(base, o) {
  const B = blankCamp(), d = { ...B, ...clone(base) }; d.audience = { ...B.audience, ...d.audience }; d.schedule = { ...B.schedule, ...d.schedule }; d.schedule.retry = { ...B.schedule.retry, ...(d.schedule.retry || {}) };
  const win = d.schedule.window || hoursOf(proj());
  const I = (v, a) => h('input', { value: v ?? '', ...(a || {}) });
  const f = { name: I(d.name, { 'aria-label': 'Campaign name' }), product: h('select', { 'aria-label': 'Product' }, h('option', { value: '' }, 'Choose a product'), SG.tenantProducts().map((p) => h('option', { value: p.id, selected: p.id === d.product_id }, p.product_name))),
    ct: h('select', { 'aria-label': 'Call type' }, ctOptions(o.cts).map(([v, n]) => h('option', { value: v, selected: v === d.call_type_id }, n))), lang: h('select', { 'aria-label': 'Language' }, LANGS.map(([k, n]) => h('option', { value: k, selected: k === d.language }, n))),
    start: hourSel(win[0], 'Calling window starts', 0, 23), end: hourSel(win[1], 'Calling window ends', 1, 24), par: I(d.schedule.max_parallel, { type: 'number', min: 1, max: 20, 'aria-label': 'Calls at the same time' }), pace: I(d.schedule.pacing_seconds, { type: 'number', min: 0, 'aria-label': 'Seconds between calls' }),
    tries: I(d.schedule.retry.max_attempts, { type: 'number', min: 1, max: 10, 'aria-label': 'Tries per person' }), gap: I(Math.round(d.schedule.retry.retry_after_minutes / 60 * 10) / 10, { type: 'number', min: 0.1, step: 0.5, 'aria-label': 'Hours between tries' }),
    tone: toneField(d.tone_level), tags: I((d.audience.tags || []).join(', '), { 'aria-label': 'Tags' }), min: I(d.audience.min_score ?? '', { type: 'number', min: 0, max: 100, placeholder: 'optional', 'aria-label': 'Minimum score' }) };
  const stages = new Set(d.audience.stages || []), days = new Set(d.schedule.days || []);
  const chipset = (items, set, label) => h('div', { class: 'chips', role: 'group', 'aria-label': label }, items.map(([k, n]) => h('label', { class: 'chip' }, h('input', { type: 'checkbox', style: 'width:auto;height:auto', checked: set.has(k), onchange: (e) => { e.target.checked ? set.add(k) : set.delete(k); } }), n)));
  const auto = h('input', { type: 'checkbox', checked: d.dial_mode === 'auto', style: 'width:auto;height:auto', 'aria-label': 'Automatic dialling' });
  const collect = () => ({ name: f.name.value.trim(), product_id: f.product.value, call_type_id: f.ct.value, language: f.lang.value, tone_level: f.tone.sel.value, dial_mode: auto.checked ? 'auto' : 'manual', project_id: S.tenant,
    audience: { ...d.audience, stages: STAGES.filter((s) => stages.has(s)), tags: csv(f.tags.value).map((t) => t.toLowerCase()), min_score: f.min.value === '' ? null : num(f.min.value) },
    schedule: { ...d.schedule, window: [+f.start.value, +f.end.value], days: [0, 1, 2, 3, 4, 5, 6].filter((x) => days.has(x)), max_parallel: num(f.par.value) || 1, pacing_seconds: num(f.pace.value) ?? 0,
      retry: { ...d.schedule.retry, max_attempts: Math.max(1, Math.round(num(f.tries.value) || 1)), retry_after_minutes: Math.max(1, Math.round((num(f.gap.value) || 1) * 60)) } } });
  const apply = (p) => {
    if (p.name) f.name.value = p.name; if (p.product_id && [...f.product.options].some((x) => x.value === p.product_id)) f.product.value = p.product_id;
    if (p.call_type_id && [...f.ct.options].some((x) => x.value === p.call_type_id)) f.ct.value = p.call_type_id; if (p.language) f.lang.value = p.language; if (p.tone_level && TONES.some((t) => t[0] === p.tone_level)) { f.tone.sel.value = p.tone_level; f.tone.sel.dispatchEvent(new Event('change')); }
    const a = p.audience; if (a) { stages.clear(); (a.stages || []).forEach((s) => stages.add(s)); f.tags.value = (a.tags || []).join(', '); f.min.value = a.min_score ?? ''; d.audience = { ...d.audience, languages: a.languages || d.audience.languages }; chipsA.replaceWith(chipsA = chipset(STAGES.map((s) => [s, stageName(s)]), stages, 'Stages')); }
    const s = p.schedule; if (s) { if (s.window && s.window.length === 2) { f.start.value = s.window[0]; f.end.value = s.window[1]; } if (s.max_parallel) f.par.value = s.max_parallel; if (s.pacing_seconds != null) f.pace.value = s.pacing_seconds;
      if (s.retry) { if (s.retry.max_attempts) f.tries.value = s.retry.max_attempts; if (s.retry.retry_after_minutes) f.gap.value = Math.round(s.retry.retry_after_minutes / 60 * 10) / 10; }
      if (s.days) { days.clear(); s.days.forEach((x) => days.add(x)); chipsD.replaceWith(chipsD = chipset(DAYS.map((x, i) => [i, x]), days, 'Days')); } }
  };
  let chipsA = chipset(STAGES.map((s) => [s, stageName(s)]), stages, 'Stages'), chipsD = chipset(DAYS.map((x, i) => [i, x]), days, 'Days');
  const section = (t, ...k) => h('div', { class: 'panel', style: 'margin-bottom:16px' }, h('div', { class: 'panel-head' }, h('b', {}, t)), h('div', { style: 'padding:4px 18px 16px' }, k));
  const el = h('div', {}, aiBox({ kind: 'campaign', keys: ['name', 'call_type_id', 'language', 'audience', 'schedule'], placeholder: 'e.g. call everyone interested in the gym plan who has not answered twice, weekday evenings', context: () => ({ project: S.tenant, campaign: collect(), products: SG.tenantProducts().map((p) => ({ id: p.id, name: p.product_name })), calltypes: ctOptions(o.cts).map(([id, name]) => ({ id, name })) }), current: collect, apply }),
    section('What to call about', fld('Name', f.name), h('div', { class: 'grid2' }, fld('Product', f.product), fld('Call type', f.ct)), fld('Language', f.lang), f.tone.el),
    section('Audience · who gets called', h('label', { class: 'lbl' }, 'Stages (none ticked = everyone who is not won, lost or on Do-not-call)'), chipsA, h('div', { class: 'grid2' }, fld('Tags (any of)', f.tags), fld('Minimum score (0-100)', f.min)), (d.audience.customer_ids || []).length ? h('div', { class: 'hint', style: 'margin-top:8px' }, (d.audience.customer_ids || []).length + ' people were added by hand: only they (and who matches the filters above) are called.') : null),
    section('Schedule', h('div', { class: 'grid2' }, fld('Calling window starts', f.start), fld('Calling window ends', f.end)), h('label', { class: 'lbl' }, 'Days'), chipsD,
      h('div', { class: 'grid2' }, fld('Calls at the same time', f.par), fld('Seconds between calls', f.pace)), h('div', { class: 'grid2' }, fld('Tries per person when unanswered', f.tries), fld('Hours between tries', f.gap))),
    section('Dial mode', h('p', { class: 'hint', style: 'margin:8px 0' }, 'Manual (default): you press Call for each person from the queue. Auto places calls by itself inside the calling window; it needs your explicit confirmation when you save, and the server must also have automatic dialling switched on.'), h('label', { class: 'chip' }, auto, 'Switch this campaign to automatic dialling')));
  return { el, collect };
}
async function campaignPage(el, ctx) {
  const isNew = ctx.id === 'new', tab = ctx.q.get('tab') || 'settings';
  const [list, cts] = await Promise.all([campaigns(true), calltypes()]); if (!ctx.alive()) return;
  const base = isNew ? blankCamp() : list.find((c) => c.id === ctx.id);
  if (!base) { el.replaceChildren(h('div', { class: 'pad' }, naBox('This campaign', _camps && !_camps.ok ? { na: _camps.na, err: 'The campaigns API did not answer.' } : { err: 'Not found' }))); return; }
  setCrumb(base.name || 'New campaign');
  const form = campaignForm(base, { cts }), err = h('div', { class: 'err' });
  const doSave = async (o) => {
    const r = await saveObj('/api/campaigns', isNew ? '' : base.id, o);
    if (!r.ok) { err.textContent = r.na ? 'Saving campaigns is not available on this server yet.' : r.err; return; }
    _camps = null; toast('Saved.'); const saved = r.data.campaign || r.data;
    if (ctx.q.get('return') === 'onboarding') { const st = obLoad(S.tenant); st.campaignId = saved.id; await obSave(st); go('onboarding'); } else go('campaigns', { id: saved.id });
  };
  const save = async () => {
    const o = form.collect(); err.textContent = '';
    if (!o.name) { err.textContent = 'Give the campaign a name.'; return; } if (!o.product_id) { err.textContent = 'Pick a product.'; return; }
    if (o.dial_mode === 'auto' && base.dial_mode !== 'auto') return confirmBox('Switch to automatic dialling?', 'Once this campaign is started the system places calls by itself inside the calling window, skipping anyone on Do-not-call. You confirm this now and again when you start it.', 'Yes, allow automatic dialling', () => doSave({ ...o, confirm_auto: true }), { danger: true });
    await doSave(o);
  };
  const running = base.status === 'running';
  el.replaceChildren(h('div', { class: 'pad narrow' }, head(isNew ? 'New campaign' : base.name, isNew ? '' : null, h('span', { class: 'tag ' + (statusTone[base.status] || '') }, base.status || 'draft'),
    isNew ? null : h('button', { class: 'btn', onclick: () => toggleCampaign(base, () => campaignPage(el, ctx)) }, running ? [ic('pause', 'sm'), 'Pause'] : [ic('play', 'sm'), 'Start']),
    h('button', { class: 'btn pri', onclick: save }, ic('save', 'sm'), 'Save')),
    isNew ? null : tabs([['settings', 'Settings'], ['queue', 'Queue: who is next']], tab, (t) => go('campaigns', { id: base.id, query: { tab: t } })),
    tab === 'queue' && !isNew ? queueView(base) : [form.el, err, h('div', { class: 'row', style: 'margin:8px 0 40px' }, h('button', { class: 'btn pri', onclick: save }, ic('save', 'sm'), 'Save'), h('a', { class: 'btn ghost', href: '#/campaigns' }, 'Back to campaigns'))]));
}
function queueView(c) {
  const box = h('div', {}, h('div', { class: 'empty' }, 'Loading the queue…'));
  tryApi(`/api/campaigns/${esc(c.id)}/queue`).then((r) => {
    if (!r.ok) { box.replaceChildren(naBox('The queue', r)); return; }
    const d = r.data, q = d.eligible || [], w = d.window || {}, sk = (d.skipped || []).length;
    const win = (w.window ? hh(w.window[0]) + '–' + hh(w.window[1]) + ' ' + (w.timezone || '') : '') + (w.open ? ': open now' : w.next_open_at ? ': next opens ' + when(w.next_open_at) : '');
    box.replaceChildren(h('p', { class: 'hint', style: 'margin:12px 0' }, (c.dial_mode === 'auto' ? 'Automatic dialling: this is the order calls will be placed in.' : 'Manual dialling: nothing is placed until you press Call next to a person.') + (win ? ' Calling window ' + win + '.' : '') + (c.status === 'running' ? '' : ' The campaign is not running (' + (d.status || c.status) + ').')),
      q.length ? h('div', { class: 'tablewrap' }, h('table', { class: 'tbl' }, h('thead', {}, h('tr', {}, ['#', 'Who', 'Why they are here', 'Call type', ''].map((t) => h('th', {}, t)))), h('tbody', {}, q.map((x, i) => h('tr', {}, h('td', { class: 'mono' }, i + 1),
        h('td', {}, h('a', { class: 'lnk', href: '#/customers/' + esc(x.customer_id) }, x.name || x.phone), h('div', { class: 'hint' }, x.phone || '')), h('td', {}, x.reason || '—'), h('td', { class: 'hint' }, x.call_type || c.call_type_id || '—'),
        h('td', {}, h('button', { class: 'btn sm', onclick: () => callNow({ id: x.customer_id, name: x.name, phone: x.phone, language: x.language }, { campaignId: c.id, callType: x.call_type || c.call_type_id }) }, ic('phone', 'sm'), 'Call'))))))) : h('div', { class: 'empty' }, h('b', {}, 'Nobody is queued'), 'Add customers matching the audience, or adjust the audience.' + (sk ? ' ' + sk + ' people are held back (Do-not-call, stage, score or waiting for a retry).' : '')));
  });
  return box;
}

/* ------------------------------------------------------------ Follow-ups */
const FU = { due: 'today' };
const FU_TABS = [['now', 'Due now'], ['today', 'By end of today'], ['week', 'This week'], ['all', 'All scheduled']];
async function pageFollowups(el, ctx) {
  const load = async () => {
    el.replaceChildren(h('div', { class: 'pad' }, h('div', { class: 'empty' }, 'Loading…')));
    const r = await tryApi('/api/followups?' + qs({ project: S.tenant, due: FU.due })); if (!ctx.alive()) return;
    const list = r.ok ? arr(r.data, 'followups') : [];
    el.replaceChildren(h('div', { class: 'pad' }, head('Follow-ups', 'Calls the agent promised or that are worth making, from the outcome of earlier calls. One click starts the call.'),
      tabs(FU_TABS, FU.due, (t) => { FU.due = t; load(); }),
      !r.ok ? naBox('Follow-ups', r, load) : list.length ? h('div', { class: 'tablewrap' }, h('table', { class: 'tbl' }, h('thead', {}, h('tr', {}, ['Who', 'Due', 'Why', 'Call type', ''].map((t) => h('th', {}, t)))), h('tbody', {}, list.map((x) => {
        const cu = { id: x.customer_id, name: x.name, phone: x.phone, language: x.language }, na = x.next_action || {};
        return h('tr', {}, h('td', {}, h('a', { class: 'lnk', href: '#/customers/' + esc(cu.id) }, cu.name || cu.phone), h('div', { class: 'hint' }, cu.phone || '')), h('td', { class: x.overdue ? 'late' : '' }, when(x.due_at || na.due_at) + (na.due_estimated ? ' (estimated)' : '')), h('td', {}, na.note || (na.type || '').replaceAll('_', ' ') || '—'), h('td', { class: 'hint' }, na.call_type || 'follow_up'),
          h('td', {}, h('button', { class: 'btn pri sm', onclick: () => callNow(cu, { campaignId: na.campaign_id, callType: na.call_type || 'follow_up', callTypeName: 'a follow-up call' }) }, ic('phone', 'sm'), 'Start call')));
      })))) : h('div', { class: 'empty' }, h('b', {}, 'Nothing here'), 'Follow-ups appear here after calls where someone asked to be called back or a link was sent.')));
  };
  await load();
}

/* ------------------------------------------------------------ Settings */
async function pageSettings(el, ctx) {
  const tab = ctx.q.get('tab') || 'customization', p = proj();
  const body = h('div');
  el.replaceChildren(h('div', { class: 'pad narrow' }, head('Settings', 'Make the system fit how ' + p.name + ' works.'), tabs([['customization', 'Customization'], ['project', 'Project']], tab, (t) => go('settings', { query: { tab: t } })), body));
  const saveProject = async (patch, msg) => { const r = await saveObj('/api/projects', p.id, patch); if (!r.ok && !r.na) return notice(r.err); if (r.na && !S.projectsApi) { const r2 = await post('/api/tenants/' + esc(p.id), { name: patch.name || p.name, color: patch.color || p.color }, 'PUT'); if (!r2.ok) return notice(r2.err); } Object.assign(p, patch); toast(msg || 'Saved.'); if (r.na) toast('Saved on this screen only: the projects API is not available yet.'); };
  if (tab === 'project') return projectTab(body, p, saveProject);
  const [fields, cts] = [await fieldDefs(), await calltypes()]; if (!ctx.alive()) return;
  /* custom fields */
  const flist = h('div'), fs = clone(fields);
  const drawF = () => flist.replaceChildren(...fs.map((fd, i) => h('div', { class: 'frow' }, h('input', { value: fd.label || '', placeholder: 'Label, e.g. Budget', 'aria-label': 'Field label', oninput: (e) => { fd.label = e.target.value; if (!fd._saved) fd.key = slug(e.target.value).replaceAll('-', '_'); } }),
    h('select', { 'aria-label': 'Field type', onchange: (e) => { fd.type = e.target.value; drawF(); } }, ['text', 'number', 'date', 'choice'].map((t) => h('option', { value: t, selected: (fd.type || 'text') === t }, t))),
    h('select', { 'aria-label': 'Applies to', onchange: (e) => { fd.applies_to = e.target.value; } }, [['both', 'Customers and leads'], ['customer', 'Customers'], ['lead', 'Leads']].map(([v, n]) => h('option', { value: v, selected: (fd.applies_to || 'both') === v }, n))),
    fd.type === 'choice' ? h('input', { value: (fd.choices || []).join(', '), placeholder: 'options, comma separated', 'aria-label': 'Options', oninput: (e) => { fd.choices = csv(e.target.value); } }) : h('span', { class: 'hint mono' }, fd.key || ''),
    h('button', { class: 'btn ghost sm danger', 'aria-label': 'Remove field', onclick: () => { fs.splice(i, 1); drawF(); } }, ic('trash', 'sm')))));
  drawF();
  const saveFields = async () => {
    const keep = new Set(fs.map((x) => x.key)); let bad = null;
    for (const old of fields) if (!keep.has(old.key)) { const r = await tryApi(`/api/projects/${esc(p.id)}/fields/${esc(old.key)}`, { method: 'DELETE' }); if (!r.ok) bad = r; }
    const had = new Set(fields.map((x) => x.key));
    for (const fd of fs) { if (!fd.key) continue; const { _saved, ...o } = fd; const r = had.has(fd.key) ? await post(`/api/projects/${esc(p.id)}/fields/${esc(fd.key)}`, o, 'PUT') : await post(`/api/projects/${esc(p.id)}/fields`, o); if (!r.ok) bad = r; }
    bad ? notice(bad.na ? 'Custom fields are not available on this server yet.' : bad.err) : (toast('Fields saved.'), pageSettings(el, ctx));
  };
  const stageIn = {}, dc = { ...(p.default_calltypes || {}) };
  const sec = (t, d, ...k) => h('div', { class: 'panel', style: 'margin-bottom:16px' }, h('div', { class: 'panel-head' }, h('b', {}, t)), h('div', { style: 'padding:4px 18px 16px' }, d ? h('p', { class: 'hint', style: 'margin:8px 0' }, d) : null, k));
  const purposes = [['lead', 'New leads'], ['follow_up', 'Follow-ups'], ['payment', 'Payment reminders'], ['default', 'Everything else']];
  body.replaceChildren(
    sec('Custom fields', 'Extra details to keep for every customer and lead, such as budget, school, or vehicle. They show on the customer page and can be mapped when importing a CSV.', flist,
      h('div', { class: 'row', style: 'margin:10px 0' }, h('button', { class: 'btn sm', onclick: () => { fs.push({ key: '', label: '', type: 'text', applies_to: 'both' }); drawF(); } }, ic('plus', 'sm'), 'Add field'), h('button', { class: 'btn pri sm', onclick: saveFields }, ic('save', 'sm'), 'Save fields')),
      aiBox({ kind: 'customer_fields', title: 'Suggest fields with AI', placeholder: 'e.g. I sell second-hand cars, what should I record about each buyer?', context: () => ({ project: S.tenant, existing: fs }), current: () => ({ fields: fs }), apply: (d) => { (d.fields || []).forEach((x) => { const key = x.key || slug(x.label).replaceAll('-', '_'); if (key && !fs.some((y) => y.key === key)) fs.push({ ...x, key, type: x.type || 'text' }); }); drawF(); } })),
    sec('Stage names', 'Rename the pipeline stages to your own words. The order and meaning stay the same.', h('div', { class: 'grid2' }, STAGES.map((s) => fld(STAGE_LABEL[s], stageIn[s] = h('input', { value: (p.stages || {})[s] || '', placeholder: STAGE_LABEL[s] })))),
      h('div', { class: 'row', style: 'margin-top:10px' }, h('button', { class: 'btn pri sm', onclick: () => { const st = {}; STAGES.forEach((s) => { if (stageIn[s].value.trim()) st[s] = stageIn[s].value.trim(); }); saveProject({ stages: st }, 'Stage names saved.'); } }, ic('save', 'sm'), 'Save stage names'))),
    sec('Default call types', 'Which call type is used when you do not pick one yourself.', h('div', { class: 'grid2' }, purposes.map(([k, n]) => fld(n, h('select', { onchange: (e) => { dc[k] = e.target.value; } }, h('option', { value: '' }, 'Built-in default'), ctOptions(cts).map(([v, nm]) => h('option', { value: v, selected: dc[k] === v }, nm)))))),
      h('div', { class: 'row', style: 'margin-top:10px' }, h('button', { class: 'btn pri sm', onclick: () => saveProject({ default_calltypes: dc }, 'Defaults saved.') }, ic('save', 'sm'), 'Save defaults'))));
}
function projectTab(body, p, save) {
  const I = (v, a) => h('input', { value: v ?? '', ...(a || {}) });
  const tzs = (typeof Intl.supportedValuesOf === 'function' ? Intl.supportedValuesOf('timeZone') : ['Asia/Kolkata', 'UTC']);
  const f = { name: I(p.name), color: I(p.color || '#4f7cff', { type: 'color', style: 'width:64px;padding:2px' }), tz: h('select', {}, tzs.map((z) => h('option', { value: z, selected: z === (p.timezone || 'Asia/Kolkata') }, z))), s: hourSel(hoursOf(p)[0], 'Calling hours start', 0, 23), e: hourSel(hoursOf(p)[1], 'Calling hours end', 1, 24), notes: h('textarea', { rows: 3 }, p.compliance_notes || '') };
  const langs = new Set(p.languages || ['en', 'hi', 'hinglish']);
  const webhook = `${location.origin}/api/leads/webhook/${esc(p.id)}${p.webhook_token ? '?token=' + esc(p.webhook_token) : ''}`;
  body.replaceChildren(h('div', { class: 'panel' }, h('div', { class: 'panel-head' }, h('b', {}, 'Project basics')), h('div', { style: 'padding:4px 18px 16px' },
    h('div', { class: 'grid2' }, fld('Name', f.name), fld('Colour', f.color)), fld('Languages', h('div', { class: 'chips' }, LANGS.map(([k, n]) => h('label', { class: 'chip' }, h('input', { type: 'checkbox', style: 'width:auto;height:auto', checked: langs.has(k), onchange: (e) => { e.target.checked ? langs.add(k) : langs.delete(k); } }), n)))),
    fld('Time zone', f.tz), h('div', { class: 'grid2' }, fld('Calling hours start', f.s), fld('Calling hours end', f.e)), fld('Compliance notes', f.notes, 'Anything the agent must always respect for this business.'),
    fld('Lead webhook URL', h('div', { class: 'row' }, I(webhook, { readonly: true, class: 'mono', 'aria-label': 'Webhook URL' }), h('button', { class: 'btn sm', onclick: () => { navigator.clipboard && navigator.clipboard.writeText(webhook); toast('Copied.'); } }, ic('copy', 'sm'), 'Copy')), p.webhook_token ? 'POST a JSON lead (name, phone, …) to this link from your website or ad form.' : 'The server adds the secret token to this link once the CRM API is on. Keep the link private.'),
    h('div', { class: 'row', style: 'margin-top:14px' }, h('button', { class: 'btn pri', onclick: () => save({ name: f.name.value.trim() || p.name, color: f.color.value, languages: [...langs], timezone: f.tz.value, call_hours: [+f.s.value, +f.e.value], compliance_notes: f.notes.value }).then(() => SG.loadTenants().then(() => refresh())) }, ic('save', 'sm'), 'Save')))));
}

/* ------------------------------------------------------------ Onboarding wizard (resumable) */
const OB_STEPS = ['Project basics', 'Product', 'Call type', 'Customers', 'Campaign', 'Test call'];
async function pageOnboarding(el, ctx) {
  const fresh = ctx.q.get('new');
  if (fresh) { try { localStorage.removeItem(obKey('')); } catch {} PS.ob = null; location.replace('#/onboarding'); }
  const st = PS.ob && PS.ob.projectId === S.tenant ? PS.ob : (PS.ob = fresh ? obLoad('') : (S.tenants.length ? obLoad(S.tenant) : obLoad('')));
  if (!fresh && st.projectId === '' && S.tenant && !obLoad('').basics.name) { Object.assign(st, obLoad(S.tenant)); }
  const back = ctx.q.get('step'); if (back) st.step = Math.max(1, Math.min(7, +back));
  const draw = async () => {
    if (!ctx.alive()) return;
    await Promise.all([SG.loadProducts().catch(() => {}), calltypes(true), campaigns(true)]);
    const stepper = h('ol', { class: 'stepper', 'aria-label': 'Setup steps' }, OB_STEPS.concat(['Go live']).map((n, i) => h('li', { class: (st.step === i + 1 ? 'on ' : '') + (st.step > i + 1 || st.done ? 'done' : '') }, h('button', { disabled: i + 1 > st.step && !st.done, onclick: () => { st.step = i + 1; obSave(st); draw(); }, 'aria-current': st.step === i + 1 ? 'step' : null }, h('span', { class: 'sn2' }, st.step > i + 1 || st.done ? ic('check', 'sm') : String(i + 1)), h('span', { class: 'sl' }, n)))));
    const body = h('div', { class: 'panel obbody' });
    el.replaceChildren(h('div', { class: 'pad narrow' }, head('Set up a project', 'Six short steps, saved as you go. You can leave and come back: this page picks up where you stopped.', st.projectId ? h('a', { class: 'btn ghost sm', href: '#/dashboard' }, 'Leave for now') : null), stepper, body));
    const next = async (n) => { st.step = n; await obSave(st); draw(); };
    const nav = (canNext, onNext, label) => h('div', { class: 'df', style: 'position:static' }, st.step > 1 ? h('button', { class: 'btn ghost', onclick: () => next(st.step - 1) }, 'Back') : null, h('span', { style: 'flex:1' }), h('button', { class: 'btn pri', disabled: !canNext, onclick: onNext }, label || 'Next', ic('arrow', 'sm')));
    [null, ob1, ob2, ob3, ob4, ob5, ob6, ob7][st.step](body, st, { next, nav, draw, ctx });
  };
  await draw();
}
const sec2 = (t, d) => h('div', { class: 'dh' }, h('h3', {}, t), d ? h('p', {}, d) : null);
function ob1(body, st, o) {
  const p = S.tenants.find((t) => t.id === st.projectId), b = { name: '', languages: ['en', 'hi', 'hinglish'], timezone: 'Asia/Kolkata', start: 10, end: 20, ...(p ? { name: p.name, languages: p.languages || ['en', 'hi', 'hinglish'], timezone: p.timezone || 'Asia/Kolkata', start: hoursOf(p)[0], end: hoursOf(p)[1] } : {}), ...(st.basics || {}) };
  const tzs = typeof Intl.supportedValuesOf === 'function' ? Intl.supportedValuesOf('timeZone') : ['Asia/Kolkata', 'UTC'];
  const name = h('input', { value: b.name, placeholder: 'e.g. FitZone Gym', 'aria-label': 'Project name' }), tz = h('select', {}, tzs.map((z) => h('option', { value: z, selected: z === b.timezone }, z)));
  const s = hourSel(+b.start, 'Calling hours start', 0, 23), e = hourSel(+b.end, 'Calling hours end', 1, 24), langs = new Set(b.languages), err = h('div', { class: 'err' });
  body.append(sec2('1. Project basics', 'A project is one business or brand the agent calls for.'), h('div', { class: 'db' }, fld('Project name', name),
    fld('Languages the agent may use', h('div', { class: 'chips' }, LANGS.map(([k, n]) => h('label', { class: 'chip' }, h('input', { type: 'checkbox', style: 'width:auto;height:auto', checked: langs.has(k), onchange: (ev) => { ev.target.checked ? langs.add(k) : langs.delete(k); } }), n)))),
    fld('Time zone', tz), h('div', { class: 'grid2' }, fld('Calling hours start', s), fld('Calling hours end', e), ), h('div', { class: 'hint', style: 'margin-top:6px' }, 'The agent never calls outside these hours.'), err),
    o.nav(true, async () => {
      if (!name.value.trim()) { err.textContent = 'Name the project first.'; return; }
      const basics = { name: name.value.trim(), languages: [...langs], timezone: tz.value, call_hours: [+s.value, +e.value] };
      st.basics = { name: basics.name, languages: basics.languages, timezone: basics.timezone, start: +s.value, end: +e.value };
      if (!st.projectId) {
        const id = slug(basics.name) || 'project-' + Date.now().toString(36);
        let r = await saveObj('/api/projects', '', { id, color: '#4f7cff', ...basics });
        if (!r.ok && r.na) r = await post('/api/tenants/' + esc(id), { name: basics.name, color: '#4f7cff' }, 'PUT');
        if (!r.ok) { err.textContent = r.err || 'Could not create the project.'; return; }
        try { localStorage.removeItem(obKey('')); } catch {}
        st.projectId = id; await SG.loadTenants(); S.tenant = id; store.set('sg_tenant', id);
      } else { const r = await saveObj('/api/projects', st.projectId, basics); if (!r.ok && !r.na) { err.textContent = r.err; return; } }
      await SG.loadTenants(); o.next(2);
    }));
}
function ob2(body, st, o) {
  const ps = SG.tenantProducts(), name = h('input', { value: st.productName || '', placeholder: 'Product name only, e.g. ZenBrew Coffee Subscription', 'aria-label': 'Product name' });
  const pick = h('select', { 'aria-label': 'Existing product', onchange: (e) => { st.productId = e.target.value; obSave(st); o.draw(); } }, h('option', { value: '' }, ps.length ? 'Pick an existing product' : 'No products yet'), ps.map((x) => h('option', { value: x.id, selected: x.id === st.productId }, x.product_name)));
  body.append(sec2('2. Product', 'What will the agent talk about? Type just the name and the AI drafts the knowledge, offer and decision tree for you to review.'), h('div', { class: 'db' },
    fld('New product', name), h('div', { class: 'row', style: 'margin-top:8px;flex-wrap:wrap' }, h('button', { class: 'btn pri', onclick: () => { st.productName = name.value.trim(); st.step = 2; obSave(st); location.href = '/productgen?' + qs({ project: st.projectId, name: st.productName, return: '/#/onboarding?step=2' }); } }, ic('spark', 'sm'), 'Generate with AI'), h('button', { class: 'btn', onclick: () => SG.editProduct(null) }, 'Add by hand')),
    h('div', { class: 'sec-lbl', style: 'margin:18px 0 6px' }, 'Or use one you already have'), pick, st.productId ? h('div', { class: 'tag green', style: 'margin-top:8px' }, 'Using ' + SG.pname(st.productId)) : null),
    o.nav(!!(st.productId || ps.length), () => { if (!st.productId) st.productId = ps[0].id; o.next(3); }));
}
async function ob3(body, st, o) {
  const res = await calltypes(); const list = res.list.length ? res.list : BUILTIN_CT.map(([id, name]) => ({ id, name, builtin: true }));
  const draft = h('div');
  body.append(sec2('3. Call type', 'What kind of call is this? Pick one, change its steps, or describe it in your own words.'), h('div', { class: 'db' },
    h('div', { class: 'cards' }, list.map((c) => h('label', { class: 'pcard pick' + (st.calltypeId === c.id ? ' on' : '') }, h('input', { type: 'radio', name: 'ct', checked: st.calltypeId === c.id, onchange: () => { st.calltypeId = c.id; obSave(st); o.draw(); }, style: 'width:auto;height:auto;float:right' }),
      h('b', {}, c.name || c.id), h('p', { class: 'hint', style: 'margin:6px 0' }, c.objective || (c.builtin ? 'Built in' : 'Custom')), h('button', { class: 'btn sm', onclick: (e) => { e.preventDefault(); go('calltypes', { id: c.id, query: { return: 'onboarding' } }); } }, c.builtin ? 'View steps' : 'Customise steps')))),
    h('div', { class: 'sec-lbl', style: 'margin:18px 0 6px' }, 'Or describe it'), draft),
    o.nav(!!st.calltypeId, () => o.next(4)));
  draft.append(aiBox({ kind: 'calltype', keys: ['name', 'objective', 'opening', 'steps', 'allowed_actions', 'success_outcomes', 'closing', 'tone', 'max_call_seconds', 'selling_allowed', 'negotiation_allowed', 'compliance_lines'], placeholder: 'e.g. call people who got the payment link but did not pay, remind them kindly and offer help', context: () => ({ project: S.tenant, calltype: blankCT() }), current: blankCT,
    apply: async (p) => { const ct = { ...blankCT(), ...p, project_id: S.tenant }; delete ct.id; ct.name = ct.name || 'Custom call'; ct.steps = (ct.steps || []).map((s, i) => ({ id: s.id || 'step_' + (i + 1), answer_type: 'text', required: false, ...s, text: typeof s.text === 'string' ? { en: s.text } : s.text || {} })); st.draftCT = ct; draft.append(h('div', { class: 'row', style: 'margin-top:10px' }, h('button', { class: 'btn pri', onclick: async () => { const r = await saveObj('/api/calltypes', '', st.draftCT); if (!r.ok) return notice(r.na ? 'Saving call types is not available yet.' : r.err); _cts = null; st.calltypeId = (r.data.calltype || r.data).id; delete st.draftCT; await obSave(st); o.draw(); } }, ic('save', 'sm'), 'Save as a new call type'), h('span', { class: 'hint' }, 'Review it first, or save and edit the steps afterwards.'))); } }));
}
function ob4(body, st, o) {
  let mode = st.custMode || 'csv'; const area = h('div');
  const p = proj(), hook = `${location.origin}/api/leads/webhook/${esc(S.tenant)}${p.webhook_token ? '?token=' + esc(p.webhook_token) : ''}`;
  const drawMode = () => {
    area.replaceChildren(mode === 'csv' ? importer((n) => { st.customers = (st.customers || 0) + (+n || 0); obSave(st); o.draw(); }) : mode === 'paste' ? importer((n) => { st.customers = (st.customers || 0) + (+n || 0); obSave(st); o.draw(); }, 'name, phone\n')
      : mode === 'manual' ? h('div', {}, h('button', { class: 'btn pri', onclick: async () => custForm(null, await fieldDefs(), () => { st.customers = (st.customers || 0) + 1; obSave(st); o.draw(); }) }, ic('plus', 'sm'), 'Add a customer'))
      : h('div', {}, h('p', { class: 'hint' }, 'Send leads from your website, ad form or CRM to this link as JSON (name, phone, optional fields). Keep it private.'), h('div', { class: 'row' }, h('input', { readonly: true, value: hook, class: 'mono', 'aria-label': 'Webhook URL' }), h('button', { class: 'btn sm', onclick: () => { navigator.clipboard && navigator.clipboard.writeText(hook); toast('Copied.'); } }, ic('copy', 'sm'), 'Copy')), p.webhook_token ? null : h('p', { class: 'hint' }, 'The secret token is added by the server when the CRM API is on.')));
  };
  body.append(sec2('4. Customers', 'Who might the agent call? Add a few now, you can add more any time.'), h('div', { class: 'db' }, tabs([['csv', 'CSV file'], ['paste', 'Paste a list'], ['manual', 'One by one'], ['webhook', 'Lead webhook']], mode, (t) => { st.custMode = t; mode = t; o.draw(); }), area, st.customers ? h('div', { class: 'tag green', style: 'margin-top:12px' }, st.customers + ' added in this setup') : null),
    o.nav(true, () => o.next(5), st.customers ? 'Next' : 'Skip for now'));
  drawMode();
}
async function ob5(body, st, o) {
  const [list, cts] = await Promise.all([campaigns(true), calltypes()]); const mine = st.campaignId ? list.find((c) => c.id === st.campaignId) : null;
  const form = campaignForm(mine || { ...blankCamp(), name: proj().name + ' first campaign', product_id: st.productId, call_type_id: st.calltypeId }, { cts }), err = h('div', { class: 'err' });
  body.append(sec2('5. Campaign', 'Who to call, when, with which call type and product. Dial mode stays manual: nothing is called until you decide.'), h('div', { class: 'db' }, form.el, err),
    o.nav(true, async () => {
      const c = form.collect(); if (!c.name) { err.textContent = 'Name the campaign.'; return; } if (!c.product_id) { err.textContent = 'Pick a product.'; return; }
      c.dial_mode = 'manual';
      const r = await saveObj('/api/campaigns', st.campaignId || '', c);
      if (!r.ok) { if (r.na) { err.textContent = 'Campaigns are not available on this server yet. You can skip this step and come back.'; return; } err.textContent = r.err; return; }
      st.campaignId = (r.data.campaign || r.data).id; _camps = null; st.productId = c.product_id; st.calltypeId = c.call_type_id || st.calltypeId; o.next(6);
    }, 'Save and test'));
  const skip = h('button', { class: 'btn ghost sm', onclick: () => o.next(6) }, 'Skip this step'); body.querySelector('.df').insertBefore(skip, body.querySelector('.df').children[1] || null);
}
function ob6(body, st, o) {
  const T = st.sim = st.sim || { id: null, turns: [], busy: false }; T.turns = T.turns || [];
  const inp = h('input', { placeholder: T.id ? 'Say something as the customer' : 'Start the test first', disabled: !T.id, 'aria-label': 'Customer message', onkeydown: (e) => { if (e.key === 'Enter') say(); } });
  const err = h('div', { class: 'err' }), last = [...T.turns].reverse().find((t) => t.role === 'agent' && t.trace), map = T.map;
  const start = async () => {
    err.textContent = ''; T.turns = []; T.id = null; T.map = null; T.busy = true; o.draw();
    const r = await post('/api/sim/start', { persona: 'onboarding', campaign: st.productId, language: (st.basics.languages || ['hinglish']).includes('hinglish') ? 'hinglish' : 'en', backend: 'gemini', call_type: st.calltypeId || undefined, project_id: S.tenant });
    T.busy = false; if (!r.ok) { T.err = r.err; } else { T.id = r.data.sim_id; T.turns = [{ role: 'agent', text: r.data.agent, intent: 'opening' }]; T.err = ''; } o.draw();
  };
  const say = async () => {
    const text = inp.value.trim(); if (!text || !T.id || T.busy) return; T.busy = true; T.turns.push({ role: 'customer', text }); o.draw();
    const r = await post(`/api/sim/${T.id}`, { text }); T.busy = false;
    if (r.ok) { T.turns.push({ role: 'agent', ...r.data, text: r.data.agent }); if (r.data.map) T.map = r.data.map; } else T.turns.push({ role: 'agent', text: '⚠ ' + r.err }); st.tested = true; obSave(st); o.draw();
  };
  body.append(sec2('6. Test call', 'Talk to the agent by text, exactly as it would on a call, without dialling anyone. Check the opening, the answers and the decision behind each reply.'), h('div', { class: 'db' },
    h('div', { class: 'row', style: 'margin-bottom:10px' }, h('button', { class: 'btn pri', disabled: T.busy || !st.productId, onclick: start }, ic('play', 'sm'), T.id ? 'Restart test' : 'Start test'), !st.productId ? h('span', { class: 'hint' }, 'Pick a product in step 2 first.') : null),
    T.err ? h('div', { class: 'err' }, T.err) : null,
    h('div', { class: 'obsim' }, h('div', { class: 'tx', style: 'padding:6px 0' }, T.turns.length ? T.turns.map((t) => SG.turnView(t, 'gemini')) : h('div', { class: 'empty' }, 'The conversation appears here.'), T.busy ? h('div', { class: 'hint' }, 'thinking…') : null),
      h('div', { class: 'obmap' }, h('div', { class: 'sec-lbl' }, 'Info map'), map ? SG.mapView(map) : last ? [h('div', { class: 'hint', style: 'margin:6px 0' }, 'Decision path of the last reply (a full info map is shown on real calls).'), SG.pathView(last)] : h('div', { class: 'hint' }, 'Appears after the first reply.'))),
    h('div', { class: 'composer', style: 'padding:12px 0 0' }, inp, h('button', { class: 'btn pri', disabled: !T.id || T.busy, onclick: say }, ic('send', 'sm'), 'Send')), err),
    o.nav(true, () => o.next(7), st.tested ? 'Next: go live' : 'Skip the test'));
  if (T.id && !T.busy) setTimeout(() => inp.focus(), 0);
}
function ob7(body, st, o) {
  const ck = h('input', { type: 'checkbox', style: 'width:auto;height:auto' }), btn = h('button', { class: 'btn pri', disabled: true, onclick: async () => {
    btn.disabled = true; let msg = 'Setup finished.';
    if (st.campaignId) { const r = await post(`/api/campaigns/${esc(st.campaignId)}/start`); msg = r.ok ? 'The campaign is live. Dial mode is manual: open its queue and press Call for each person.' : r.na ? 'Setup finished. Starting campaigns is not available on this server yet.' : 'Setup finished, but the campaign did not start: ' + r.err; }
    st.done = true; await obSave(st); _camps = null; toast(msg); go(st.campaignId ? 'campaigns' : 'dashboard', st.campaignId ? { id: st.campaignId, query: { tab: 'queue' } } : undefined);
  } }, ic('phone', 'sm'), 'Go live');
  ck.addEventListener('change', () => { btn.disabled = !ck.checked; });
  body.append(sec2('Go live', 'Nothing has been dialled so far. Check the summary, then confirm.'), h('div', { class: 'db' },
    h('dl', { class: 'kv', style: 'border:0;padding:6px 0' }, h('dt', {}, 'project'), h('dd', {}, proj().name), h('dt', {}, 'product'), h('dd', {}, st.productId ? SG.pname(st.productId) : '—'), h('dt', {}, 'call type'), h('dd', {}, st.calltypeId || '—'), h('dt', {}, 'customers'), h('dd', {}, String(st.customers || 0) + ' added in setup'), h('dt', {}, 'campaign'), h('dd', {}, st.campaignId || 'none yet'), h('dt', {}, 'test call'), h('dd', {}, st.tested ? 'done' : 'not run')),
    h('label', { class: 'chip', style: 'margin-top:12px' }, ck, 'I have checked the setup and I am ready to go live. Calls only happen when I press Call (manual dial mode).'), h('div', { class: 'row', style: 'margin-top:14px' }, btn)),
    h('div', { class: 'df', style: 'position:static' }, h('button', { class: 'btn ghost', onclick: () => o.next(6) }, 'Back')));
}

/* ------------------------------------------------------------ styles + page table */
const css = `
.gs{position:relative}
.side .switcher{padding:10px 12px 0;position:relative}
.side .switcher .org{border-color:hsl(var(--rule-strong));background:hsl(var(--panel))}
.side .switcher .menu.pm{left:12px;right:12px;bottom:auto;top:calc(100% + 2px)}
.nav-item{text-decoration:none}
.side-foot .nav-item{margin:0}
.crumbs a{color:hsl(var(--tertiary));text-decoration:none}.crumbs a:hover{color:hsl(var(--fg))}
.topbar{gap:10px}.burger{display:none}
.searchbox{position:relative;display:flex;align-items:center;gap:6px;width:min(340px,34vw);height:32px;padding:0 10px;border:1px solid hsl(var(--rule-strong));border-radius:999px;color:hsl(var(--tertiary))}
.searchbox input{border:0;height:28px;padding:0;background:transparent;border-radius:0}.searchbox:focus-within{border-color:hsl(var(--signal))}.searchbox input:focus-visible{outline:0}
.sres{position:absolute;top:36px;left:0;right:0;z-index:60;max-height:60vh;overflow:auto;padding:4px 0;border:1px solid hsl(var(--rule-strong));border-radius:10px;background:hsl(var(--panel));box-shadow:0 18px 50px hsl(220 20% 1%/.45)}
.sres a{display:block;padding:7px 12px;color:hsl(var(--fg));text-decoration:none;font-size:12.5px}.sres a:hover,.sres a:focus{background:hsl(var(--hover))}
.scrim{display:none}
.pad.narrow{max-width:900px}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:12px;margin-bottom:18px}
.tile{display:block;padding:16px 18px;border:1px solid hsl(var(--rule));border-radius:10px;background:hsl(var(--panel));text-decoration:none;color:inherit}.tile:hover{border-color:hsl(var(--rule-heavy))}
.tile .v{font-size:28px;font-weight:500;letter-spacing:-.03em;line-height:1.1}.tile .k{margin-top:4px;font-family:var(--mono);font-size:10px;letter-spacing:.08em;text-transform:uppercase;color:hsl(var(--tertiary))}
.cols2{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:16px;align-items:start}.cols2 .panel+.panel{margin-top:0}
.cols3{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1fr);gap:16px;align-items:start}.cols3>div>.panel+.panel{margin-top:16px}
.item2{display:flex;align-items:center;gap:10px;padding:10px 16px;border-bottom:1px solid hsl(var(--rule));color:hsl(var(--fg));text-decoration:none;font-size:12.5px}.item2:last-child{border-bottom:0}.item2:hover{background:hsl(var(--hover))}.item2 svg:last-child{margin-left:auto;color:hsl(var(--tertiary))}
.chk{width:18px;height:18px;border:1.5px solid hsl(var(--rule-heavy));border-radius:50%;display:grid;place-items:center;flex:none}.chk.ok{background:hsl(var(--green));border-color:hsl(var(--green));color:hsl(var(--bg))}.chk svg{width:11px;height:11px}
.ptabs{display:flex;gap:4px;margin:0 0 16px;border-bottom:1px solid hsl(var(--rule))}.ptabs button{height:34px;padding:0 14px;border:0;border-bottom:2px solid transparent;background:none;color:hsl(var(--muted));cursor:pointer;font-weight:500}.ptabs button.on{color:hsl(var(--fg));border-color:hsl(var(--signal))}
.tablewrap{overflow-x:auto;border:1px solid hsl(var(--rule));border-radius:10px;background:hsl(var(--panel))}
.tbl{width:100%;border-collapse:collapse;font-size:12.5px}.tbl th{padding:9px 12px;text-align:left;font-family:var(--mono);font-size:10px;letter-spacing:.08em;text-transform:uppercase;color:hsl(var(--tertiary));font-weight:500;border-bottom:1px solid hsl(var(--rule));vertical-align:bottom}
.tbl td{padding:9px 12px;border-bottom:1px solid hsl(var(--rule));vertical-align:middle}.tbl tr:last-child td{border-bottom:0}.tbl tbody tr:hover{background:hsl(var(--hover)/.5)}.tbl td.bad{color:hsl(var(--red))}.tbl td.late{color:hsl(var(--red))}
.lnk{color:hsl(var(--fg));font-weight:500;text-decoration:none}.lnk:hover{color:hsl(var(--signal))}
.filters{display:flex;align-items:center;gap:8px;flex-wrap:wrap;margin-bottom:14px}.filters input,.filters select{width:auto}
.seg{display:inline-flex;border:1px solid hsl(var(--rule-strong));border-radius:999px;overflow:hidden}.seg button{display:inline-flex;align-items:center;gap:6px;height:30px;padding:0 12px;border:0;background:none;color:hsl(var(--muted));cursor:pointer}.seg button.on{background:hsl(var(--elevated));color:hsl(var(--fg))}
.bulk{display:flex;align-items:center;gap:8px;flex-wrap:wrap;padding:8px 12px;margin-bottom:10px;border:1px solid hsl(var(--signal)/.5);border-radius:10px;background:hsl(var(--signal)/.07)}
.board{display:flex;gap:12px;overflow-x:auto;padding-bottom:12px;align-items:flex-start}.col{flex:0 0 230px;min-height:120px;border:1px solid hsl(var(--rule));border-radius:10px;background:hsl(var(--panel));padding:8px}.col.over{border-color:hsl(var(--signal))}
.col-h{display:flex;align-items:center;justify-content:space-between;padding:4px 4px 8px}
.kcard{padding:9px 10px;margin-bottom:8px;border:1px solid hsl(var(--rule-strong));border-radius:8px;background:hsl(var(--bg));cursor:grab}.kcard.locked{cursor:not-allowed;opacity:.75}.kcard .kn{font-weight:500;color:hsl(var(--fg));text-decoration:none}.kcard .mv{margin-top:6px;width:100%;height:26px;font-size:11.5px}
.stagebar{display:flex;flex-wrap:wrap;gap:6px;padding:12px 16px}.stg{height:26px;padding:0 10px;border:1px solid hsl(var(--rule-strong));border-radius:999px;background:none;color:hsl(var(--muted));cursor:pointer;font-size:11.5px}.stg.on{background:hsl(var(--signal));border-color:hsl(var(--signal));color:hsl(220 10% 6%)}
.tline{padding:6px 16px 14px}.tl-item{position:relative;padding:10px 0 10px 18px;border-left:2px solid hsl(var(--rule-strong));margin-left:6px}.tl-item::before{content:'';position:absolute;left:-6px;top:15px;width:10px;height:10px;border-radius:50%;background:hsl(var(--panel));border:2px solid hsl(var(--signal))}.tl-d{font-family:var(--mono);font-size:10px;color:hsl(var(--tertiary))}
.ai{margin:0 0 16px;padding:12px 14px;border:1px solid hsl(var(--violet)/.5);border-radius:10px;background:hsl(var(--violet)/.06)}.ai-h{display:flex;align-items:center;gap:8px;margin-bottom:8px;flex-wrap:wrap}.ai-h svg{color:hsl(var(--violet))}
.diff{margin-top:12px;border-top:1px solid hsl(var(--rule));padding-top:10px}.drow{display:grid;grid-template-columns:130px minmax(0,1fr);gap:10px;padding:6px 0;border-bottom:1px dashed hsl(var(--rule))}.dk{display:flex;align-items:flex-start;gap:6px;font-family:var(--mono);font-size:10.5px;color:hsl(var(--muted));text-transform:uppercase}
.dv{white-space:pre-wrap;font-size:12px;overflow-wrap:anywhere}.dv .old{color:hsl(var(--tertiary));text-decoration:line-through;margin-bottom:3px}.dv .new{color:hsl(var(--green))}
.langs{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:8px}
.chips{display:flex;flex-wrap:wrap;gap:6px}.chip{display:inline-flex;align-items:center;gap:6px;height:28px;padding:0 10px;border:1px solid hsl(var(--rule-strong));border-radius:999px;font-size:12px;cursor:pointer}
.stepcard{padding:12px 14px;margin-bottom:10px;border:1px solid hsl(var(--rule-strong));border-radius:10px;background:hsl(var(--bg))}.stepcard.over{border-color:hsl(var(--signal));box-shadow:0 0 0 3px hsl(var(--signal)/.18)}
.sc-h{display:flex;align-items:center;gap:8px;flex-wrap:wrap;margin-bottom:8px}.sc-h input,.sc-h select{width:auto;min-width:0;max-width:150px}.sn{font-family:var(--mono);font-size:10.5px;color:hsl(var(--signal))}
.grip{border:0;background:none;cursor:grab;color:hsl(var(--tertiary));padding:4px}.stepai{margin-top:8px}.stepai summary{cursor:pointer;color:hsl(var(--violet));font-size:12px}.stepai .ai{margin-top:8px}
.frow{display:grid;grid-template-columns:1.2fr 90px 1.2fr 1.4fr 32px;gap:8px;align-items:center;margin-bottom:8px}
.stepper{display:flex;gap:4px;list-style:none;margin:0 0 16px;padding:0;overflow-x:auto}.stepper li{flex:1;min-width:96px}.stepper button{display:flex;align-items:center;gap:8px;width:100%;height:36px;padding:0 10px;border:1px solid hsl(var(--rule-strong));border-radius:8px;background:none;color:hsl(var(--tertiary));cursor:pointer;font-size:12px}
.stepper li.on button{border-color:hsl(var(--signal));color:hsl(var(--fg));background:hsl(var(--signal)/.08)}.stepper li.done button{color:hsl(var(--muted))}.stepper button:disabled{cursor:default;opacity:.6}.sn2{display:grid;place-items:center;width:18px;height:18px;border-radius:50%;border:1px solid currentColor;font-size:10px;flex:none}.sn2 svg{width:11px;height:11px}
.obbody .db{padding:10px 20px 18px}.pcard.pick{cursor:pointer}.pcard.pick.on{box-shadow:inset 0 0 0 1px hsl(var(--signal)/.6)}
.obsim{display:grid;grid-template-columns:minmax(0,1.4fr) minmax(0,1fr);gap:16px}.obmap{border-left:1px solid hsl(var(--rule));padding-left:16px;min-width:0}
.na{border:1px dashed hsl(var(--rule-strong));border-radius:10px}.tone-ex{margin-top:4px}.tone-all{margin-top:6px;font-size:11px}.tone-all div{margin-top:2px}
.toast{position:fixed;left:50%;bottom:24px;transform:translateX(-50%);z-index:200;padding:9px 16px;border-radius:999px;background:hsl(var(--elevated));border:1px solid hsl(var(--rule-heavy));font-size:12.5px;box-shadow:0 10px 30px hsl(220 20% 1%/.4)}
.dh p{color:hsl(var(--body))}.obbody .dh{padding:16px 20px 12px;border-bottom:1px solid hsl(var(--rule))}.obbody .df{border-top:1px solid hsl(var(--rule));padding:12px 20px;display:flex;gap:8px}
@media (max-width:1100px){.cols3,.cols2,.obsim{grid-template-columns:1fr}.obmap{border-left:0;padding-left:0}.frow{grid-template-columns:1fr 1fr}}
@media (max-width:760px){.side{display:flex!important;position:fixed;left:0;top:0;z-index:90;width:260px;transform:translateX(-100%);transition:transform .2s}.navopen .side{transform:none}
.navopen .scrim{display:block;position:fixed;inset:0;z-index:80;background:hsl(220 20% 2%/.55)}.burger{display:grid}.searchbox{width:auto;flex:1}.hide-sm,.conn{display:none}.crumbs{display:none}.langs{grid-template-columns:1fr}.drow{grid-template-columns:1fr}.grid2{grid-template-columns:1fr}.head .acts{width:100%}}
`;
const st = document.createElement('style'); st.textContent = css; document.head.append(st);

const PAGES = { dashboard: pageDashboard, products: pageProducts, calltypes: pageCallTypes, customers: pageCustomers, campaigns: pageCampaigns, followups: pageFollowups, settings: pageSettings, onboarding: pageOnboarding };
return { go, render, renderPage, aiBox, toneField, parseCSV, e164 };
};

# Data Analysis Course Project

A single-page site with 5 tabs (Overview, Dataset, Methodology, Live Python Demo, Results & Conclusions), published with GitHub Pages. The **Live Python Demo** tab runs real Python in the browser via [Pyodide](https://pyodide.org) — no server required.

## Files

- `index.html` — page structure and all 5 tabs
- `style.css` — styling (light/dark mode aware)
- `script.js` — tab switching + Pyodide loading/auto-run logic
- `assets/` — put your images (e.g. charts) here

## Fill in your own content

Search the files for `TODO` comments — each marks a spot to replace placeholder text with your real project details, data, and code.

The Python code that runs in the **Live Python Demo** tab lives inside the
`<script type="text/plain" id="python-code">` block near the middle of
`index.html`. Replace its contents with your own analysis code. If you need
extra packages (pandas, numpy, matplotlib, etc.), uncomment/add a
`pyodide.loadPackage([...])` call in `script.js` (see the comment inside
`getPyodideInstance()`).

## Run locally before publishing

Browsers block some features (like Pyodide) when you open `index.html`
directly as a file, so serve it locally instead:

```bash
python3 -m http.server 8000
```

Then open http://localhost:8000 in your browser.

## Publish on GitHub Pages

See the step-by-step guide provided alongside this project, or:
GitHub repo → Settings → Pages → Source → Deploy from branch → `main` / `/root`.

// ============================================================
// TAB SWITCHING
// ============================================================
const tabButtons = document.querySelectorAll(".tab-btn");
const tabPanels = document.querySelectorAll(".tab-panel");

tabButtons.forEach((btn) => {
  btn.addEventListener("click", () => {
    const target = btn.dataset.tab;

    tabButtons.forEach((b) => {
      b.classList.toggle("active", b === btn);
      b.setAttribute("aria-selected", b === btn ? "true" : "false");
    });

    tabPanels.forEach((panel) => {
      panel.classList.toggle("active", panel.id === target);
    });

    // Auto-run the Python demo the first (and only the first) time
    // its tab is opened.
    if (target === "demo") {
      runPythonDemoOnce();
    }
  });
});

// ============================================================
// PYODIDE (runs real Python in the browser)
// ============================================================
let pyodideInstance = null;
let hasRunOnce = false;

const statusEl = document.getElementById("pyodide-status");
const outputEl = document.getElementById("python-output");
const rerunBtn = document.getElementById("rerun-btn");

async function getPyodideInstance() {
  if (pyodideInstance) return pyodideInstance;

  statusEl.textContent = "Loading Python runtime… (first load can take ~5-10s)";
  pyodideInstance = await loadPyodide();

  // Capture Python's print() output into the page instead of the browser console.
  pyodideInstance.setStdout({
    batched: (msg) => {
      outputEl.textContent += msg + "\n";
    },
  });
  pyodideInstance.setStderr({
    batched: (msg) => {
      outputEl.textContent += "[error] " + msg + "\n";
    },
  });

  // If your analysis code needs extra packages (pandas, numpy, matplotlib, etc.),
  // load them here BEFORE running your code. Example:
  //
  //   await pyodideInstance.loadPackage(["pandas", "numpy"]);
  //
  // Only load what you actually import — each package adds load time.

  return pyodideInstance;
}

async function runPythonCode() {
  const code = document.getElementById("python-code").textContent;
  outputEl.textContent = "";
  rerunBtn.disabled = true;
  statusEl.textContent = "Running…";

  try {
    const pyodide = await getPyodideInstance();
    await pyodide.runPythonAsync(code);
    statusEl.textContent = "Done";
  } catch (err) {
    outputEl.textContent += "\n[Python error]\n" + err;
    statusEl.textContent = "Error — see output";
  } finally {
    rerunBtn.disabled = false;
  }
}

function runPythonDemoOnce() {
  if (hasRunOnce) return;
  hasRunOnce = true;
  runPythonCode();
}

rerunBtn.addEventListener("click", runPythonCode);

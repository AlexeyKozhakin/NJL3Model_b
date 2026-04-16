document.addEventListener('DOMContentLoaded', () => {
    // -------------------------------------------------------------------------
    // Табы
    // -------------------------------------------------------------------------
    const tabs = {
        heatmap: document.getElementById('tab-heatmap'),
        slices: document.getElementById('tab-slices'),
        components: document.getElementById('tab-components'),
    };
    const panels = {
        heatmap: document.getElementById('panel-heatmap'),
        slices: document.getElementById('panel-slices'),
        components: document.getElementById('panel-components'),
    };

    function switchTab(name) {
        for (const k in tabs) {
            if (k === name) {
                tabs[k].classList.add('text-blue-600', 'border-blue-600');
                tabs[k].classList.remove('text-gray-500', 'border-transparent');
                panels[k].classList.remove('hidden');
            } else {
                tabs[k].classList.remove('text-blue-600', 'border-blue-600');
                tabs[k].classList.add('text-gray-500', 'border-transparent');
                panels[k].classList.add('hidden');
            }
        }
    }

    Object.keys(tabs).forEach(k => {
        tabs[k].addEventListener('click', () => switchTab(k));
    });

    // -------------------------------------------------------------------------
    // Элементы управления
    // -------------------------------------------------------------------------
    const inputs = {
        mu: document.getElementById('param-mu'),
        L: document.getElementById('param-L'),
        g: document.getElementById('param-g'),
        bMin: document.getElementById('param-bmin'),
        bMax: document.getElementById('param-bmax'),
        MMin: document.getElementById('param-Mmin'),
        MMax: document.getElementById('param-Mmax'),
        Nb: document.getElementById('param-Nb'),
        NM: document.getElementById('param-NM'),
    };

    const checkboxes = {
        omegaMuL: document.getElementById('show-omegaMuL'),
        dU: document.getElementById('show-dU'),
        omegaL: document.getElementById('show-omegaL'),
        classical: document.getElementById('show-classical'),
    };

    const btnCalc = document.getElementById('btn-calculate');
    const progressContainer = document.getElementById('progress-container');
    const progressBar = document.getElementById('progress-bar');
    const progressText = document.getElementById('progress-text');
    const resultInfo = document.getElementById('result-info');

    let currentGrid = null;
    let currentG = -1.0;
    let selectedB = null;
    let selectedM = null;

    function readParams() {
        return {
            mu: parseFloat(inputs.mu.value),
            L: parseFloat(inputs.L.value),
            g: parseFloat(inputs.g.value),
            bMin: parseFloat(inputs.bMin.value),
            bMax: parseFloat(inputs.bMax.value),
            MMin: parseFloat(inputs.MMin.value),
            MMax: parseFloat(inputs.MMax.value),
            N_b: parseInt(inputs.Nb.value),
            N_M: parseInt(inputs.NM.value),
            N_h_p1: 100,
            N_h_p2: 100,
            N_h_p: 100,
            N_h_phi: 100,
            N_h_mu: 100,
        };
    }

    function updateProgress(done, total) {
        const pct = Math.round((done / total) * 100);
        progressBar.style.width = pct + '%';
        progressText.textContent = `${done} / ${total}`;
    }

    function arrayTo2D(arr, N_b, N_M) {
        const z = [];
        for (let j = 0; j < N_M; j++) {
            const row = [];
            for (let i = 0; i < N_b; i++) {
                row.push(arr[i * N_M + j]);
            }
            z.push(row);
        }
        return z;
    }

    // -------------------------------------------------------------------------
    // Отрисовка графиков
    // -------------------------------------------------------------------------
    function renderHeatmap() {
        if (!currentGrid) return;
        const g = currentGrid;
        const z = arrayTo2D(g.total, g.b_vals.length, g.M_vals.length);
        const x = Array.from(g.b_vals);
        const y = Array.from(g.M_vals);

        const data = [{
            z: z, x: x, y: y,
            type: 'heatmap',
            colorscale: 'Viridis',
            colorbar: { title: 'Ω' },
            hovertemplate: 'b: %{x:.3f}<br>M: %{y:.3f}<br>Ω: %{z:.4f}<extra></extra>',
        }];

        const layout = {
            title: `Ω_total (μ=${inputs.mu.value}, L=${inputs.L.value}, g=${inputs.g.value})`,
            xaxis: { title: 'b' },
            yaxis: { title: 'M' },
            margin: { t: 40, r: 30, b: 50, l: 60 },
            annotations: [{
                x: g.b_min, y: g.M_min,
                xref: 'x', yref: 'y',
                text: '★',
                showarrow: false,
                font: { size: 24, color: 'white' },
            }],
        };

        Plotly.newPlot('plot-heatmap', data, layout, { responsive: true });

        const plotEl = document.getElementById('plot-heatmap');
        plotEl.on('plotly_click', (evt) => {
            const pt = evt.points[0];
            selectedB = pt.x;
            selectedM = pt.y;
            renderSlices();
            switchTab('slices');
        });
    }

    function renderComponents() {
        if (!currentGrid) return;
        const g = currentGrid;
        const N_b = g.b_vals.length;
        const N_M = g.M_vals.length;
        const x = Array.from(g.b_vals);
        const y = Array.from(g.M_vals);

        const comps = [
            { id: 'plot-comp-omegaMuL', z: arrayTo2D(g.omegaMuL, N_b, N_M), title: 'Ω_μL', cs: 'Plasma' },
            { id: 'plot-comp-dU', z: arrayTo2D(g.dU, N_b, N_M), title: 'dU', cs: 'Cividis' },
            { id: 'plot-comp-omegaL', z: arrayTo2D(g.omegaL, N_b, N_M), title: 'Ω_L^phys', cs: 'Inferno' },
        ];

        comps.forEach(c => {
            Plotly.newPlot(c.id, [{
                z: c.z, x, y, type: 'heatmap', colorscale: c.cs,
                colorbar: { title: c.title },
                hovertemplate: 'b: %{x:.3f}<br>M: %{y:.3f}<br>' + c.title + ': %{z:.4f}<extra></extra>',
            }], {
                title: c.title,
                xaxis: { title: 'b' }, yaxis: { title: 'M' },
                margin: { t: 40, r: 30, b: 50, l: 60 },
            }, { responsive: true });
        });

        // Классический вклад M^2/(2g)
        const classical = [];
        for (let j = 0; j < N_M; j++) {
            const row = [];
            const M = g.M_vals[j];
            const val = (M * M) / (2.0 * currentG);
            for (let i = 0; i < N_b; i++) row.push(val);
            classical.push(row);
        }
        Plotly.newPlot('plot-comp-classical', [{
            z: classical, x, y, type: 'heatmap', colorscale: 'Greys',
            colorbar: { title: 'M²/(2g)' },
        }], {
            title: 'M²/(2g)',
            xaxis: { title: 'b' }, yaxis: { title: 'M' },
            margin: { t: 40, r: 30, b: 50, l: 60 },
        }, { responsive: true });
    }

    function renderSlices() {
        if (!currentGrid) return;
        const g = currentGrid;
        const b = selectedB !== null ? selectedB : g.b_min;
        const M = selectedM !== null ? selectedM : g.M_min;

        const slice = NJL.extractSlice(g, b, M);
        const M_arr = Array.from(g.M_vals);
        const b_arr = Array.from(g.b_vals);

        const tracesB = [{ x: M_arr, y: Array.from(slice.slice_b), mode: 'lines', name: 'Ω_total', line: { width: 2, color: '#1f77b4' } }];
        const tracesM = [{ x: b_arr, y: Array.from(slice.slice_M), mode: 'lines', name: 'Ω_total', line: { width: 2, color: '#1f77b4' } }];

        function addComponent(arrB, arrM, name, color, dash) {
            tracesB.push({ x: M_arr, y: Array.from(arrB), mode: 'lines', name, line: { color, dash } });
            tracesM.push({ x: b_arr, y: Array.from(arrM), mode: 'lines', name, line: { color, dash } });
        }

        if (checkboxes.omegaMuL.checked) {
            addComponent(slice.slice_b_omegaMuL, slice.slice_M_omegaMuL, 'Ω_μL', '#ff7f0e', 'dash');
        }
        if (checkboxes.dU.checked) {
            addComponent(slice.slice_b_dU, slice.slice_M_dU, 'dU', '#2ca02c', 'dash');
        }
        if (checkboxes.omegaL.checked) {
            addComponent(slice.slice_b_omegaL, slice.slice_M_omegaL, 'Ω_L', '#d62728', 'dash');
        }
        if (checkboxes.classical.checked) {
            const classicalB = M_arr.map(m => (m * m) / (2.0 * currentG));
            const classicalM = b_arr.map(() => (slice.M_actual * slice.M_actual) / (2.0 * currentG));
            addComponent(new Float64Array(classicalB), new Float64Array(classicalM), 'M²/(2g)', '#9467bd', 'dash');
        }

        const layoutB = {
            title: `Срез при b ≈ ${slice.b_actual.toFixed(3)}`,
            xaxis: { title: 'M' },
            yaxis: { title: 'Ω' },
            margin: { t: 40, r: 20, b: 60, l: 60 },
            shapes: [{
                type: 'line', x0: slice.M_actual, x1: slice.M_actual,
                y0: 0, y1: 1, yref: 'paper', line: { color: 'gray', dash: 'dot', width: 2 }
            }],
            legend: { orientation: 'h', y: -0.25 },
        };

        const layoutM = {
            title: `Срез при M ≈ ${slice.M_actual.toFixed(3)}`,
            xaxis: { title: 'b' },
            yaxis: { title: 'Ω' },
            margin: { t: 40, r: 20, b: 60, l: 60 },
            shapes: [{
                type: 'line', x0: slice.b_actual, x1: slice.b_actual,
                y0: 0, y1: 1, yref: 'paper', line: { color: 'gray', dash: 'dot', width: 2 }
            }],
            legend: { orientation: 'h', y: -0.25 },
        };

        Plotly.newPlot('plot-slice-b', tracesB, layoutB, { responsive: true });
        Plotly.newPlot('plot-slice-M', tracesM, layoutM, { responsive: true });

        // Совмещённые срезы (два subplot рядом)
        const combinedData = [
            { x: M_arr, y: Array.from(slice.slice_b), xaxis: 'x', yaxis: 'y', type: 'scatter', mode: 'lines', name: 'Ω(M)' },
            { x: b_arr, y: Array.from(slice.slice_M), xaxis: 'x2', yaxis: 'y2', type: 'scatter', mode: 'lines', name: 'Ω(b)', line: { dash: 'dash' } },
        ];
        // Добавим отдельные компоненты на оба subplot
        if (checkboxes.omegaMuL.checked) {
            combinedData.push({ x: M_arr, y: Array.from(slice.slice_b_omegaMuL), xaxis: 'x', yaxis: 'y', type: 'scatter', mode: 'lines', name: 'Ω_μL(M)', line: { dash: 'dot' } });
            combinedData.push({ x: b_arr, y: Array.from(slice.slice_M_omegaMuL), xaxis: 'x2', yaxis: 'y2', type: 'scatter', mode: 'lines', name: 'Ω_μL(b)', line: { dash: 'dot' } });
        }

        const combinedLayout = {
            grid: { rows: 1, columns: 2, pattern: 'independent' },
            xaxis: { title: 'M', domain: [0, 0.45], anchor: 'y' },
            yaxis: { title: 'Ω', domain: [0, 1], anchor: 'x' },
            xaxis2: { title: 'b', domain: [0.55, 1], anchor: 'y2' },
            yaxis2: { title: 'Ω', domain: [0, 1], anchor: 'x2' },
            title: `Совмещённые срезы (b=${slice.b_actual.toFixed(2)}, M=${slice.M_actual.toFixed(2)})`,
            margin: { t: 40, r: 20, b: 50, l: 60 },
            legend: { orientation: 'h', y: -0.15 },
        };
        Plotly.newPlot('plot-slice-combined', combinedData, combinedLayout, { responsive: true });
    }

    // -------------------------------------------------------------------------
    // Запуск расчёта
    // -------------------------------------------------------------------------
    function runCalculation() {
        const params = readParams();
        if (isNaN(params.mu) || isNaN(params.L) || params.N_b < 2 || params.N_M < 2) {
            alert('Проверьте корректность параметров (N_b, N_M ≥ 2)');
            return;
        }

        currentG = params.g;
        btnCalc.disabled = true;
        btnCalc.textContent = 'Вычисление...';
        progressContainer.classList.remove('hidden');
        resultInfo.classList.add('hidden');
        progressBar.style.width = '0%';
        progressText.textContent = '0 / ' + params.N_b;

        setTimeout(() => {
            const t0 = performance.now();
            const grid = NJL.computeGrid(params, (done, total) => {
                if (done % 5 === 0 || done === total) {
                    updateProgress(done, total);
                }
            });
            const t1 = performance.now();
            console.log('Calculation took', ((t1 - t0) / 1000).toFixed(2), 's');

            currentGrid = grid;
            selectedB = grid.b_min;
            selectedM = grid.M_min;

            document.getElementById('res-bmin').textContent = grid.b_min.toFixed(4);
            document.getElementById('res-mmin').textContent = grid.M_min.toFixed(4);
            document.getElementById('res-omin').textContent = grid.Omega_min.toFixed(4);
            resultInfo.classList.remove('hidden');

            renderHeatmap();
            renderSlices();
            renderComponents();

            btnCalc.disabled = false;
            btnCalc.textContent = 'Рассчитать';
            progressContainer.classList.add('hidden');
        }, 50);
    }

    btnCalc.addEventListener('click', runCalculation);

    Object.values(checkboxes).forEach(cb => {
        cb.addEventListener('change', () => {
            if (currentGrid) renderSlices();
        });
    });

    // Первый запуск
    runCalculation();
});

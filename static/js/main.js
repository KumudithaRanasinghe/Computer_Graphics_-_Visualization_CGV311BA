/* SAMS AI Attendance Dashboard - Main JavaScript */

let currentSummaryData = null;
let processedResultsCache = {};
let trendChart = null;
let pieChart = null;

document.addEventListener('DOMContentLoaded', () => {
    initDragAndDrop();
    fetchSummary();
});

// Tab Switching
function switchTab(tabId) {
    document.querySelectorAll('.tab-btn').forEach(btn => btn.classList.remove('active'));
    document.querySelectorAll('.tab-panel').forEach(panel => panel.classList.remove('active'));

    const activeBtn = document.querySelector(`.tab-btn[onclick*="${tabId}"]`);
    const activePanel = document.getElementById(`tab-${tabId}`);

    if (activeBtn) activeBtn.classList.add('active');
    if (activePanel) activePanel.classList.add('active');

    if (tabId === 'summary') {
        fetchSummary();
    }
}

// Drag and Drop Logic
function initDragAndDrop() {
    const dropzone = document.getElementById('dropzone');
    if (!dropzone) return;

    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        dropzone.addEventListener(eventName, preventDefaults, false);
    });

    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    ['dragenter', 'dragover'].forEach(eventName => {
        dropzone.addEventListener(eventName, () => dropzone.classList.add('dragover'), false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropzone.addEventListener(eventName, () => dropzone.classList.remove('dragover'), false);
    });

    dropzone.addEventListener('drop', (e) => {
        const dt = e.dataTransfer;
        const files = dt.files;
        const input = document.getElementById('image-input');
        input.files = files;
        updateSelectedFilesList();
    });
}

function updateSelectedFilesList() {
    const input = document.getElementById('image-input');
    const container = document.getElementById('selected-files-list');
    container.innerHTML = '';

    if (!input.files || input.files.length === 0) return;

    Array.from(input.files).forEach(file => {
        const badge = document.createElement('div');
        badge.className = 'file-badge';
        badge.innerHTML = `
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z"/></svg>
            ${file.name} (${(file.size / 1024).toFixed(1)} KB)
        `;
        container.appendChild(badge);
    });
}

// Upload & Process Submission
async function handleUploadSubmit(event) {
    event.preventDefault();
    const imageInput = document.getElementById('image-input');
    
    if (!imageInput.files || imageInput.files.length === 0) {
        alert('Please select or drag at least one signing sheet image!');
        return;
    }

    const formData = new FormData(document.getElementById('upload-form'));

    showProcessingUI('Uploading & Running Computer Vision Pipeline...');

    try {
        const response = await fetch('/api/upload-and-process', {
            method: 'POST',
            body: formData
        });

        const data = await response.json();

        if (response.ok && data.status === 'success') {
            updateProgressBar(100);
            cacheProcessedResults(data.results);
            setTimeout(() => {
                hideProcessingUI();
                switchTab('summary');
            }, 600);
        } else {
            alert('Error processing images: ' + (data.error || 'Unknown error'));
            hideProcessingUI();
        }
    } catch (err) {
        alert('Failed to connect to server: ' + err.message);
        hideProcessingUI();
    }
}

// Load Pre-loaded Sample Sheets
async function loadSampleSheets() {
    showProcessingUI('Processing Pre-loaded Sample Signing Sheets (1.jpeg - 5.jpeg)...');

    try {
        const response = await fetch('/api/process-samples', {
            method: 'POST'
        });
        const data = await response.json();

        if (response.ok && data.status === 'success') {
            updateProgressBar(100);
            cacheProcessedResults(data.results);
            setTimeout(() => {
                hideProcessingUI();
                switchTab('summary');
            }, 600);
        } else {
            alert('Error processing samples: ' + (data.error || 'Unknown error'));
            hideProcessingUI();
        }
    } catch (err) {
        alert('Failed to process sample sheets: ' + err.message);
        hideProcessingUI();
    }
}

function cacheProcessedResults(results) {
    if (!results) return;
    results.forEach(res => {
        if (res.sheet_name) {
            processedResultsCache[res.sheet_name] = res;
        }
    });
}

// UI Processing Indicators
function showProcessingUI(titleText) {
    const card = document.getElementById('processing-card');
    const title = document.getElementById('processing-status-title');
    const fill = document.getElementById('progress-bar-fill');
    
    card.style.display = 'block';
    title.textContent = titleText;
    fill.style.width = '20%';

    let width = 20;
    window.processingInterval = setInterval(() => {
        if (width < 90) {
            width += (90 - width) * 0.1;
            fill.style.width = width + '%';
        }
    }, 300);
}

function updateProgressBar(percentage) {
    if (window.processingInterval) clearInterval(window.processingInterval);
    const fill = document.getElementById('progress-bar-fill');
    fill.style.width = percentage + '%';
}

function hideProcessingUI() {
    if (window.processingInterval) clearInterval(window.processingInterval);
    const card = document.getElementById('processing-card');
    card.style.display = 'none';
}

// Fetch & Display Attendance Summary
async function fetchSummary() {
    try {
        const response = await fetch('/api/summary');
        const data = await response.json();
        currentSummaryData = data;

        updateStatCards(data.overall);
        renderCharts(data.sessions, data.overall);
        renderStudentsTable(data.students);
        populateSheetSelector(data.sessions);

    } catch (err) {
        console.error('Error fetching summary:', err);
    }
}

function updateStatCards(overall) {
    if (!overall) return;
    document.getElementById('stat-sessions').textContent = overall.total_sessions || 0;
    document.getElementById('stat-students').textContent = overall.total_students || 0;
    document.getElementById('stat-rate').textContent = (overall.overall_percentage || 0) + '%';
    document.getElementById('stat-absents').textContent = overall.total_absents || 0;
}

// Render Chart.js Visualizations
function renderCharts(sessions, overall) {
    if (!sessions || sessions.length === 0) return;

    // 1. Session Trend Bar Chart
    const labels = sessions.map(s => `${s.sheet} (${s.date})`);
    const presentData = sessions.map(s => s.present);
    const absentData = sessions.map(s => s.absent);

    const ctxTrend = document.getElementById('trend-chart').getContext('2d');

    if (trendChart) trendChart.destroy();

    trendChart = new Chart(ctxTrend, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [
                {
                    label: 'Present Students',
                    data: presentData,
                    backgroundColor: '#10b981',
                    borderRadius: 6
                },
                {
                    label: 'Absent Students',
                    data: absentData,
                    backgroundColor: '#f43f5e',
                    borderRadius: 6
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { labels: { color: '#9ca3af', font: { family: 'Inter' } } }
            },
            scales: {
                x: { ticks: { color: '#9ca3af', font: { size: 10 } }, grid: { color: 'rgba(255,255,255,0.05)' } },
                y: { ticks: { color: '#9ca3af', stepSize: 1 }, grid: { color: 'rgba(255,255,255,0.05)' } }
            }
        }
    });

    // 2. Pie Chart
    const ctxPie = document.getElementById('pie-chart').getContext('2d');
    if (pieChart) pieChart.destroy();

    const presents = overall.total_presents || 0;
    const absents = overall.total_absents || 0;

    pieChart = new Chart(ctxPie, {
        type: 'doughnut',
        data: {
            labels: ['Present', 'Absent'],
            datasets: [{
                data: [presents, absents],
                backgroundColor: ['#10b981', '#f43f5e'],
                borderWidth: 0
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { position: 'bottom', labels: { color: '#9ca3af', font: { family: 'Inter' } } }
            },
            cutout: '70%'
        }
    });
}

// Student Table & Search Filter
function renderStudentsTable(students) {
    const tbody = document.getElementById('students-table-body');
    tbody.innerHTML = '';

    if (!students || students.length === 0) {
        tbody.innerHTML = '<tr><td colspan="6" style="text-align: center; color: var(--text-muted);">No attendance records in database yet. Process images to view summary.</td></tr>';
        return;
    }

    students.forEach(s => {
        const isEligible = s.percentage >= 75.0;
        const badgeClass = isEligible ? 'badge-success' : 'badge-danger';
        const tr = document.createElement('tr');
        
        tr.innerHTML = `
            <td style="font-weight: 700; font-family: 'Outfit', sans-serif;">${s.index}</td>
            <td>${s.name}</td>
            <td>${s.present_count} / ${s.total_sessions}</td>
            <td>
                <div style="display:flex; align-items:center; gap:0.5rem;">
                    <span>${s.percentage}%</span>
                    <div class="density-bar-container">
                        <div class="density-bar-fill" style="width: ${s.percentage}%; background: ${isEligible ? '#10b981' : '#f43f5e'};"></div>
                    </div>
                </div>
            </td>
            <td><span class="badge ${badgeClass}">${s.status}</span></td>
            <td>
                <button class="btn btn-secondary" style="padding: 0.3rem 0.75rem; font-size: 0.8rem;" onclick='openStudentModal(${JSON.stringify(s)})'>
                    History
                </button>
            </td>
        `;
        tbody.appendChild(tr);
    });
}

function filterStudentTable() {
    const query = document.getElementById('student-search').value.toLowerCase();
    const rows = document.querySelectorAll('#students-table-body tr');

    rows.forEach(row => {
        const text = row.innerText.toLowerCase();
        row.style.display = text.includes(query) ? '' : 'none';
    });
}

// Session Selector for Inspector Tab
function populateSheetSelector(sessions) {
    const selector = document.getElementById('sheet-selector');
    selector.innerHTML = '<option value="">Select a processed sheet...</option>';

    if (!sessions) return;

    sessions.forEach(s => {
        const opt = document.createElement('option');
        opt.value = s.sheet;
        opt.textContent = `${s.sheet} (${s.date}) — ${s.present}/${s.total} Present`;
        selector.appendChild(opt);
    });
}

// Session Pipeline Inspector
function loadSessionInspector(sheetName) {
    const inspectorContent = document.getElementById('inspector-content');
    if (!sheetName) {
        inspectorContent.style.display = 'none';
        return;
    }

    inspectorContent.style.display = 'block';

    const cleanSheet = sheetName.replace(/\.[^/.]+$/, "");
    
    // Step images list (Standard pipeline output filenames)
    const stepNames = [
        { key: '01_original', name: '1. Original Image' },
        { key: '02_greyscale', name: '2. Greyscale Conversion' },
        { key: '03_gaussian_blur', name: '3. Gaussian Blur' },
        { key: '04_binary_otsu', name: '4. Otsu Binarization' },
        { key: '05_morphology', name: '5. Morphological Clean' },
        { key: '06_deskewed', name: '6. Deskewed Align' },
        { key: '07_row_detection', name: '7. Row & Table Detect' },
        { key: '08_signature_cells', name: '8. Signature Crop Strip' },
        { key: '09_annotated', name: '9. Annotated Result' }
    ];

    const gallery = document.getElementById('steps-gallery');
    gallery.innerHTML = '';

    stepNames.forEach(step => {
        const stepFilename = `${cleanSheet}_${step.key}.jpg`;
        const imgUrl = `/api/step-image/${stepFilename}`;

        const card = document.createElement('div');
        card.className = 'step-card';
        card.onclick = () => openImageModal(imgUrl, step.name);

        card.innerHTML = `
            <img class="step-img" src="${imgUrl}" alt="${step.name}" onerror="this.src='/static/placeholder.png'">
            <div class="step-info">
                <div class="step-name">${step.name}</div>
            </div>
        `;
        gallery.appendChild(card);
    });

    // Populate student density table for this session
    if (currentSummaryData && currentSummaryData.sessions) {
        const sessionInfo = currentSummaryData.sessions.find(s => s.sheet === sheetName);
        if (sessionInfo) {
            document.getElementById('inspector-sheet-title').textContent = `Sheet: ${sessionInfo.sheet}`;
            document.getElementById('inspector-sheet-date').textContent = `Session Date: ${sessionInfo.date}`;
            document.getElementById('inspector-present-count').textContent = sessionInfo.present;
            document.getElementById('inspector-absent-count').textContent = sessionInfo.absent;
        }
    }

    // Populate ink meter student list
    const tbody = document.getElementById('inspector-students-body');
    tbody.innerHTML = '';

    if (currentSummaryData && currentSummaryData.students) {
        currentSummaryData.students.forEach(st => {
            const record = st.history.find(h => h.sheet === sheetName);
            if (record) {
                const tr = document.createElement('tr');
                const isPres = record.present;
                tr.innerHTML = `
                    <td style="font-weight:700;">${st.index}</td>
                    <td>${st.name}</td>
                    <td><span class="badge ${isPres ? 'badge-success' : 'badge-danger'}">${isPres ? 'PRESENT' : 'ABSENT'}</span></td>
                    <td>${isPres ? 'Detected (>0.5% ink)' : 'No Signature'}</td>
                    <td>
                        <div class="density-bar-container">
                            <div class="density-bar-fill" style="width: ${isPres ? '100%' : '5%'}; background: ${isPres ? '#10b981' : '#f43f5e'};"></div>
                        </div>
                    </td>
                `;
                tbody.appendChild(tr);
            }
        });
    }
}

// Lightbox Modal
function openImageModal(imgUrl, title) {
    document.getElementById('modal-image-title').textContent = title;
    document.getElementById('modal-image-src').src = imgUrl;
    document.getElementById('image-modal').classList.add('active');
}

function closeImageModal(e) {
    document.getElementById('image-modal').classList.remove('active');
}

// Student History Modal
function openStudentModal(student) {
    document.getElementById('modal-student-name').textContent = student.name;
    document.getElementById('modal-student-index').textContent = `Student Index: ${student.index} | Overall: ${student.percentage}% (${student.present_count}/${student.total_sessions})`;

    const tbody = document.getElementById('modal-student-history');
    tbody.innerHTML = '';

    if (student.history) {
        student.history.forEach(h => {
            const tr = document.createElement('tr');
            tr.innerHTML = `
                <td>${h.date}</td>
                <td>${h.sheet}</td>
                <td><span class="badge ${h.present ? 'badge-success' : 'badge-danger'}">${h.present ? 'PRESENT' : 'ABSENT'}</span></td>
            `;
            tbody.appendChild(tr);
        });
    }

    document.getElementById('student-modal').classList.add('active');
}

function closeStudentModal() {
    document.getElementById('student-modal').classList.remove('active');
}

// Clear Database
async function confirmClearData() {
    if (confirm('Are you sure you want to clear all attendance records from the database?')) {
        try {
            const response = await fetch('/api/clear-data', { method: 'POST' });
            const data = await response.json();
            alert(data.message || 'Database reset successfully.');
            fetchSummary();
        } catch (err) {
            alert('Failed to clear database: ' + err.message);
        }
    }
}

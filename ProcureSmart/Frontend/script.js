// script.js (Final Merged Version)

const API_BASE = 'http://127.0.0.1:5000';
let priceChartInstance = null;

// --- DOM ELEMENTS ---
const uploadForm = document.getElementById('uploadForm');
const materialSelect = document.getElementById('material-select');
const analysisContainer = document.getElementById('analysis-container');
const vendorDetails = document.getElementById('vendor-details');
const predictionTableBody = document.getElementById('prediction-table-body');
const allVendorsTableBody = document.getElementById('all-vendors-table-body');
const chartOverlay = document.getElementById('chart-overlay');
const notificationArea = document.getElementById('notification-area');
const fetchButton = document.getElementById('fetch-button');
const uploadButton = document.getElementById('uploadButton');
const uploadText = document.getElementById('uploadText');
const uploadIcon = document.getElementById('uploadIcon');

// --- UTILITY: NOTIFICATIONS ---
function showNotification(message, type = 'success') {
    const colorMap = {
        success: 'bg-emerald-900/80 border-emerald-500 text-emerald-200',
        error: 'bg-red-900/80 border-red-500 text-red-200',
        warning: 'bg-yellow-900/80 border-yellow-500 text-yellow-200'
    };
    const notification = document.createElement('div');
    notification.className = `p-4 border-l-4 rounded-lg shadow-xl backdrop-blur-sm ${colorMap[type]} transition-all duration-300 transform translate-y-0`;
    notification.innerHTML = `<p class="font-bold">${type.charAt(0).toUpperCase() + type.slice(1)}</p><p class="text-sm">${message}</p>`;
    notificationArea.appendChild(notification);
    setTimeout(() => {
        notification.classList.add('opacity-0', '-translate-y-4');
        notification.addEventListener('transitionend', () => notification.remove());
    }, 5000);
}

// --- CORE LOGIC: DATA FETCHING ---
async function loadMaterials() {
    try {
        const response = await fetch(`${API_BASE}/api/materials`);
        if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
        const materials = await response.json();
        
        materialSelect.innerHTML = '<option value="">-- Please select a material --</option>';
        if (materials && Array.isArray(materials) && materials.length > 0) {
            materials.forEach(material => {
                const option = document.createElement('option');
                option.value = material.material_id;
                option.textContent = `${material.material_id} - ${material.name || 'No Name'}`;
                materialSelect.appendChild(option);
            });
            analysisContainer.style.display = 'grid';
            fetchButton.disabled = true; // Disabled until a material is chosen
        } else {
            analysisContainer.style.display = 'none';
            showNotification("Materials data is empty. Please upload the CSV files.", 'warning');
        }
    } catch (error) {
        analysisContainer.style.display = 'none';
        showNotification(`Failed to load materials. Is the backend running?`, 'error');
        console.error("Error loading materials:", error);
    }
}

async function fetchData() {
    const materialId = materialSelect.value;
    if (!materialId) {
        showNotification("Please select a material first.", "warning");
        return;
    }
    
    fetchButton.disabled = true;
    chartOverlay.style.display = 'flex';
    chartOverlay.innerHTML = '<p class="text-lg text-sky-400 animate-pulse">Analyzing Insights...</p>';
    vendorDetails.innerHTML = '<p class="text-center p-4 text-sky-400 animate-pulse">Calculating best vendor...</p>';
    renderAllVendorsTable([]);

    try {
        const [priceResponse, vendorResponse] = await Promise.all([
            fetch(`${API_BASE}/api/predict?material_id=${materialId}`),
            fetch(`${API_BASE}/api/recommend_vendor?material_id=${materialId}`)
        ]);

        const priceData = await priceResponse.json();
        if (!priceResponse.ok) throw new Error(priceData.error || 'Prediction fetch failed.');
        renderPriceChart(priceData.historical_data, priceData.predictions);
        renderPredictionTable(priceData.predictions);
        chartOverlay.style.display = 'none';

        const vendorData = await vendorResponse.json();
        if (!vendorResponse.ok) throw new Error(vendorData.error || 'Vendor recommendation failed.');
        renderVendorDetails(vendorData.best_vendor, vendorData.weighted_score);
        renderAllVendorsTable(vendorData.all_vendors);

    } catch (error) {
        showNotification(error.message, 'error');
        chartOverlay.innerHTML = `<p class="text-lg text-red-400">${error.message}</p>`;
        vendorDetails.innerHTML = `<p class="text-center p-4 text-red-400">${error.message}</p>`;
    } finally {
        fetchButton.disabled = false;
    }
}

// --- CORE LOGIC: RENDERING ---
function renderPriceChart(historicalData, predictions) {
    const allLabels = historicalData.map(d => d.date).concat(predictions.map(p => p.date));
    const historicalPrices = historicalData.map(d => d.price);
    const forecastPrices = Array(historicalPrices.length).fill(null).concat(predictions.map(p => p.predicted_price));
    const confidenceLow = Array(historicalPrices.length).fill(null).concat(predictions.map(p => p.confidence_low));
    const confidenceHigh = Array(historicalPrices.length).fill(null).concat(predictions.map(p => p.confidence_high));
    
    const ctx = document.getElementById('priceChart').getContext('2d');
    if (priceChartInstance) priceChartInstance.destroy();

    Chart.defaults.color = '#94a3b8';
    Chart.defaults.borderColor = '#334155';

    priceChartInstance = new Chart(ctx, {
        type: 'line',
        data: {
            labels: allLabels,
            datasets: [
                { label: 'Historical', data: historicalPrices, borderColor: '#38bdf8', backgroundColor: 'rgba(56, 189, 248, 0.1)', tension: 0.1, fill: true },
                { label: 'Predicted', data: forecastPrices, borderColor: '#34d399', borderDash: [5, 5], tension: 0.1 },
                { label: 'Confidence', data: confidenceHigh, borderColor: 'rgba(252, 211, 77, 0.3)', pointRadius: 0, fill: '+1' },
                { label: 'Confidence', data: confidenceLow, borderColor: 'rgba(252, 211, 77, 0.3)', backgroundColor: 'rgba(252, 211, 77, 0.1)', pointRadius: 0, fill: 'origin' }
            ]
        },
        options: { responsive: true, maintainAspectRatio: false, scales: { y: { title: { display: true, text: 'Price' }}, x: { title: { display: true, text: 'Date' }}}, plugins: { legend: { labels: { color: '#cbd5e1' }}}}
    });
}

function renderPredictionTable(predictions) {
    predictionTableBody.innerHTML = '';
    if (predictions.length === 0) {
        predictionTableBody.innerHTML = '<tr><td colspan="3" class="p-3 text-center text-slate-500">No predictions available.</td></tr>';
        return;
    }
    predictions.forEach(p => {
        const row = predictionTableBody.insertRow();
        row.innerHTML = `<td class="p-3 text-sm font-medium text-slate-300">${p.date}</td><td class="p-3 text-sm font-semibold text-emerald-400">$${p.predicted_price.toFixed(2)}</td><td class="p-3 text-sm text-slate-400">$${p.confidence_low.toFixed(2)} - $${p.confidence_high.toFixed(2)}</td>`;
    });
}

function renderVendorDetails(bestVendor, score) {
    vendorDetails.innerHTML = `<p class="text-3xl font-bold text-emerald-400">${bestVendor.vendor_id}</p><div class="mt-4 p-3 bg-emerald-900/50 rounded-lg text-center"><span class="text-sm font-medium text-emerald-300">Final Score: </span><span class="text-lg font-bold text-emerald-200">${(score * 100).toFixed(2)}%</span></div>`;
}

function renderAllVendorsTable(vendors) {
    allVendorsTableBody.innerHTML = '';
    if (vendors.length === 0) {
        allVendorsTableBody.innerHTML = '<tr><td colspan="3" class="p-2 text-center text-xs text-slate-500">No vendors to compare.</td></tr>';
        return;
    }
    vendors.sort((a, b) => b.final_score - a.final_score).forEach((v, index) => {
        const isBest = index === 0;
        const row = allVendorsTableBody.insertRow();
        row.className = isBest ? 'bg-emerald-900/50 font-semibold' : '';
        row.innerHTML = `<td class="p-2 text-xs text-slate-300">${isBest ? '🥇 ' : ''}${v.vendor_id}</td><td class="p-2 text-xs text-slate-300">$${v.price_per_unit.toFixed(2)}</td><td class="p-2 text-xs text-right ${isBest ? 'text-emerald-400' : 'text-slate-300'}">${(v.final_score * 100).toFixed(2)}%</td>`;
    });
}

// --- EVENT LISTENERS ---
materialSelect.addEventListener('change', (event) => {
    const materialId = event.target.value;
    fetchButton.disabled = !materialId;
    if (materialId) {
        fetchData();
    }
});

uploadForm.addEventListener('submit', async (event) => {
    event.preventDefault();
    const formData = new FormData();
    const pricesFile = document.getElementById('pricesFile').files[0];
    const vendorsFile = document.getElementById('vendorsFile').files[0];
    const materialsFile = document.getElementById('materialsFile').files[0];

    if (!pricesFile || !vendorsFile || !materialsFile) {
        showNotification("Please select all three CSV files.", 'warning');
        return;
    }

    formData.append('material_prices.csv', pricesFile);
    formData.append('vendors.csv', vendorsFile);
    formData.append('materials.csv', materialsFile);

    uploadButton.disabled = true;
    uploadText.textContent = 'Uploading...';
    uploadIcon.innerHTML = `<svg class="animate-spin -ml-1 mr-2 h-5 w-5" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24"><circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle><path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path></svg>`;
    
    try {
        const response = await fetch(`${API_BASE}/api/upload_data`, { method: 'POST', body: formData });
        const data = await response.json();
        if (response.ok) {
            showNotification(data.message, 'success');
            await loadMaterials();
            uploadForm.reset();
        } else {
            throw new Error(data.error || 'Upload failed. Check backend logs.');
        }
    } catch (error) {
        showNotification(`Upload Failed: ${error.message}`, 'error');
    } finally {
        uploadButton.disabled = false;
        uploadText.textContent = 'Upload';
        uploadIcon.innerHTML = `<svg xmlns="http://www.w3.org/2000/svg" class="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2"><path stroke-linecap="round" stroke-linejoin="round" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-8l-4-4m0 0L8 8m4-4v12" /></svg>`;
    }
});

// --- INITIALIZATION ---
window.addEventListener('load', loadMaterials);
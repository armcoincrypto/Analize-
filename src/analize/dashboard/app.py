"""
Dashboard application.

A simple HTML dashboard served by FastAPI.
"""

from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from analize import __version__
from analize.config import get_settings

# Dashboard HTML template
DASHBOARD_HTML = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Analize Dashboard</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        .card { @apply bg-white rounded-lg shadow p-6; }
        .metric { @apply text-3xl font-bold; }
        .metric-label { @apply text-gray-500 text-sm; }
    </style>
</head>
<body class="bg-gray-100 min-h-screen">
    <nav class="bg-indigo-600 text-white p-4">
        <div class="container mx-auto flex justify-between items-center">
            <h1 class="text-xl font-bold">Analize Dashboard</h1>
            <div class="flex gap-4">
                <a href="#overview" class="hover:underline">Overview</a>
                <a href="#signals" class="hover:underline">Signals</a>
                <a href="#reports" class="hover:underline">Reports</a>
                <a href="#optimizer" class="hover:underline">Optimizer</a>
            </div>
        </div>
    </nav>

    <main class="container mx-auto p-6">
        <!-- Status Banner -->
        <div id="status-banner" class="bg-green-100 border border-green-400 text-green-700 px-4 py-3 rounded mb-6">
            <span class="font-bold">Status:</span> <span id="system-status">Loading...</span>
        </div>

        <!-- Overview Section -->
        <section id="overview" class="mb-8">
            <h2 class="text-2xl font-bold mb-4">Overview</h2>
            <div class="grid grid-cols-1 md:grid-cols-4 gap-4">
                <div class="card">
                    <div class="metric" id="total-signals">-</div>
                    <div class="metric-label">Total Signals</div>
                </div>
                <div class="card">
                    <div class="metric" id="win-rate">-</div>
                    <div class="metric-label">Win Rate</div>
                </div>
                <div class="card">
                    <div class="metric" id="profit-factor">-</div>
                    <div class="metric-label">Profit Factor</div>
                </div>
                <div class="card">
                    <div class="metric" id="active-symbols">-</div>
                    <div class="metric-label">Active Symbols</div>
                </div>
            </div>
        </section>

        <!-- Charts Section -->
        <section id="charts" class="mb-8">
            <h2 class="text-2xl font-bold mb-4">Performance</h2>
            <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div class="card">
                    <h3 class="font-bold mb-4">Equity Curve</h3>
                    <canvas id="equity-chart"></canvas>
                </div>
                <div class="card">
                    <h3 class="font-bold mb-4">Win Rate by Hour</h3>
                    <canvas id="hourly-chart"></canvas>
                </div>
            </div>
        </section>

        <!-- Signals Section -->
        <section id="signals" class="mb-8">
            <h2 class="text-2xl font-bold mb-4">Recent Signals</h2>
            <div class="card overflow-x-auto">
                <table class="w-full text-sm">
                    <thead>
                        <tr class="border-b">
                            <th class="text-left p-2">Time</th>
                            <th class="text-left p-2">Symbol</th>
                            <th class="text-left p-2">Entry</th>
                            <th class="text-left p-2">Exit</th>
                            <th class="text-right p-2">PnL %</th>
                            <th class="text-left p-2">Reason</th>
                        </tr>
                    </thead>
                    <tbody id="signals-table">
                        <tr><td colspan="6" class="p-4 text-center">Loading...</td></tr>
                    </tbody>
                </table>
            </div>
        </section>

        <!-- Optimizer Section -->
        <section id="optimizer" class="mb-8">
            <h2 class="text-2xl font-bold mb-4">Parameter Optimizer</h2>
            <div class="card">
                <form id="optimize-form" class="grid grid-cols-1 md:grid-cols-3 gap-4">
                    <div>
                        <label class="block text-sm font-medium mb-1">Symbol</label>
                        <select id="opt-symbol" class="w-full border rounded p-2">
                            <option value="BTCUSDT">BTCUSDT</option>
                            <option value="ETHUSDT">ETHUSDT</option>
                        </select>
                    </div>
                    <div>
                        <label class="block text-sm font-medium mb-1">Date Range</label>
                        <input type="date" id="opt-start" class="w-full border rounded p-2">
                    </div>
                    <div>
                        <label class="block text-sm font-medium mb-1">Objective</label>
                        <select id="opt-objective" class="w-full border rounded p-2">
                            <option value="profit_factor">Profit Factor</option>
                            <option value="win_rate">Win Rate</option>
                            <option value="sharpe">Sharpe Ratio</option>
                        </select>
                    </div>
                    <div class="md:col-span-3">
                        <button type="submit" class="bg-indigo-600 text-white px-4 py-2 rounded hover:bg-indigo-700">
                            Run Optimization
                        </button>
                    </div>
                </form>
                <div id="opt-results" class="mt-4 hidden">
                    <h4 class="font-bold mb-2">Results</h4>
                    <pre id="opt-output" class="bg-gray-100 p-4 rounded text-sm overflow-x-auto"></pre>
                </div>
            </div>
        </section>

        <!-- Suggestions Section -->
        <section id="suggestions" class="mb-8">
            <h2 class="text-2xl font-bold mb-4">Parameter Suggestions</h2>
            <div id="suggestions-list" class="space-y-4">
                <div class="card">
                    <p class="text-gray-500">No suggestions available. Run an optimization first.</p>
                </div>
            </div>
        </section>
    </main>

    <footer class="bg-gray-800 text-white p-4 mt-8">
        <div class="container mx-auto text-center text-sm">
            Analize v{{ version }} | Analysis-only mode |
            <a href="/docs" class="underline">API Docs</a>
        </div>
    </footer>

    <script>
        // API base URL
        const API_BASE = '';

        // Fetch status on load
        async function fetchStatus() {
            try {
                const res = await fetch(`${API_BASE}/status`);
                const data = await res.json();
                document.getElementById('system-status').textContent =
                    `${data.status} | v${data.version} | ${data.environment}`;

                if (data.storage_stats) {
                    document.getElementById('total-signals').textContent =
                        data.storage_stats.total_records?.toLocaleString() || '-';
                    document.getElementById('active-symbols').textContent =
                        data.storage_stats.unique_symbols || '-';
                }
            } catch (e) {
                document.getElementById('system-status').textContent = 'Error loading status';
                document.getElementById('status-banner').className =
                    'bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded mb-6';
            }
        }

        // Initialize charts
        function initCharts() {
            // Equity chart placeholder
            const equityCtx = document.getElementById('equity-chart').getContext('2d');
            new Chart(equityCtx, {
                type: 'line',
                data: {
                    labels: ['Day 1', 'Day 2', 'Day 3', 'Day 4', 'Day 5'],
                    datasets: [{
                        label: 'Cumulative PnL %',
                        data: [0, 1.2, 0.8, 2.1, 3.5],
                        borderColor: 'rgb(79, 70, 229)',
                        tension: 0.1
                    }]
                },
                options: {
                    responsive: true,
                    scales: { y: { beginAtZero: true } }
                }
            });

            // Hourly chart placeholder
            const hourlyCtx = document.getElementById('hourly-chart').getContext('2d');
            new Chart(hourlyCtx, {
                type: 'bar',
                data: {
                    labels: Array.from({length: 24}, (_, i) => `${i}:00`),
                    datasets: [{
                        label: 'Win Rate %',
                        data: Array.from({length: 24}, () => Math.random() * 30 + 40),
                        backgroundColor: 'rgba(79, 70, 229, 0.5)'
                    }]
                },
                options: {
                    responsive: true,
                    scales: { y: { beginAtZero: true, max: 100 } }
                }
            });
        }

        // Handle optimize form
        document.getElementById('optimize-form').addEventListener('submit', async (e) => {
            e.preventDefault();
            const results = document.getElementById('opt-results');
            const output = document.getElementById('opt-output');

            results.classList.remove('hidden');
            output.textContent = 'Running optimization...';

            try {
                const res = await fetch(`${API_BASE}/optimize`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        symbols: [document.getElementById('opt-symbol').value],
                        start_date: document.getElementById('opt-start').value || '2024-01-01',
                        end_date: new Date().toISOString().split('T')[0],
                        objective: document.getElementById('opt-objective').value
                    })
                });
                const data = await res.json();
                output.textContent = JSON.stringify(data, null, 2);
            } catch (e) {
                output.textContent = 'Error: ' + e.message;
            }
        });

        // Initialize
        fetchStatus();
        initCharts();
    </script>
</body>
</html>
'''


def create_dashboard_app() -> FastAPI:
    """Create the dashboard FastAPI application."""
    app = FastAPI(
        title="Analize Dashboard",
        version=__version__,
    )

    @app.get("/", response_class=HTMLResponse)
    async def dashboard_home(request: Request) -> HTMLResponse:
        """Serve the dashboard."""
        html = DASHBOARD_HTML.replace("{{ version }}", __version__)
        return HTMLResponse(content=html)

    return app


def mount_dashboard(app: FastAPI, path: str = "/dashboard") -> None:
    """Mount the dashboard on an existing FastAPI app."""
    dashboard = create_dashboard_app()
    app.mount(path, dashboard)

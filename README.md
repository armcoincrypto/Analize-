# Analize

**Cloud AI Analyzer for ScalperBot**

A comprehensive analysis platform that produces ML-ready datasets, statistical reports, and parameter optimization suggestions for trading bots. **Analysis-only by design** - it recommends but does not execute trades.

## Features

- **Data Ingestion**: Parse ScalperBot databases, logs, and market data (CSV, Parquet, JSON)
- **Feature Generation**: Compute 50+ technical indicators across multiple timeframes
- **Outcome Labeling**: Label signals with MFE/MAE, time-windowed outcomes
- **Statistics Engine**: Calculate win rate, profit factor, Sharpe ratio, and more
- **Parameter Optimizer**: Grid search, random search, and walk-forward validation
- **REST API**: Full-featured API for programmatic access
- **Dashboard**: Web UI for visualization and interaction
- **Notifications**: Alerts via Telegram, Slack, and email

## Quick Start

### Using Docker Compose

```bash
# Clone the repository
git clone https://github.com/armcoincrypto/Analize-.git
cd Analize-

# Start all services
docker-compose up -d

# Access the API
open http://localhost:8000/docs

# Access the dashboard
open http://localhost:8000/dashboard
```

### Local Development

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows

# Install dependencies
pip install -e ".[dev]"

# Start the API server
analize serve --reload

# Or run directly
uvicorn analize.api.app:app --reload
```

## Configuration

Copy `.env.example` to `.env` and configure:

```env
# Environment
ENVIRONMENT=development
LOG_LEVEL=INFO

# Database
DB_POSTGRES_URL=postgresql://user:pass@localhost:5432/analize
DB_SCALPERBOT_DB_URL=sqlite:///path/to/scalperbot.db

# Redis
REDIS_URL=redis://localhost:6379/0

# Storage
STORAGE_S3_BUCKET=analize-data
STORAGE_LOCAL_DATA_PATH=./data

# Notifications (optional)
NOTIFY_TELEGRAM_BOT_TOKEN=your-bot-token
NOTIFY_TELEGRAM_CHAT_ID=your-chat-id
NOTIFY_SLACK_WEBHOOK_URL=https://hooks.slack.com/...
```

## CLI Commands

```bash
# Start API server
analize serve --host 0.0.0.0 --port 8000

# Ingest data
analize ingest --source db --symbol BTCUSDT
analize ingest --source logs --path ./logs/
analize ingest --source csv --path ./data/signals.csv

# Run analysis
analize analyze --symbol BTCUSDT --start-date 2024-01-01 --end-date 2024-01-31

# Optimize parameters
analize optimize --symbol BTCUSDT --start-date 2024-01-01 --end-date 2024-01-31 \
  --objective profit_factor --tp-range 0.5,5.0,0.5 --sl-range 0.5,3.0,0.5

# Generate report
analize report --date 2024-01-15 --symbol BTCUSDT --output report.json

# Show storage stats
analize storage-stats
```

## HFT Research Tools (Standardized CLI)

The `tools/` directory contains research and analysis tools with standardized CLI arguments:

### Walk-Forward Optimization

```bash
# Basic 3-way walk-forward (prevents overfitting)
python tools/walkforward_run.py --symbol XRPUSDT --start 2024-01-01 --end 2024-03-01 --3way

# With database path and alias for --final-test-days
python tools/walkforward_run.py --db hft_trades.db --symbol XRP --final-days 7 --3way \
    --start 2024-01-01 --end 2024-03-01

# Custom parameter grid
python tools/walkforward_run.py --symbol XRPUSDT --start 2024-01-01 --end 2024-02-01 \
    --grid "tp=0.2,0.3,0.4; sl=0.1,0.15; tstop=30,60"
```

### Maker Fill Analysis

```bash
# Global analysis (all symbols)
python tools/maker_fill_report.py --db hft_trades.db --days 7

# Filter by symbol
python tools/maker_fill_report.py --db hft_trades.db --symbol XRP --days 30

# Calibrate fill model
python tools/maker_fill_report.py --db hft_trades.db --calibrate
```

### Strategy Stress Testing

```bash
# Basic stress test
python tools/stress_test.py --db hft_trades.db --days 7

# Filter by symbol
python tools/stress_test.py --db hft_trades.db --symbol XRP --days 30

# Load best params from walk-forward CSV and run stress test
python tools/stress_test.py --db hft_trades.db --symbol XRP --days 30 \
    --from-best reports/xrp_wf_3way.csv
```

### Complete Workflow Example

```bash
# 1. Run walk-forward optimization
python tools/walkforward_run.py --db hft_trades.db --symbol XRP --final-days 7 --3way \
    --start 2024-01-01 --end 2024-03-01

# 2. Analyze maker fills
python tools/maker_fill_report.py --db hft_trades.db --symbol XRP --days 30

# 3. Stress test with best params from walk-forward
python tools/stress_test.py --db hft_trades.db --symbol XRP --days 30 \
    --from-best reports/walkforward_3way_XRPUSDT_*.csv
```

### Common Options

| Option | Description |
|--------|-------------|
| `--db` | Database path (default: hft_trades.db) |
| `--symbol` | Symbol filter (e.g., XRP, XRPUSDT) |
| `--days` | Analysis period in days |
| `--json` | Output as JSON |
| `--final-days` | Alias for --final-test-days (walkforward) |
| `--from-best` | Load params from CSV (stress_test) |

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/status` | GET | Analyzer health & status |
| `/health` | GET | Health check |
| `/ingest` | POST | Trigger data ingestion |
| `/signals` | GET | Query signal records |
| `/reports/daily` | GET | Get daily report |
| `/reports/weekly` | GET | Get weekly report |
| `/optimize` | POST | Start optimization job |
| `/jobs/{job_id}` | GET | Get job status |
| `/suggestions` | GET | Get parameter suggestions |
| `/export/signals` | GET | Export signals to file |

## Architecture

```
analize/
├── api/            # FastAPI REST API
├── dashboard/      # Web dashboard
├── features/       # Technical indicator calculations
├── ingestion/      # Data ingestion (DB, logs, files)
├── labeler/        # Outcome labeling (MFE/MAE)
├── models/         # Pydantic & SQLAlchemy models
├── notifications/  # Alert notifications
├── optimizer/      # Parameter optimization
├── stats/          # Statistics engine
├── storage/        # Parquet & S3 storage
└── utils/          # Utilities (logging, hashing)
```

## Data Schema

### Signal Record (ML-ready)

| Field | Type | Description |
|-------|------|-------------|
| signal_id | UUID | Unique identifier |
| timestamp_utc | datetime | Signal timestamp |
| symbol | string | Trading pair |
| price_open/high/low/close | float | OHLC prices |
| volume | float | Volume |
| ema_1h_20, rsi_1h, atr_1h | float | HTF indicators |
| rsi_1m, bb_width, vol_zscore_20 | float | LTF indicators |
| filters_passed | list | Passed filters |
| expected_tp_pct, expected_sl_pct | float | TP/SL levels |
| mfe_pct, mae_pct | float | Max favorable/adverse excursion |
| pnl_pct, exit_reason | float, enum | Outcome |

## Deployment

### Kubernetes

```bash
# Apply manifests
kubectl apply -f deploy/kubernetes/

# Check status
kubectl -n analize get pods
```

### Terraform (AWS)

```bash
cd deploy/terraform
terraform init
terraform plan
terraform apply
```

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Run tests with coverage
pytest --cov=analize --cov-report=html

# Lint code
ruff check src/

# Type check
mypy src/analize
```

## Security

- **Analysis-only mode**: No trading credentials are stored
- **Read-only DB access**: Uses read replicas or exported copies
- **TLS**: All network connections encrypted
- **Audit trail**: Every optimization run logged with data hash

## License

MIT License - See [LICENSE](LICENSE) for details.

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

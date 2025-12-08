"""
Command-line interface for Analize.
"""

import asyncio
from datetime import datetime
from pathlib import Path

import click


@click.group()
@click.version_option()
def main() -> None:
    """Analize - Cloud AI Analyzer for ScalperBot."""
    pass


@main.command()
@click.option("--host", default="0.0.0.0", help="Host to bind to")
@click.option("--port", default=8000, help="Port to bind to")
@click.option("--reload", is_flag=True, help="Enable auto-reload")
def serve(host: str, port: int, reload: bool) -> None:
    """Start the API server."""
    import uvicorn

    uvicorn.run(
        "analize.api.app:app",
        host=host,
        port=port,
        reload=reload,
    )


@main.command()
@click.option("--source", type=click.Choice(["db", "logs", "csv"]), required=True)
@click.option("--path", type=click.Path(exists=True), help="Path to source file/directory")
@click.option("--symbol", help="Filter by symbol")
@click.option("--start-date", type=click.DateTime(), help="Start date")
@click.option("--end-date", type=click.DateTime(), help="End date")
def ingest(
    source: str,
    path: str | None,
    symbol: str | None,
    start_date: datetime | None,
    end_date: datetime | None,
) -> None:
    """Ingest data from various sources."""
    click.echo(f"Ingesting from {source}...")

    if source == "db":
        from analize.ingestion import ScalperBotDBIngestor

        ingestor = ScalperBotDBIngestor(path)
        stats = ingestor.get_stats()
        click.echo(f"Database stats: {stats}")

        signals = ingestor.ingest_trades_as_signals(start_date, end_date, symbol)
        click.echo(f"Ingested {len(signals)} signals")

    elif source == "logs":
        from analize.ingestion import LogParser

        parser = LogParser(Path(path) if path else None)
        entries = list(parser.parse_directory())
        click.echo(f"Parsed {len(entries)} log entries")
        click.echo(f"Stats: {parser.get_stats()}")

    elif source == "csv":
        from analize.ingestion import FileIngestor

        ingestor = FileIngestor()
        signals = ingestor.ingest_signals_csv(Path(path))
        click.echo(f"Ingested {len(signals)} signals from CSV")


@main.command()
@click.option("--symbol", required=True, help="Symbol to analyze")
@click.option("--start-date", type=click.DateTime(), required=True)
@click.option("--end-date", type=click.DateTime(), required=True)
@click.option("--output", type=click.Path(), help="Output file path")
def analyze(
    symbol: str,
    start_date: datetime,
    end_date: datetime,
    output: str | None,
) -> None:
    """Run analysis on signals."""
    from analize.storage import ParquetStorage
    from analize.stats import StatsEngine
    from analize.models.signals import SignalRecord

    click.echo(f"Analyzing {symbol} from {start_date} to {end_date}...")

    # Load signals
    storage = ParquetStorage()
    df = storage.read_signals(start_date.date(), end_date.date(), symbols=[symbol])

    if df.empty:
        click.echo("No signals found")
        return

    # Convert to SignalRecord
    signals = []
    for _, row in df.iterrows():
        try:
            signals.append(SignalRecord(**row.to_dict()))
        except Exception:
            continue

    click.echo(f"Loaded {len(signals)} signals")

    # Run analysis
    engine = StatsEngine()
    results = engine.analyze_signals(signals)

    # Output
    if output:
        import json

        with open(output, "w") as f:
            json.dump(results, f, indent=2, default=str)
        click.echo(f"Results saved to {output}")
    else:
        import json

        click.echo(json.dumps(results, indent=2, default=str))


@main.command()
@click.option("--symbol", required=True, help="Symbol to optimize")
@click.option("--start-date", type=click.DateTime(), required=True)
@click.option("--end-date", type=click.DateTime(), required=True)
@click.option("--objective", default="profit_factor", help="Optimization objective")
@click.option("--tp-range", default="0.5,5.0,0.5", help="TP range: min,max,step")
@click.option("--sl-range", default="0.5,3.0,0.5", help="SL range: min,max,step")
def optimize(
    symbol: str,
    start_date: datetime,
    end_date: datetime,
    objective: str,
    tp_range: str,
    sl_range: str,
) -> None:
    """Run parameter optimization."""
    from analize.storage import ParquetStorage
    from analize.optimizer import GridSearchOptimizer, ParameterSpace
    from analize.models.reports import OptimizationObjective
    from analize.models.signals import SignalRecord

    click.echo(f"Optimizing {symbol}...")

    # Parse ranges
    tp_min, tp_max, tp_step = map(float, tp_range.split(","))
    sl_min, sl_max, sl_step = map(float, sl_range.split(","))

    # Load signals
    storage = ParquetStorage()
    df = storage.read_signals(start_date.date(), end_date.date(), symbols=[symbol])

    if df.empty:
        click.echo("No signals found")
        return

    signals = []
    for _, row in df.iterrows():
        try:
            signals.append(SignalRecord(**row.to_dict()))
        except Exception:
            continue

    click.echo(f"Loaded {len(signals)} signals")

    # Run optimization
    param_spaces = [
        ParameterSpace.from_range("tp_pct", tp_min, tp_max, tp_step),
        ParameterSpace.from_range("sl_pct", sl_min, sl_max, sl_step),
    ]

    optimizer = GridSearchOptimizer(
        objective=OptimizationObjective(objective),
    )

    with click.progressbar(length=100, label="Optimizing") as bar:
        def progress(current: int, total: int) -> None:
            bar.update(int(current / total * 100) - bar.pos)

        result = optimizer.optimize_fast(signals, param_spaces)

    click.echo(f"\nBest parameters: {result.best_parameters}")
    click.echo(f"Best score: {result.best_score:.4f}")
    click.echo(f"Top results:")
    for i, r in enumerate(result.top_results[:5], 1):
        click.echo(f"  {i}. {r}")


@main.command()
@click.option("--date", type=click.DateTime(), required=True, help="Report date")
@click.option("--symbol", help="Filter by symbol")
@click.option("--output", type=click.Path(), help="Output file path")
def report(date: datetime, symbol: str | None, output: str | None) -> None:
    """Generate daily report."""
    from analize.storage import ParquetStorage
    from analize.stats import StatsEngine
    from analize.models.signals import SignalRecord

    click.echo(f"Generating report for {date.date()}...")

    # Load signals
    storage = ParquetStorage()
    symbols = [symbol] if symbol else None
    df = storage.read_signals(date.date(), date.date(), symbols=symbols)

    if df.empty:
        click.echo("No signals found")
        return

    signals = []
    for _, row in df.iterrows():
        try:
            signals.append(SignalRecord(**row.to_dict()))
        except Exception:
            continue

    # Generate report
    engine = StatsEngine()
    report = engine.generate_daily_report(signals, date)

    # Output
    if output:
        import json

        with open(output, "w") as f:
            json.dump(report.model_dump(), f, indent=2, default=str)
        click.echo(f"Report saved to {output}")
    else:
        click.echo(f"Total signals: {report.total_signals}")
        click.echo(f"Total trades: {report.total_trades}")
        click.echo(f"Win rate: {report.overall_win_rate:.1f}%")
        click.echo(f"Profit factor: {report.overall_profit_factor:.2f}")


@main.command()
def init_db() -> None:
    """Initialize the database."""
    from analize.storage.database import init_database

    click.echo("Initializing database...")
    asyncio.run(init_database())
    click.echo("Database initialized")


@main.command()
def storage_stats() -> None:
    """Show storage statistics."""
    from analize.storage import ParquetStorage

    storage = ParquetStorage()
    stats = storage.get_stats()

    click.echo("Storage Statistics:")
    click.echo(f"  Total files: {stats['total_files']}")
    click.echo(f"  Total size: {stats['total_size_mb']} MB")
    click.echo(f"  Total records: {stats['total_records']}")
    click.echo(f"  Symbols: {', '.join(stats['symbols']) or 'none'}")
    click.echo(f"  Date range: {stats['date_range']['min']} to {stats['date_range']['max']}")


if __name__ == "__main__":
    main()

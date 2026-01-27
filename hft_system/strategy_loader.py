"""
Strategy Loader
===============
Loads and applies strategy configurations from registry or generated files.

This module handles:
1. Loading strategies from YAML registry
2. Generating active strategy config files
3. Applying strategy overrides to runtime config
4. Validating strategy configurations

Usage:
    from hft_system.strategy_loader import StrategyLoader, StrategyConfig

    loader = StrategyLoader("strategies/registry.yml")
    strategy = loader.get_strategy("xrp_momentum_v1")

    # Apply to runtime
    loader.apply_to_config(strategy, SYSTEM_CONFIG, COST_MODEL)
"""

import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
import json

logger = logging.getLogger(__name__)

# Try to import yaml, fall back to json-only mode
try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False
    logger.warning("PyYAML not installed. Using JSON-only mode.")


@dataclass
class CausesConfig:
    """Cause signal configuration."""
    enabled: List[str] = field(default_factory=list)
    disabled: List[str] = field(default_factory=list)
    probe_causes: List[str] = field(default_factory=list)
    block_causes: List[str] = field(default_factory=list)


@dataclass
class ParamsConfig:
    """Strategy parameters."""
    take_profit_pct: float = 0.15
    stop_loss_pct: float = 0.10
    time_stop_seconds: int = 90
    min_conditions: int = 3
    min_imbalance: float = 0.60
    min_confidence: float = 0.6


@dataclass
class ExecutionConfig:
    """Execution policy configuration."""
    mode: str = "taker"  # taker | maker | auto
    auto_maker_spread_threshold_bps: float = 2.0
    max_spread_bps: float = 5.0
    maker_timeout_ms: int = 5000
    maker_price_offset_bps: float = 0.5


@dataclass
class RiskConfig:
    """Risk management configuration."""
    position_size_pct: float = 1.0
    max_positions: int = 1
    max_daily_trades: int = 50
    max_daily_loss_pct: float = 2.0
    max_drawdown_pct: float = 5.0
    loss_streak_pause_trades: int = 5
    loss_streak_pause_minutes: int = 30


@dataclass
class StrategyMetadata:
    """Strategy metadata for tracking."""
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    commit_hash: Optional[str] = None
    walkforward_run_id: Optional[str] = None
    final_test_pnl_pct: Optional[float] = None
    final_test_sharpe: Optional[float] = None
    final_test_win_rate: Optional[float] = None
    final_test_trades: Optional[int] = None
    stress_test_passed: bool = False
    stress_test_date: Optional[str] = None
    promotion_approved: bool = False
    promotion_date: Optional[str] = None
    notes: Optional[str] = None


@dataclass
class StrategyConfig:
    """Complete strategy configuration."""
    strategy_id: str
    symbol: str
    status: str  # active | champion | challenger | retired | testing
    description: str = ""

    regimes_allowed: List[str] = field(default_factory=list)
    regimes_blocked: List[str] = field(default_factory=list)

    causes: CausesConfig = field(default_factory=CausesConfig)
    params: ParamsConfig = field(default_factory=ParamsConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    metadata: StrategyMetadata = field(default_factory=StrategyMetadata)

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "strategy_id": self.strategy_id,
            "symbol": self.symbol,
            "status": self.status,
            "description": self.description,
            "regimes_allowed": self.regimes_allowed,
            "regimes_blocked": self.regimes_blocked,
            "causes": {
                "enabled": self.causes.enabled,
                "disabled": self.causes.disabled,
                "probe_causes": self.causes.probe_causes,
                "block_causes": self.causes.block_causes,
            },
            "params": {
                "take_profit_pct": self.params.take_profit_pct,
                "stop_loss_pct": self.params.stop_loss_pct,
                "time_stop_seconds": self.params.time_stop_seconds,
                "min_conditions": self.params.min_conditions,
                "min_imbalance": self.params.min_imbalance,
                "min_confidence": self.params.min_confidence,
            },
            "execution": {
                "mode": self.execution.mode,
                "auto_maker_spread_threshold_bps": self.execution.auto_maker_spread_threshold_bps,
                "max_spread_bps": self.execution.max_spread_bps,
                "maker_timeout_ms": self.execution.maker_timeout_ms,
                "maker_price_offset_bps": self.execution.maker_price_offset_bps,
            },
            "risk": {
                "position_size_pct": self.risk.position_size_pct,
                "max_positions": self.risk.max_positions,
                "max_daily_trades": self.risk.max_daily_trades,
                "max_daily_loss_pct": self.risk.max_daily_loss_pct,
                "max_drawdown_pct": self.risk.max_drawdown_pct,
                "loss_streak_pause_trades": self.risk.loss_streak_pause_trades,
                "loss_streak_pause_minutes": self.risk.loss_streak_pause_minutes,
            },
            "metadata": {
                "created_at": self.metadata.created_at,
                "updated_at": self.metadata.updated_at,
                "commit_hash": self.metadata.commit_hash,
                "walkforward_run_id": self.metadata.walkforward_run_id,
                "final_test_pnl_pct": self.metadata.final_test_pnl_pct,
                "final_test_sharpe": self.metadata.final_test_sharpe,
                "final_test_win_rate": self.metadata.final_test_win_rate,
                "final_test_trades": self.metadata.final_test_trades,
                "stress_test_passed": self.metadata.stress_test_passed,
                "stress_test_date": self.metadata.stress_test_date,
                "promotion_approved": self.metadata.promotion_approved,
                "promotion_date": self.metadata.promotion_date,
                "notes": self.metadata.notes,
            }
        }

    @classmethod
    def from_dict(cls, strategy_id: str, data: Dict) -> "StrategyConfig":
        """Create from dictionary."""
        # Parse causes
        causes_data = data.get("causes", {})
        causes = CausesConfig(
            enabled=causes_data.get("enabled", []),
            disabled=causes_data.get("disabled", []),
            probe_causes=causes_data.get("probe_causes", []),
            block_causes=causes_data.get("block_causes", []),
        )

        # Parse params
        params_data = data.get("params", {})
        params = ParamsConfig(
            take_profit_pct=params_data.get("take_profit_pct", 0.15),
            stop_loss_pct=params_data.get("stop_loss_pct", 0.10),
            time_stop_seconds=params_data.get("time_stop_seconds", 90),
            min_conditions=params_data.get("min_conditions", 3),
            min_imbalance=params_data.get("min_imbalance", 0.60),
            min_confidence=params_data.get("min_confidence", 0.6),
        )

        # Parse execution
        exec_data = data.get("execution", {})
        execution = ExecutionConfig(
            mode=exec_data.get("mode", "taker"),
            auto_maker_spread_threshold_bps=exec_data.get("auto_maker_spread_threshold_bps", 2.0),
            max_spread_bps=exec_data.get("max_spread_bps", 5.0),
            maker_timeout_ms=exec_data.get("maker_timeout_ms", 5000),
            maker_price_offset_bps=exec_data.get("maker_price_offset_bps", 0.5),
        )

        # Parse risk
        risk_data = data.get("risk", {})
        risk = RiskConfig(
            position_size_pct=risk_data.get("position_size_pct", 1.0),
            max_positions=risk_data.get("max_positions", 1),
            max_daily_trades=risk_data.get("max_daily_trades", 50),
            max_daily_loss_pct=risk_data.get("max_daily_loss_pct", 2.0),
            max_drawdown_pct=risk_data.get("max_drawdown_pct", 5.0),
            loss_streak_pause_trades=risk_data.get("loss_streak_pause_trades", 5),
            loss_streak_pause_minutes=risk_data.get("loss_streak_pause_minutes", 30),
        )

        # Parse metadata
        meta_data = data.get("metadata", {})
        metadata = StrategyMetadata(
            created_at=meta_data.get("created_at"),
            updated_at=meta_data.get("updated_at"),
            commit_hash=meta_data.get("commit_hash"),
            walkforward_run_id=meta_data.get("walkforward_run_id"),
            final_test_pnl_pct=meta_data.get("final_test_pnl_pct"),
            final_test_sharpe=meta_data.get("final_test_sharpe"),
            final_test_win_rate=meta_data.get("final_test_win_rate"),
            final_test_trades=meta_data.get("final_test_trades"),
            stress_test_passed=meta_data.get("stress_test_passed", False),
            stress_test_date=meta_data.get("stress_test_date"),
            promotion_approved=meta_data.get("promotion_approved", False),
            promotion_date=meta_data.get("promotion_date"),
            notes=meta_data.get("notes"),
        )

        return cls(
            strategy_id=strategy_id,
            symbol=data.get("symbol", ""),
            status=data.get("status", "testing"),
            description=data.get("description", ""),
            regimes_allowed=data.get("regimes_allowed", []),
            regimes_blocked=data.get("regimes_blocked", []),
            causes=causes,
            params=params,
            execution=execution,
            risk=risk,
            metadata=metadata,
        )


class StrategyLoader:
    """
    Loads and manages strategies from registry.
    """

    def __init__(self, registry_path: str = None):
        """
        Initialize strategy loader.

        Args:
            registry_path: Path to registry.yml (default: strategies/registry.yml)
        """
        if registry_path is None:
            # Find registry relative to this file or cwd
            possible_paths = [
                Path(__file__).parent.parent / "strategies" / "registry.yml",
                Path.cwd() / "strategies" / "registry.yml",
                Path("/root/Analize-/strategies/registry.yml"),
            ]
            for p in possible_paths:
                if p.exists():
                    registry_path = str(p)
                    break

        self.registry_path = registry_path
        self.registry_data: Dict = {}
        self.strategies: Dict[str, StrategyConfig] = {}
        self.assignments: Dict[str, Dict] = {}
        self.defaults: Dict = {}

        if registry_path and Path(registry_path).exists():
            self._load_registry()

    def _load_registry(self):
        """Load registry from file."""
        path = Path(self.registry_path)

        if path.suffix in [".yml", ".yaml"]:
            if not YAML_AVAILABLE:
                raise ImportError("PyYAML required for YAML registry. Install with: pip install pyyaml")
            with open(path, "r") as f:
                self.registry_data = yaml.safe_load(f)
        elif path.suffix == ".json":
            with open(path, "r") as f:
                self.registry_data = json.load(f)
        else:
            raise ValueError(f"Unsupported registry format: {path.suffix}")

        # Parse strategies
        strategies_data = self.registry_data.get("strategies", {})
        for strategy_id, strategy_dict in strategies_data.items():
            self.strategies[strategy_id] = StrategyConfig.from_dict(strategy_id, strategy_dict)

        # Parse assignments
        self.assignments = self.registry_data.get("assignments", {})

        # Parse defaults
        self.defaults = self.registry_data.get("defaults", {})

        logger.info(f"Loaded {len(self.strategies)} strategies from {self.registry_path}")

    def get_strategy(self, strategy_id: str) -> Optional[StrategyConfig]:
        """Get strategy by ID."""
        return self.strategies.get(strategy_id)

    def get_champion(self, symbol: str) -> Optional[StrategyConfig]:
        """Get champion strategy for symbol."""
        assignment = self.assignments.get(symbol, {})
        champion_id = assignment.get("champion")
        if champion_id:
            return self.get_strategy(champion_id)
        return None

    def get_challenger(self, symbol: str) -> Optional[StrategyConfig]:
        """Get challenger strategy for symbol."""
        assignment = self.assignments.get(symbol, {})
        challenger_id = assignment.get("challenger")
        if challenger_id:
            return self.get_strategy(challenger_id)
        return None

    def get_strategies_for_symbol(self, symbol: str) -> List[StrategyConfig]:
        """Get all strategies for a symbol."""
        return [s for s in self.strategies.values() if s.symbol == symbol]

    def get_active_strategies(self) -> List[StrategyConfig]:
        """Get all active/champion strategies."""
        return [s for s in self.strategies.values() if s.status in ["active", "champion"]]

    def list_strategies(self) -> Dict[str, Dict]:
        """List all strategies with summary info."""
        result = {}
        for sid, strategy in self.strategies.items():
            result[sid] = {
                "symbol": strategy.symbol,
                "status": strategy.status,
                "description": strategy.description,
                "final_test_pnl_pct": strategy.metadata.final_test_pnl_pct,
                "stress_test_passed": strategy.metadata.stress_test_passed,
            }
        return result

    def validate_strategy(self, strategy: StrategyConfig) -> List[str]:
        """
        Validate strategy configuration.

        Returns list of validation errors (empty if valid).
        """
        errors = []

        # Required fields
        if not strategy.symbol:
            errors.append("Missing symbol")

        # Params validation
        if strategy.params.take_profit_pct <= 0:
            errors.append("take_profit_pct must be positive")
        if strategy.params.stop_loss_pct <= 0:
            errors.append("stop_loss_pct must be positive")
        if strategy.params.time_stop_seconds <= 0:
            errors.append("time_stop_seconds must be positive")

        # Risk validation
        if strategy.risk.position_size_pct <= 0 or strategy.risk.position_size_pct > 100:
            errors.append("position_size_pct must be between 0 and 100")
        if strategy.risk.max_daily_loss_pct <= 0:
            errors.append("max_daily_loss_pct must be positive")

        # Execution validation
        if strategy.execution.mode not in ["taker", "maker", "auto"]:
            errors.append(f"Invalid execution mode: {strategy.execution.mode}")

        # For production readiness
        if strategy.status in ["champion", "active"]:
            if not strategy.metadata.stress_test_passed:
                errors.append("Champion/active strategy must pass stress tests")
            if not strategy.metadata.promotion_approved:
                errors.append("Champion/active strategy must be promotion approved")

        return errors

    def generate_config_file(
        self,
        strategy: StrategyConfig,
        output_path: str,
        format: str = "yaml"
    ) -> str:
        """
        Generate a config file from strategy.

        Args:
            strategy: Strategy configuration
            output_path: Output file path
            format: "yaml" or "json"

        Returns:
            Path to generated file
        """
        config_dict = {
            "# Generated strategy config": None,
            "# Source": f"registry:{strategy.strategy_id}",
            "# Generated at": datetime.now().isoformat(),
            "strategy": strategy.to_dict()
        }

        # Clean up comment keys for actual output
        output_dict = {
            "_meta": {
                "source": f"registry:{strategy.strategy_id}",
                "generated_at": datetime.now().isoformat(),
            },
            "strategy": strategy.to_dict()
        }

        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        if format == "yaml" and YAML_AVAILABLE:
            with open(path, "w") as f:
                yaml.dump(output_dict, f, default_flow_style=False, sort_keys=False)
        else:
            with open(path, "w") as f:
                json.dump(output_dict, f, indent=2)

        logger.info(f"Generated config file: {path}")
        return str(path)

    def load_config_file(self, config_path: str) -> Optional[StrategyConfig]:
        """
        Load strategy from a generated config file.

        Args:
            config_path: Path to config file

        Returns:
            StrategyConfig or None if not found
        """
        path = Path(config_path)
        if not path.exists():
            logger.error(f"Config file not found: {config_path}")
            return None

        if path.suffix in [".yml", ".yaml"]:
            if not YAML_AVAILABLE:
                raise ImportError("PyYAML required")
            with open(path, "r") as f:
                data = yaml.safe_load(f)
        else:
            with open(path, "r") as f:
                data = json.load(f)

        strategy_dict = data.get("strategy", {})
        strategy_id = strategy_dict.get("strategy_id", path.stem)

        return StrategyConfig.from_dict(strategy_id, strategy_dict)


def apply_strategy_to_runtime(
    strategy: StrategyConfig,
    system_config: Any,
    cost_model: Any = None
) -> Dict[str, Any]:
    """
    Apply strategy configuration to runtime config objects.

    Args:
        strategy: Strategy configuration
        system_config: SYSTEM_CONFIG object from config.py
        cost_model: COST_MODEL object from config.py (optional)

    Returns:
        Dict of changes made
    """
    changes = {}

    # Apply params
    if hasattr(system_config, 'take_profit_pct'):
        old = system_config.take_profit_pct
        system_config.take_profit_pct = strategy.params.take_profit_pct
        changes['take_profit_pct'] = (old, strategy.params.take_profit_pct)

    if hasattr(system_config, 'stop_loss_pct'):
        old = system_config.stop_loss_pct
        system_config.stop_loss_pct = strategy.params.stop_loss_pct
        changes['stop_loss_pct'] = (old, strategy.params.stop_loss_pct)

    if hasattr(system_config, 'time_stop_seconds'):
        old = system_config.time_stop_seconds
        system_config.time_stop_seconds = strategy.params.time_stop_seconds
        changes['time_stop_seconds'] = (old, strategy.params.time_stop_seconds)

    if hasattr(system_config, 'min_conditions'):
        old = system_config.min_conditions
        system_config.min_conditions = strategy.params.min_conditions
        changes['min_conditions'] = (old, strategy.params.min_conditions)

    # Apply risk
    if hasattr(system_config, 'position_size_pct'):
        old = system_config.position_size_pct
        system_config.position_size_pct = strategy.risk.position_size_pct
        changes['position_size_pct'] = (old, strategy.risk.position_size_pct)

    if hasattr(system_config, 'max_daily_trades'):
        old = system_config.max_daily_trades
        system_config.max_daily_trades = strategy.risk.max_daily_trades
        changes['max_daily_trades'] = (old, strategy.risk.max_daily_trades)

    # Apply execution mode to cost model
    if cost_model and hasattr(cost_model, 'execution_mode'):
        old = cost_model.execution_mode
        cost_model.execution_mode = strategy.execution.mode
        changes['execution_mode'] = (old, strategy.execution.mode)

    if cost_model and hasattr(cost_model, 'auto_maker_spread_threshold'):
        old = cost_model.auto_maker_spread_threshold
        cost_model.auto_maker_spread_threshold = strategy.execution.auto_maker_spread_threshold_bps / 100
        changes['auto_maker_spread_threshold'] = (old, strategy.execution.auto_maker_spread_threshold_bps / 100)

    logger.info(f"Applied strategy {strategy.strategy_id}: {len(changes)} config changes")
    return changes


def get_config_diff(strategy: StrategyConfig, system_config: Any, cost_model: Any = None) -> Dict[str, tuple]:
    """
    Get the diff between strategy and current config without applying.

    Args:
        strategy: Strategy configuration
        system_config: Current SYSTEM_CONFIG
        cost_model: Current COST_MODEL (optional)

    Returns:
        Dict of {param: (current_value, new_value)}
    """
    diff = {}

    # Compare params
    param_mappings = [
        ('take_profit_pct', strategy.params.take_profit_pct),
        ('stop_loss_pct', strategy.params.stop_loss_pct),
        ('time_stop_seconds', strategy.params.time_stop_seconds),
        ('min_conditions', strategy.params.min_conditions),
        ('position_size_pct', strategy.risk.position_size_pct),
        ('max_daily_trades', strategy.risk.max_daily_trades),
    ]

    for attr, new_val in param_mappings:
        if hasattr(system_config, attr):
            current = getattr(system_config, attr)
            if current != new_val:
                diff[attr] = (current, new_val)

    # Compare execution
    if cost_model and hasattr(cost_model, 'execution_mode'):
        current = cost_model.execution_mode
        if current != strategy.execution.mode:
            diff['execution_mode'] = (current, strategy.execution.mode)

    return diff

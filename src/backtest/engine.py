"""
Walk-Forward Backtesting Engine (QARE-MM v1.3)
Coordinates purged walk-forward cross-validation, temperature calibration,
and out-of-sample quantitative performance attribution net of 15 bps friction.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np

from src.backtest.metrics import BacktestMetrics
from src.backtest.walk_forward import FoldIndices, WalkForwardSplitter
from src.models.backbone_lgbm import LightGBMBackbone


@dataclass
class FoldMetrics:
    fold_id: int
    train_size: int
    val_size: int
    test_size: int
    rank_ic: float
    brier_score: float
    sharpe_gross: float
    sharpe_net: float
    win_rate: float
    cumulative_return: float
    max_drawdown: float
    fitted_temperature: float


@dataclass
class BacktestResult:
    fold_metrics: List[FoldMetrics] = field(default_factory=list)
    mean_rank_ic: float = 0.0
    ic_ir: float = 0.0
    overall_brier_score: float = 0.0
    overall_gross_sharpe: float = 0.0
    overall_net_sharpe: float = 0.0
    overall_max_drawdown: float = 0.0
    overall_win_rate: float = 0.0
    total_trades: int = 0
    oos_predictions: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    oos_scores: np.ndarray = field(default_factory=lambda: np.zeros(0))
    oos_realized_returns: np.ndarray = field(default_factory=lambda: np.zeros(0))
    oos_net_returns: np.ndarray = field(default_factory=lambda: np.zeros(0))
    passed_gate1_ic: bool = False
    passed_gate1_sharpe: bool = False

    def summary(self) -> Dict[str, Any]:
        return {
            "num_folds": len(self.fold_metrics),
            "mean_rank_ic": round(self.mean_rank_ic, 4),
            "ic_ir": round(self.ic_ir, 4),
            "overall_brier_score": round(self.overall_brier_score, 4),
            "overall_gross_sharpe": round(self.overall_gross_sharpe, 4),
            "overall_net_sharpe": round(self.overall_net_sharpe, 4),
            "overall_max_drawdown": round(self.overall_max_drawdown, 4),
            "overall_win_rate": round(self.overall_win_rate, 4),
            "total_trades": self.total_trades,
            "passed_gate1_ic": self.passed_gate1_ic,
            "passed_gate1_sharpe": self.passed_gate1_sharpe,
        }


class WalkForwardBacktester:
    """
    Executes locked walk-forward cross-validation:
    756d Train -> 10d Purge -> 126d Val -> 10d Purge -> 63d Test (Quarterly Step: 63d).
    Trains the backbone on train folds with sample uniqueness weights,
    calibrates temperature on validation folds, and evaluates out-of-sample metrics.
    """

    GATE1_IC_HURDLE: float = 0.025
    GATE1_SHARPE_HURDLE: float = 0.0  # Net of 15 bps friction

    def __init__(
        self,
        splitter: Optional[WalkForwardSplitter] = None,
        model_factory: Optional[Callable[[], Any]] = None,
        round_trip_cost: float = 0.0015,  # 15 bps
        execution_mode: str = "LONG_ONLY",  # "LONG_ONLY" or "LONG_SHORT"
        score_threshold: float = 0.0,
    ) -> None:
        self.splitter = splitter or WalkForwardSplitter()
        self.model_factory = model_factory or (
            lambda: LightGBMBackbone(n_estimators=50, learning_rate=0.05, n_jobs=1)
        )
        self.round_trip_cost = round_trip_cost
        self.execution_mode = execution_mode
        self.score_threshold = score_threshold

    def run(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_days: np.ndarray,
        barrier_end_days: np.ndarray,
        realized_returns: np.ndarray,
        sample_weights: Optional[np.ndarray] = None,
        total_days: Optional[int] = None,
    ) -> BacktestResult:
        """
        Runs the full walk-forward backtest loop across all valid purged folds.
        """
        X_arr = np.asarray(X)
        y_arr = np.asarray(y, dtype=int)
        days = np.asarray(sample_days, dtype=int)
        barrier_ends = np.asarray(barrier_end_days, dtype=int)
        returns = np.asarray(realized_returns, dtype=np.float64)
        weights = (
            np.asarray(sample_weights, dtype=np.float64)
            if sample_weights is not None
            else np.ones(len(y_arr), dtype=np.float64)
        )

        fold_metrics_list: List[FoldMetrics] = []
        all_oos_probs: List[np.ndarray] = []
        all_oos_scores: List[np.ndarray] = []
        all_oos_realized: List[np.ndarray] = []
        all_oos_net: List[np.ndarray] = []
        all_oos_labels: List[np.ndarray] = []
        all_oos_trades: List[np.ndarray] = []

        for fold in self.splitter.split_panel(
            sample_days=days,
            barrier_end_days=barrier_ends,
            total_days=total_days,
        ):
            train_idx = fold.train_indices
            val_idx = fold.val_indices
            test_idx = fold.test_indices

            if len(train_idx) < 30 or len(val_idx) < 10 or len(test_idx) < 5:
                continue

            # 1. Instantiate fresh model instance per fold
            model = self.model_factory()

            # 2. Train on train fold
            X_tr, y_tr, w_tr = X_arr[train_idx], y_arr[train_idx], weights[train_idx]
            model.fit(X_tr, y_tr, sample_weights=w_tr)

            # 3. Fit post-hoc Temperature Scaling on validation fold
            X_val, y_val = X_arr[val_idx], y_arr[val_idx]
            fitted_temp = 1.0
            if hasattr(model, "fit_temperature"):
                try:
                    fitted_temp = model.fit_temperature(X_val, y_val)
                except Exception:
                    fitted_temp = 1.0

            # 4. Predict calibrated probabilities on OOS test fold
            X_test, y_test = X_arr[test_idx], y_arr[test_idx]
            r_test = returns[test_idx]

            if hasattr(model, "predict_calibrated_proba"):
                probs = model.predict_calibrated_proba(X_test)
            elif hasattr(model, "predict_proba"):
                probs = model.predict_proba(X_test)
            else:
                probs = np.full((len(test_idx), 3), 1.0 / 3.0)

            # Continuous conviction score: p_buy - p_sell in [-1.0, +1.0]
            scores = probs[:, 0] - probs[:, 2]

            # Generate trade indicators
            if self.execution_mode == "LONG_SHORT":
                trades = np.zeros(len(scores))
                trades[scores > self.score_threshold] = 1.0
                trades[scores < -self.score_threshold] = -1.0
            else:  # LONG_ONLY
                trades = np.where(scores > self.score_threshold, 1.0, 0.0)

            net_ret = trades * r_test - np.abs(trades) * self.round_trip_cost

            # Evaluate fold metrics
            rank_ic = BacktestMetrics.rank_information_coefficient(scores, r_test)
            brier = BacktestMetrics.brier_score(probs, y_test)
            sharpe_gross = BacktestMetrics.annualized_sharpe(
                trades * r_test, trades, include_costs=False
            )
            sharpe_net = BacktestMetrics.annualized_sharpe(
                trades * r_test, trades, include_costs=True
            )
            cum_ret = float(np.sum(net_ret))
            mdd = BacktestMetrics.max_drawdown(net_ret)
            wr = BacktestMetrics.win_rate(net_ret, trade_mask=(trades != 0.0))

            fold_metric = FoldMetrics(
                fold_id=fold.fold_id,
                train_size=len(train_idx),
                val_size=len(val_idx),
                test_size=len(test_idx),
                rank_ic=rank_ic,
                brier_score=brier,
                sharpe_gross=sharpe_gross,
                sharpe_net=sharpe_net,
                win_rate=wr,
                cumulative_return=cum_ret,
                max_drawdown=mdd,
                fitted_temperature=fitted_temp,
            )
            fold_metrics_list.append(fold_metric)

            all_oos_probs.append(probs)
            all_oos_scores.append(scores)
            all_oos_realized.append(r_test)
            all_oos_net.append(net_ret)
            all_oos_labels.append(y_test)
            all_oos_trades.append(trades)

        if not fold_metrics_list:
            return BacktestResult()

        # Concatenate all OOS results across folds
        oos_p = np.vstack(all_oos_probs)
        oos_s = np.concatenate(all_oos_scores)
        oos_r = np.concatenate(all_oos_realized)
        oos_net = np.concatenate(all_oos_net)
        oos_y = np.concatenate(all_oos_labels)
        oos_t = np.concatenate(all_oos_trades)

        ic_vals = [f.rank_ic for f in fold_metrics_list]
        mean_ic = float(np.mean(ic_vals))
        std_ic = float(np.std(ic_vals))
        ic_ir = float(mean_ic / (std_ic + 1e-12)) if len(ic_vals) > 1 else 0.0

        overall_brier = BacktestMetrics.brier_score(oos_p, oos_y)
        overall_gross_sharpe = BacktestMetrics.annualized_sharpe(
            oos_r * oos_t, oos_t, include_costs=False
        )
        overall_net_sharpe = BacktestMetrics.annualized_sharpe(
            oos_r * oos_t, oos_t, include_costs=True
        )
        overall_mdd = BacktestMetrics.max_drawdown(oos_net)
        overall_wr = BacktestMetrics.win_rate(oos_net, trade_mask=(oos_t != 0.0))
        total_trades = int(np.sum(oos_t != 0.0))

        passed_ic = mean_ic >= self.GATE1_IC_HURDLE
        passed_sharpe = overall_net_sharpe > self.GATE1_SHARPE_HURDLE

        return BacktestResult(
            fold_metrics=fold_metrics_list,
            mean_rank_ic=mean_ic,
            ic_ir=ic_ir,
            overall_brier_score=overall_brier,
            overall_gross_sharpe=overall_gross_sharpe,
            overall_net_sharpe=overall_net_sharpe,
            overall_max_drawdown=overall_mdd,
            overall_win_rate=overall_wr,
            total_trades=total_trades,
            oos_predictions=oos_p,
            oos_scores=oos_s,
            oos_realized_returns=oos_r,
            oos_net_returns=oos_net,
            passed_gate1_ic=passed_ic,
            passed_gate1_sharpe=passed_sharpe,
        )

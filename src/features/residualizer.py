"""
2-Stage Sequential OLS Residualizer (QARE-MM v1.3)
Isolates idiosyncratic asset return by stripping market (SPY) and orthogonalized sector factor risk.
Eliminates multicollinearity via sequential univariate projections over rolling 60-day historical windows.
Preserves idiosyncratic alpha drift by strictly avoiding intercept subtraction out-of-sample.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class BetaEstimates:
    beta_mkt: float
    beta_sec: float
    alpha: float
    r_squared: float
    beta_s_mkt: float = 0.0
    alpha_s: float = 0.0


@dataclass(frozen=True)
class RollingResidualOutput:
    residuals: np.ndarray
    betas_mkt: np.ndarray
    betas_sec: np.ndarray
    sector_perp: np.ndarray
    alphas: np.ndarray
    r_squared: np.ndarray


class Residualizer:
    """
    Computes rolling 60-day 2-stage sequential OLS betas and idiosyncratic residuals.

    Stage 1: Regresses Sector ETF return on SPY over historical window [t-60, t-1].
             Decomposes sector return into market component and strictly orthogonal sector residual:
             r_{ETF_s, perp} = r_{ETF_s} - (alpha_s + beta_{s, mkt} * r_SPY)

    Stage 2: Decomposes stock return into market and orthogonalized sector exposures:
             Because Cov(r_SPY, r_{ETF_s, perp}) = 0 in-sample, multivariable regression
             decouples into two independent univariate projections without multicollinearity.

    Out-of-sample Residual:
             epsilon_{i, t} = r_{i, t} - (beta_{i, mkt} * r_{SPY, t} + beta_{i, sec} * r_{ETF_s, perp, t})
             Alpha drift is strictly preserved (intercept is NOT subtracted).
    """

    def __init__(
        self,
        window_size: int = 60,
        beta_mkt_bounds: Tuple[float, float] = (-1.0, 4.0),
        beta_sec_bounds: Tuple[float, float] = (-3.0, 3.0),
        beta_s_mkt_bounds: Tuple[float, float] = (-1.0, 3.0),
        ridge_floor: float = 1e-8,
    ) -> None:
        self.window_size = window_size
        self.beta_mkt_bounds = beta_mkt_bounds
        self.beta_sec_bounds = beta_sec_bounds
        self.beta_s_mkt_bounds = beta_s_mkt_bounds
        self.ridge_floor = ridge_floor

    def orthogonalize_sector(
        self,
        r_sec_hist: np.ndarray,
        r_spy_hist: np.ndarray,
        r_sec_current: float,
        r_spy_current: float,
    ) -> float:
        """
        Stage 1: Regresses Sector ETF return on SPY over historical window [t-60, t-1].
        Projects out-of-sample orthogonal sector return for day t.
        """
        beta_s, alpha_s, _ = self.fit_stage1_model(r_sec_hist, r_spy_hist)
        # Out-of-sample projection on day t
        r_sec_perp = r_sec_current - (alpha_s + beta_s * r_spy_current)
        return float(r_sec_perp)

    def fit_stage1_model(
        self,
        r_sec_hist: np.ndarray,
        r_spy_hist: np.ndarray,
    ) -> Tuple[float, float, np.ndarray]:
        """
        Fits Stage 1 over historical window and returns in-sample parameters and residuals:
        Returns: (beta_s_mkt, alpha_s, nu_s_hist) where Cov(r_spy_hist, nu_s_hist) == 0.
        """
        n_obs = len(r_spy_hist)
        s_m = r_spy_hist - np.mean(r_spy_hist)
        e_m = r_sec_hist - np.mean(r_sec_hist)

        cov_sec_spy = np.sum(s_m * e_m) / n_obs
        var_spy = np.sum(s_m**2) / n_obs

        beta_s = float(
            np.clip(
                cov_sec_spy / (var_spy + self.ridge_floor),
                self.beta_s_mkt_bounds[0],
                self.beta_s_mkt_bounds[1],
            )
        )
        alpha_s = float(np.mean(r_sec_hist) - beta_s * np.mean(r_spy_hist))

        nu_s_hist = r_sec_hist - (alpha_s + beta_s * r_spy_hist)
        return beta_s, alpha_s, nu_s_hist

    def estimate_betas(
        self,
        r_asset_hist: np.ndarray,
        r_spy_hist: np.ndarray,
        r_sec_perp_hist: np.ndarray,
    ) -> BetaEstimates:
        """
        Stage 2: Regresses asset return on SPY and orthogonalized sector return.
        Exploits exact in-sample orthogonality: Cov(r_SPY, r_sec_perp) == 0.
        """
        n_obs = len(r_asset_hist)
        a_m = r_asset_hist - np.mean(r_asset_hist)
        s_m = r_spy_hist - np.mean(r_spy_hist)
        p_m = r_sec_perp_hist - np.mean(r_sec_perp_hist)

        # Univariate projection onto SPY: Cov(r_i, r_spy) / (Var(r_spy) + 1e-8)
        cov_asset_spy = np.sum(a_m * s_m) / n_obs
        var_spy = np.sum(s_m**2) / n_obs
        beta_mkt = float(
            np.clip(
                cov_asset_spy / (var_spy + self.ridge_floor),
                self.beta_mkt_bounds[0],
                self.beta_mkt_bounds[1],
            )
        )

        # Univariate projection onto orthogonal sector residual: Cov(r_i, nu_s) / (Var(nu_s) + 1e-8)
        cov_asset_sec = np.sum(a_m * p_m) / n_obs
        var_sec_perp = np.sum(p_m**2) / n_obs
        beta_sec = float(
            np.clip(
                cov_asset_sec / (var_sec_perp + self.ridge_floor),
                self.beta_sec_bounds[0],
                self.beta_sec_bounds[1],
            )
        )

        alpha = float(
            np.mean(r_asset_hist)
            - (beta_mkt * np.mean(r_spy_hist) + beta_sec * np.mean(r_sec_perp_hist))
        )

        # In-sample R-squared calculation
        y_hat = alpha + beta_mkt * r_spy_hist + beta_sec * r_sec_perp_hist
        ss_res = np.sum((r_asset_hist - y_hat) ** 2)
        ss_tot = np.sum(a_m**2)
        r_squared = float(max(0.0, min(1.0, 1.0 - (ss_res / (ss_tot + 1e-12)))))

        return BetaEstimates(
            beta_mkt=beta_mkt,
            beta_sec=beta_sec,
            alpha=alpha,
            r_squared=r_squared,
        )

    def compute_residual(
        self,
        r_asset: float,
        r_spy: float,
        r_sec_perp: float,
        betas: BetaEstimates,
    ) -> float:
        """
        Computes day t out-of-sample idiosyncratic return epsilon_{i, t}.
        Strictly preserves alpha drift by only subtracting factor beta exposures.
        """
        return float(r_asset - (betas.beta_mkt * r_spy + betas.beta_sec * r_sec_perp))

    def compute_rolling_residuals(
        self,
        stock_returns: np.ndarray,
        spy_returns: np.ndarray,
        sector_returns: np.ndarray,
    ) -> RollingResidualOutput:
        """
        Fast rolling 60-day 2-stage sequential OLS across full time series arrays.
        Guarantees strictly out-of-sample projection on each day t >= window_size.
        """
        n = len(stock_returns)
        if len(spy_returns) != n or len(sector_returns) != n:
            raise ValueError(
                f"Array length mismatch: stock ({n}), spy ({len(spy_returns)}), sector ({len(sector_returns)})"
            )

        residuals = np.full(n, np.nan, dtype=np.float64)
        betas_mkt = np.full(n, np.nan, dtype=np.float64)
        betas_sec = np.full(n, np.nan, dtype=np.float64)
        sector_perp = np.full(n, np.nan, dtype=np.float64)
        alphas = np.full(n, np.nan, dtype=np.float64)
        r_squared = np.full(n, np.nan, dtype=np.float64)

        if n < self.window_size + 1:
            return RollingResidualOutput(
                residuals=residuals,
                betas_mkt=betas_mkt,
                betas_sec=betas_sec,
                sector_perp=sector_perp,
                alphas=alphas,
                r_squared=r_squared,
            )

        w = self.window_size
        for t in range(w, n):
            # Strict historical window [t-w, t-1]
            s_hist = spy_returns[t - w : t]
            e_hist = sector_returns[t - w : t]
            a_hist = stock_returns[t - w : t]

            # Stage 1: Sector orthogonalization
            beta_s, alpha_s, nu_s_hist = self.fit_stage1_model(e_hist, s_hist)

            # Day t out-of-sample orthogonal sector return
            sec_perp_t = sector_returns[t] - (alpha_s + beta_s * spy_returns[t])
            sector_perp[t] = sec_perp_t

            # Stage 2: Stock regression
            betas = self.estimate_betas(a_hist, s_hist, nu_s_hist)
            betas_mkt[t] = betas.beta_mkt
            betas_sec[t] = betas.beta_sec
            alphas[t] = betas.alpha
            r_squared[t] = betas.r_squared

            # Day t idiosyncratic residual
            eps_t = stock_returns[t] - (betas.beta_mkt * spy_returns[t] + betas.beta_sec * sec_perp_t)
            residuals[t] = eps_t

        return RollingResidualOutput(
            residuals=residuals,
            betas_mkt=betas_mkt,
            betas_sec=betas_sec,
            sector_perp=sector_perp,
            alphas=alphas,
            r_squared=r_squared,
        )

    def residualize_series(
        self,
        stock_returns: pd.Series,
        spy_returns: pd.Series,
        sector_returns: pd.Series,
    ) -> pd.DataFrame:
        """
        Aligns dates by inner join and calculates rolling out-of-sample residuals and factor betas.
        Returns formatted DataFrame indexed by date.
        """
        aligned = pd.concat(
            [
                stock_returns.rename("stock_return"),
                spy_returns.rename("spy_return"),
                sector_returns.rename("sector_return"),
            ],
            axis=1,
            join="inner",
        ).dropna()

        if aligned.empty:
            raise ValueError("No overlapping dates found between stock, SPY, and sector returns.")

        out = self.compute_rolling_residuals(
            aligned["stock_return"].values,
            aligned["spy_return"].values,
            aligned["sector_return"].values,
        )

        res_df = pd.DataFrame(
            {
                "stock_return": aligned["stock_return"],
                "spy_return": aligned["spy_return"],
                "sector_return": aligned["sector_return"],
                "beta_mkt": out.betas_mkt,
                "beta_sec": out.betas_sec,
                "sector_perp": out.sector_perp,
                "alpha": out.alphas,
                "r_squared": out.r_squared,
                "residual": out.residuals,
                "cum_residual": np.nan_to_num(out.residuals).cumsum(),
            },
            index=aligned.index,
        )
        # Restore NaNs in cum_residual before warmup window
        res_df.loc[res_df["residual"].isna(), "cum_residual"] = np.nan
        return res_df

    def residualize_ticker(
        self,
        ticker: str,
        market_loader,
        universe_selector,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        save_to_parquet: bool = False,
    ) -> pd.DataFrame:
        """
        Loads daily market data for ticker, its GICS Sector ETF, and SPY,
        computes daily log returns, executes 2-stage sequential OLS residualization,
        and optionally persists to data/processed/residuals/{ticker}.parquet using atomic POSIX write.
        """
        sec_etf = universe_selector.get_sector_etf(ticker)

        df_stock = market_loader.fetch_or_load_daily(ticker, start_date=start_date, end_date=end_date)
        df_spy = market_loader.fetch_or_load_daily("SPY", start_date=start_date, end_date=end_date)
        df_sec = market_loader.fetch_or_load_daily(sec_etf, start_date=start_date, end_date=end_date)

        if df_stock is None or df_stock.empty:
            raise ValueError(f"No price data available for '{ticker}'.")
        if df_spy is None or df_spy.empty:
            raise ValueError("No price data available for market benchmark 'SPY'.")
        if df_sec is None or df_sec.empty:
            raise ValueError(f"No price data available for sector ETF '{sec_etf}'.")

        # Compute log returns
        r_stock = np.log(df_stock["close"] / df_stock["close"].shift(1)).dropna()
        r_spy = np.log(df_spy["close"] / df_spy["close"].shift(1)).dropna()
        r_sec = np.log(df_sec["close"] / df_sec["close"].shift(1)).dropna()

        res_df = self.residualize_series(r_stock, r_spy, r_sec)

        if save_to_parquet:
            residuals_dir = market_loader.data_dir / "processed" / "residuals"
            residuals_dir.mkdir(parents=True, exist_ok=True)
            target_path = residuals_dir / f"{ticker}.parquet"
            market_loader.atomic_write_parquet(res_df, target_path)

        return res_df

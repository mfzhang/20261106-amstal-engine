"""
Dynamic Rolling GICS & Cross-Correlation Peer Graph Builder (QARE-MM v1.3 PRD Section 5.7 & 5.10)
Constructs 10-node subgraphs centered at target stock v_0:
- Node 0 (v_0): Target equity (Role = 0.0)
- Nodes 1..4 (v_1..v_4): 4 GICS sub-industry peers (Role = 0.33)
- Nodes 5..8 (v_5..v_8): 4 rolling 5-day lead-lag cross-correlation peers (Role = 0.66)
- Node 9 (v_9): Corresponding GICS Sector ETF, e.g., XLK, XLE, XLV (Role = 1.0)
Constructs bidirectional star topology + self-loops with strict Point-in-Time causality.
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Batch, Data

from src.data.universe import UniverseSelector


class GraphBuilder:
    """
    Constructs Point-in-Time 10-node subgraphs for Graph Attention Network (GAT) processing.
    Supports both the locked 10-channel PRD Section 5.7 node schema (with categorical node roles
    in channel 9) and 15-channel microstructure node feature vectors.
    """

    NUM_NODES = 10
    NUM_PEERS = 8
    NUM_GICS_PEERS = 4
    NUM_LEAD_LAG_PEERS = 4
    PRD_NODE_DIM = 10
    FEATURE_DIM = 15

    # Categorical Node Roles (PRD Section 5.7, Channel 9):
    # 0.0 = Target (v0), 0.33 = GICS Peer (v1..v4), 0.66 = Lead-Lag Peer (v5..v8), 1.0 = Sector ETF (v9)
    NODE_ROLES: Tuple[float, ...] = (0.0, 0.33, 0.33, 0.33, 0.33, 0.66, 0.66, 0.66, 0.66, 1.0)

    def __init__(
        self,
        universe_selector: Optional[UniverseSelector] = None,
        lookback_corr: int = 60,
        feature_dim: Optional[int] = None,
    ) -> None:
        self.universe_selector = universe_selector or UniverseSelector(target_size=250)
        self.lookback_corr = lookback_corr
        self.feature_dim = feature_dim

    def select_gics_and_lead_lag_peers(
        self,
        target_ticker: str,
        residual_returns: Optional[pd.DataFrame] = None,
        as_of_date: Optional[Union[str, pd.Timestamp]] = None,
    ) -> Tuple[List[str], List[str], Dict[str, float]]:
        """
        Selects:
        1. Top 4 GICS sub-industry / sector peers (v_1..v_4)
        2. Top 4 rolling 5-day lead-lag cross-correlation peers (v_5..v_8) where peer 5d return
           leads target 5d return over trailing lookback_corr days strictly up to as_of_date.
        Returns (gics_peers_4, lead_lag_peers_4, lead_lag_corr_map).
        """
        target_ticker = target_ticker.upper()
        gics_candidates = self.universe_selector.get_peers(target_ticker, min_count=self.NUM_PEERS)
        if not gics_candidates:
            all_tickers = [t for t in self.universe_selector.get_all_registered_tickers() if t != target_ticker]
            gics_candidates = all_tickers[:self.NUM_PEERS]

        gics_peers = gics_candidates[:self.NUM_GICS_PEERS]
        remaining_pool = [t for t in gics_candidates if t not in gics_peers]
        corr_map: Dict[str, float] = {}

        if residual_returns is not None and not residual_returns.empty and target_ticker in residual_returns.columns:
            hist_df = residual_returns
            if as_of_date is not None:
                hist_df = hist_df.loc[:pd.Timestamp(as_of_date)]

            hist_window = hist_df.iloc[-self.lookback_corr:]
            if len(hist_window) >= 12:
                # 5-day rolling residual returns: peer(t-1) leading target(t)
                r5 = hist_window.rolling(5, min_periods=1).sum()
                target_5d = r5[target_ticker].values[1:]
                candidates = [c for c in hist_window.columns if c != target_ticker and c not in gics_peers]
                if not candidates:
                    candidates = [c for c in remaining_pool if c in hist_window.columns]

                scored = []
                for c in candidates:
                    peer_5d_lag = r5[c].values[:-1]
                    valid = ~(np.isnan(target_5d) | np.isnan(peer_5d_lag))
                    if np.sum(valid) >= 8:
                        c_val = np.corrcoef(peer_5d_lag[valid], target_5d[valid])[0, 1]
                        corr_val = 0.0 if np.isnan(c_val) else float(c_val)
                    else:
                        corr_val = 0.0
                    corr_map[c] = corr_val
                    scored.append((c, abs(corr_val)))

                scored.sort(key=lambda item: item[1], reverse=True)
                lead_lag_peers = [item[0] for item in scored[:self.NUM_LEAD_LAG_PEERS]]
                for c in remaining_pool:
                    if len(lead_lag_peers) >= self.NUM_LEAD_LAG_PEERS:
                        break
                    if c not in lead_lag_peers:
                        lead_lag_peers.append(c)
                return gics_peers, lead_lag_peers[:self.NUM_LEAD_LAG_PEERS], corr_map

        lead_lag_peers = remaining_pool[:self.NUM_LEAD_LAG_PEERS]
        return gics_peers, lead_lag_peers, corr_map

    def select_peers(
        self,
        target_ticker: str,
        residual_returns: Optional[pd.DataFrame] = None,
        as_of_date: Optional[Union[str, pd.Timestamp]] = None,
    ) -> List[str]:
        """
        Selects top 8 peers for target_ticker from the same GICS sub-industry or sector.
        Orders by rolling lookback_corr correlation of idiosyncratic residuals up to as_of_date (Point-in-Time).
        Falls back to highest liquidity (adv_rank) if return history is insufficient.
        """
        target_ticker = target_ticker.upper()
        candidates = self.universe_selector.get_peers(target_ticker, min_count=self.NUM_PEERS)
        if not candidates:
            all_tickers = [t for t in self.universe_selector.get_all_registered_tickers() if t != target_ticker]
            candidates = all_tickers[:self.NUM_PEERS]

        if residual_returns is not None and not residual_returns.empty and target_ticker in residual_returns.columns:
            hist_df = residual_returns
            if as_of_date is not None:
                hist_df = hist_df.loc[:pd.Timestamp(as_of_date)]

            hist_window = hist_df.iloc[-self.lookback_corr:]
            if len(hist_window) >= 10:
                avail_candidates = [c for c in candidates if c in hist_window.columns]
                if avail_candidates:
                    target_series = hist_window[target_ticker].values
                    corrs = []
                    for c in avail_candidates:
                        c_series = hist_window[c].values
                        valid_mask = ~(np.isnan(target_series) | np.isnan(c_series))
                        if np.sum(valid_mask) >= 10:
                            corr = np.corrcoef(target_series[valid_mask], c_series[valid_mask])[0, 1]
                            corr_val = 0.0 if np.isnan(corr) else abs(float(corr))
                        else:
                            corr_val = 0.0
                        corrs.append((c, corr_val))

                    corrs.sort(key=lambda item: item[1], reverse=True)
                    ranked_peers = [item[0] for item in corrs]

                    for c in candidates:
                        if c not in ranked_peers:
                            ranked_peers.append(c)
                    return ranked_peers[:self.NUM_PEERS]

        def get_rank(t: str) -> int:
            try:
                return self.universe_selector.get_security_master(t).adv_rank
            except KeyError:
                return 999

        sorted_candidates = sorted(candidates, key=get_rank)
        return sorted_candidates[:self.NUM_PEERS]

    @classmethod
    def build_prd_10d_node_vector(
        cls,
        res_20d: float = 0.0,
        res_5d: float = 0.0,
        beta_spy_60d: float = 1.0,
        pit_earnings_yield_z: float = 0.0,
        pit_rev_growth_z: float = 0.0,
        pit_op_margin_z: float = 0.0,
        idio_vol_20d_z: float = 0.0,
        lead_lag_corr_5d: float = 0.0,
        market_cap_usd: float = 50e9,
        node_role: float = 0.0,
    ) -> np.ndarray:
        """
        Constructs the exact 10-channel node feature vector defined in PRD Section 5.7:
        [0]: 20-day residual return
        [1]: 5-day residual return
        [2]: 60-day rolling Beta vs SPY ETF
        [3]: Point-in-Time Earnings Yield peer Z-score
        [4]: Point-in-Time YoY Quarterly Revenue Growth peer Z-score
        [5]: Point-in-Time Operating Margin peer Z-score
        [6]: 20-day Realized Idiosyncratic Volatility peer Z-score
        [7]: 5-day Lead-Lag cross-correlation coefficient with Target Node (v_0)
        [8]: Normalized Market Cap: ln(MarketCap_t) / 10
        [9]: Categorical Node Role: 0.0 = Target, 0.33 = GICS Peer, 0.66 = Lead-Lag Peer, 1.0 = Sector ETF
        """
        norm_mcap = float(np.log(max(market_cap_usd, 1.0)) / 10.0)
        return np.array(
            [
                float(res_20d),
                float(res_5d),
                float(beta_spy_60d),
                float(pit_earnings_yield_z),
                float(pit_rev_growth_z),
                float(pit_op_margin_z),
                float(idio_vol_20d_z),
                float(lead_lag_corr_5d),
                norm_mcap,
                float(node_role),
            ],
            dtype=np.float32,
        )

    def build_subgraph(
        self,
        target_ticker: str,
        node_features: Dict[str, Union[np.ndarray, List[float]]],
        pairwise_corrs: Optional[Dict[str, float]] = None,
        peer_tickers: Optional[List[str]] = None,
        feature_dim: Optional[int] = None,
    ) -> Data:
        """
        Builds a 10-node PyG Data subgraph for target_ticker:
        Node 0: target_ticker (Role = 0.0)
        Nodes 1..4: 4 GICS peers (Role = 0.33)
        Nodes 5..8: 4 Lead-Lag peers (Role = 0.66)
        Node 9: sector ETF (Role = 1.0)
        Topology: Bidirectional star (peers <-> target) + self-loops on all 10 nodes (28 edges total).
        """
        target_ticker = target_ticker.upper()
        if peer_tickers is None or len(peer_tickers) < self.NUM_PEERS:
            peer_tickers = self.select_peers(target_ticker)

        peers = [p.upper() for p in peer_tickers[:self.NUM_PEERS]]
        while len(peers) < self.NUM_PEERS:
            peers.append(target_ticker)

        try:
            sector_etf = self.universe_selector.get_sector_etf(target_ticker)
        except KeyError:
            sector_etf = "SPY"
        nodes_list = [target_ticker] + peers + [sector_etf]

        clean_feats = {k.upper(): v for k, v in node_features.items()} if node_features else {}

        # Infer feature dimension from explicit arg, instance setting, or provided node vectors
        inferred_dim = feature_dim or self.feature_dim
        if inferred_dim is None:
            for vec in clean_feats.values():
                arr = np.asarray(vec).flatten()
                if len(arr) in (self.PRD_NODE_DIM, self.FEATURE_DIM):
                    inferred_dim = len(arr)
                    break
            if inferred_dim is None:
                inferred_dim = self.FEATURE_DIM

        x_mat = np.zeros((self.NUM_NODES, inferred_dim), dtype=np.float32)
        for idx, ticker in enumerate(nodes_list):
            if ticker in clean_feats:
                feat = np.asarray(clean_feats[ticker], dtype=np.float32).flatten()
                if len(feat) >= inferred_dim:
                    x_mat[idx] = feat[:inferred_dim]
                else:
                    x_mat[idx, :len(feat)] = feat

            # Populate Categorical Node Role on Channel 9 when using 10-feature PRD schema
            if inferred_dim == self.PRD_NODE_DIM and (ticker not in clean_feats or len( np.asarray(clean_feats[ticker]).flatten()) < self.PRD_NODE_DIM):
                x_mat[idx, 9] = self.NODE_ROLES[idx]
            elif inferred_dim == self.PRD_NODE_DIM and x_mat[idx, 9] == 0.0 and idx > 0:
                x_mat[idx, 9] = self.NODE_ROLES[idx]

        # Edge construction:
        # Star edges: (0 -> k) and (k -> 0) for k in 1..9 (18 directed edges)
        # Self-loops: (i -> i) for i in 0..9 (10 directed edges)
        src_nodes = []
        dst_nodes = []
        edge_weights = []

        corrs = {k.upper(): v for k, v in pairwise_corrs.items()} if pairwise_corrs else {}

        # 1. Star edges between target v0 and peers/ETF v1..v9
        for k in range(1, self.NUM_NODES):
            peer = nodes_list[k]
            corr = float(corrs.get(peer, 0.50))
            # k -> 0 (incoming message to target)
            src_nodes.append(k)
            dst_nodes.append(0)
            edge_weights.append([corr])
            # 0 -> k (outgoing message from target)
            src_nodes.append(0)
            dst_nodes.append(k)
            edge_weights.append([corr])

        # 2. Self loops
        for i in range(self.NUM_NODES):
            src_nodes.append(i)
            dst_nodes.append(i)
            edge_weights.append([1.0])

        edge_index = torch.tensor([src_nodes, dst_nodes], dtype=torch.long)
        edge_attr = torch.tensor(edge_weights, dtype=torch.float32)
        x_tensor = torch.tensor(x_mat, dtype=torch.float32)

        data = Data(
            x=x_tensor,
            edge_index=edge_index,
            edge_attr=edge_attr,
            ticker=target_ticker,
            node_tickers=nodes_list,
        )
        return data

    def build_batch(self, subgraphs: List[Data]) -> Batch:
        """Batches a list of PyG Data subgraphs into a single Batch."""
        return Batch.from_data_list(subgraphs)

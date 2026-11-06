"""
Dynamic Rolling GICS & Cross-Correlation Peer Graph Builder (QARE-MM v1.3 PRD Section 5.10)
Constructs 10-node subgraphs centered at target stock v_0:
- Node 0 (v_0): Target equity
- Nodes 1..8 (v_1..v_8): 8 highest rolling correlation peers from the same GICS sub-industry/sector
- Node 9 (v_9): Corresponding Sector ETF (e.g., XLK, XLE, XLF)
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
    Each node represents an active equity or sector ETF with 15 microstructure features.
    """

    NUM_NODES = 10
    NUM_PEERS = 8
    FEATURE_DIM = 15

    def __init__(
        self,
        universe_selector: Optional[UniverseSelector] = None,
        lookback_corr: int = 60,
    ) -> None:
        self.universe_selector = universe_selector or UniverseSelector(target_size=250)
        self.lookback_corr = lookback_corr

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
            # Fall back to top liquid universe names if unknown ticker
            all_tickers = [t for t in self.universe_selector.get_all_registered_tickers() if t != target_ticker]
            candidates = all_tickers[:self.NUM_PEERS]

        if residual_returns is not None and not residual_returns.empty and target_ticker in residual_returns.columns:
            # Filter strictly up to as_of_date to eliminate forward lookahead
            hist_df = residual_returns
            if as_of_date is not None:
                hist_df = hist_df.loc[:pd.Timestamp(as_of_date)]

            # Take trailing lookback window
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

                    # Sort by correlation magnitude descending
                    corrs.sort(key=lambda item: item[1], reverse=True)
                    ranked_peers = [item[0] for item in corrs]

                    # If fewer than 8, pad with remaining candidates
                    for c in candidates:
                        if c not in ranked_peers:
                            ranked_peers.append(c)
                    return ranked_peers[:self.NUM_PEERS]

        # Liquidity fallback: sort candidates by adv_rank
        def get_rank(t: str) -> int:
            try:
                return self.universe_selector.get_security_master(t).adv_rank
            except KeyError:
                return 999

        sorted_candidates = sorted(candidates, key=get_rank)
        return sorted_candidates[:self.NUM_PEERS]

    def build_subgraph(
        self,
        target_ticker: str,
        node_features: Dict[str, Union[np.ndarray, List[float]]],
        pairwise_corrs: Optional[Dict[str, float]] = None,
        peer_tickers: Optional[List[str]] = None,
    ) -> Data:
        """
        Builds a 10-node PyG Data subgraph for target_ticker:
        Node 0: target_ticker
        Nodes 1..8: peer_tickers
        Node 9: sector ETF
        Topology: Bidirectional star (peers <-> target) + self-loops on all 10 nodes (28 edges total).
        """
        target_ticker = target_ticker.upper()
        if peer_tickers is None or len(peer_tickers) < self.NUM_PEERS:
            peer_tickers = self.select_peers(target_ticker)

        # Pad peers if still less than 8
        peers = [p.upper() for p in peer_tickers[:self.NUM_PEERS]]
        while len(peers) < self.NUM_PEERS:
            peers.append(target_ticker)

        try:
            sector_etf = self.universe_selector.get_sector_etf(target_ticker)
        except KeyError:
            sector_etf = "SPY"
        nodes_list = [target_ticker] + peers + [sector_etf]

        clean_feats = {k.upper(): v for k, v in node_features.items()} if node_features else {}

        # Construct (10, 15) node feature matrix
        x_mat = np.zeros((self.NUM_NODES, self.FEATURE_DIM), dtype=np.float32)
        for idx, ticker in enumerate(nodes_list):
            if ticker in clean_feats:
                feat = np.asarray(clean_feats[ticker], dtype=np.float32).flatten()
                if len(feat) >= self.FEATURE_DIM:
                    x_mat[idx] = feat[:self.FEATURE_DIM]
                else:
                    x_mat[idx, :len(feat)] = feat

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

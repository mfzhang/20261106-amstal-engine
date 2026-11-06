"""
Unit tests for Form 4 Parsing, Event Decay Kernel, and Smart-Money Divergence (QARE-MM v1.3)
Verifies:
1. Form 4 XML ingestion: CEO/CFO role detection, 10b5-1 discounting, open-market cluster scoring
2. Continuous Event-Decay Kernel: 12-channel vector geometry, half-life fidelity, and missingness masks
3. Point-in-Time causal boundary enforcement
4. Signed Smart-Money Divergence decomposition [D_t^+, D_t^-]
"""

import numpy as np
import pytest

from src.data.sec_edgar import Form4Parser, StandardInsiderTransaction
from src.features.event_decay import EventDecayKernel


def test_form4_xml_parsing_and_cluster_scoring() -> None:
    # Synthetic Form 4 XML: CEO open-market purchase of 10,000 shares @ $150
    sample_xml = """<?xml version="1.0"?>
    <ownershipDocument>
        <rptOwner>
            <rptOwnerName>
                <name>Huang Jen Hsun</name>
            </rptOwnerName>
            <reportingOwnerRelationship>
                <isOfficer>1</isOfficer>
                <officerTitle>President and CEO</officerTitle>
            </reportingOwnerRelationship>
        </rptOwner>
        <nonDerivativeTable>
            <nonDerivativeTransaction>
                <transactionCoding>
                    <transactionCode>P</transactionCode>
                </transactionCoding>
                <transactionDate>
                    <value>2023-04-10</value>
                </transactionDate>
                <transactionAmounts>
                    <transactionShares>
                        <value>10000</value>
                    </transactionShares>
                    <transactionPricePerShare>
                        <value>150.00</value>
                    </transactionPricePerShare>
                    <transactionAcquiredDisposedCode>
                        <value>A</value>
                    </transactionAcquiredDisposedCode>
                </transactionAmounts>
            </nonDerivativeTransaction>
        </nonDerivativeTable>
    </ownershipDocument>
    """

    txs = Form4Parser.parse_xml(sample_xml, filing_date="2023-04-10", acceptance_datetime="2023-04-10 16:30:00")
    assert len(txs) == 1
    tx = txs[0]
    assert tx.insider_name == "Huang Jen Hsun"
    assert tx.relation == "CEO"
    assert tx.transaction_code == "P"
    assert tx.is_10b5_1 is False
    assert tx.shares == 10000.0
    assert tx.price == 150.00
    assert tx.dollar_value == 1500000.0

    # Cluster score must be strongly positive for a $1.5M CEO purchase
    score = Form4Parser.compute_cluster_score(txs)
    assert 0.0 < score <= 1.0


def test_form4_10b5_1_discounting() -> None:
    tx_discretionary = StandardInsiderTransaction(
        filing_date="2023-05-01",
        acceptance_datetime="2023-05-01 17:00:00",
        transaction_date="2023-05-01",
        insider_name="Executive A",
        relation="CEO",
        transaction_code="P",
        is_10b5_1=False,
        shares=5000,
        price=100.0,
    )
    tx_planned = StandardInsiderTransaction(
        filing_date="2023-05-01",
        acceptance_datetime="2023-05-01 17:00:00",
        transaction_date="2023-05-01",
        insider_name="Executive B",
        relation="CEO",
        transaction_code="P",
        is_10b5_1=True,  # Pre-scheduled 10b5-1 plan
        shares=5000,
        price=100.0,
    )

    score_disc = Form4Parser.compute_cluster_score([tx_discretionary])
    score_plan = Form4Parser.compute_cluster_score([tx_planned])

    # Discretionary purchase carries 5x higher conviction than pre-scheduled 10b5-1 execution
    assert score_disc > score_plan
    assert score_plan > 0.0


def test_event_decay_kernel_12_channels_and_half_lives() -> None:
    as_of = "2023-06-15"

    # 1. Exact half-life check on news (tau = 3 days)
    # Event occurred 3 days ago: should decay by exactly 50%
    ev_news = (1.0, "2023-06-12")
    tensor = EventDecayKernel.build_event_tensor(as_of, news_event=ev_news)

    assert tensor.shape == (12,)
    # Channel 0: Decayed FinBERT sentiment
    assert np.isclose(tensor[0], 0.50, atol=1e-3)
    # Channel 1: News age
    assert tensor[1] == 3.0
    # Channel 9: News missing mask is 0.0 (present)
    assert tensor[9] == 0.0
    # Channels 10 and 11: Insider and 10-Q are missing (1.0)
    assert tensor[10] == 1.0
    assert tensor[11] == 1.0


def test_event_decay_kernel_point_in_time_causality() -> None:
    as_of = "2023-07-01"

    # Future event occurring after as_of_date must be completely suppressed
    future_news = (0.85, "2023-07-02")
    future_insider = (0.90, "2023-07-05")

    tensor = EventDecayKernel.build_event_tensor(
        as_of,
        news_event=future_news,
        insider_event=future_insider,
    )

    # Must be treated as missing
    assert tensor[0] == 0.0
    assert tensor[2] == 0.0
    assert tensor[9] == 1.0
    assert tensor[10] == 1.0


def test_smart_money_divergence_decomposition() -> None:
    # 1. Media Euphoria into Heavy Institutional Distribution
    # Positive sentiment + negative idiosyncratic return + high relative volume
    d_plus, d_minus = EventDecayKernel.compute_smart_money_divergence(
        sentiment=+0.80,
        residual_return=-0.03,
        volume=2000000,
        adv20=1000000,
    )
    assert d_plus > 0.0
    assert d_minus == 0.0

    # 2. Media Panic into Institutional Accumulation
    # Negative sentiment + positive idiosyncratic return + high volume
    d_plus2, d_minus2 = EventDecayKernel.compute_smart_money_divergence(
        sentiment=-0.80,
        residual_return=+0.03,
        volume=2000000,
        adv20=1000000,
    )
    assert d_plus2 == 0.0
    assert d_minus2 > 0.0

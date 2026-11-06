"""
Unit tests for Form 4 Parsing, XBRL Tag Fallback, SEC Redlines, FinBERT Sentiment,
Event Decay Kernel, and Smart-Money Divergence (QARE-MM v1.3)
Verifies:
1. Form 4 XML ingestion: CEO/CFO role detection, 10b5-1 discounting, open-market cluster scoring
2. US-GAAP XBRL Tag Fallback Hierarchy & Point-in-Time statement filtering
3. SEC 10-Q/10-K difflib Redline Analyzer (Item 1A & MD&A)
4. FinBERT News Sentiment Engine & post-16:00 ET Point-in-Time rollover
5. Continuous Event-Decay Kernel: 12-channel vector geometry, half-life fidelity, and modality masks
6. Signed Smart-Money Divergence decomposition [D_t^+, D_t^-]
"""

import numpy as np
import pytest

from src.data.sec_edgar import Form4Parser, StandardInsiderTransaction, XBRLTagResolver
from src.features.event_decay import EventDecayKernel
from src.features.news_sentiment import FinBertSentimentEngine, HeadlineObservation
from src.features.sec_diff import SecRedlineAnalyzer


def test_form4_xml_parsing_and_cluster_scoring() -> None:
    # Synthetic Form 4 XML: CEO open-market purchase of 10,000 shares @ $150
    sample_xml = """<?xml version="1.0"?>
    <ownershipDocument>
        <rptOwner>
            <rptOwnerName>
                <name>Huang Jen Hsun</name>
            </rptOwnerName>
            <reportingOwnerRelationship>
                <isOfficer> 1 </isOfficer>
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

    txs = Form4Parser.parse_xml(
        sample_xml,
        filing_date="2023-04-10",
        acceptance_datetime="2023-04-10T16:30:00-04:00",
    )
    assert len(txs) == 1
    tx = txs[0]
    assert tx.insider_name == "Huang Jen Hsun"
    assert tx.relation == "CEO"
    assert tx.transaction_code == "P"
    assert tx.is_10b5_1 is False
    assert tx.shares == 10000.0
    assert tx.price == 150.00
    assert tx.dollar_value == 1500000.0
    # Accepted at 16:30 ET (>16:00 ET cutoff) -> shifts to next trading day 2023-04-11
    assert tx.effective_date == "2023-04-11"

    # Point-in-Time check: as of 2023-04-10, this post-close filing must yield 0.0
    assert Form4Parser.compute_cluster_score(txs, as_of_date="2023-04-10") == 0.0
    # As of 2023-04-11, cluster score is strongly positive
    score = Form4Parser.compute_cluster_score(txs, as_of_date="2023-04-11")
    assert 0.0 < score <= 1.0


def test_form4_10b5_1_discounting() -> None:
    tx_discretionary = StandardInsiderTransaction(
        filing_date="2023-05-01",
        acceptance_datetime="2023-05-01T15:00:00-04:00",
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
        acceptance_datetime="2023-05-01T15:00:00-04:00",
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

    # Discretionary purchase carries 5x higher raw conviction than pre-scheduled 10b5-1 execution
    assert score_disc > score_plan
    assert score_plan > 0.0


def test_xbrl_tag_resolver_fallback_and_pit_filtering() -> None:
    # 1. Test fallback tag hierarchy for Revenue, Gross Profit (computed from CostOfRevenue), OpInc, EPS
    facts_q1 = {
        "SalesRevenueNet": 1200.0,  # 3rd fallback in REVENUE_TAGS
        "CostOfRevenue": 400.0,     # GrossProfit missing -> computed as 1200 - 400 = 800
        "IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest": 300.0,
        "ProfitLoss": 240.0,
        "IncomeLossFromContinuingOperationsPerDilutedShare": 2.40,
    }
    stmt_q1 = XBRLTagResolver.parse_income_statement_pit(
        period_end="2023-03-31",
        facts=facts_q1,
        acceptance_datetime="2023-05-04T17:15:00-04:00",  # After 16:00 ET -> 2023-05-05
    )
    assert stmt_q1.filed_date == "2023-05-05"
    assert stmt_q1.revenue == 1200.0
    assert stmt_q1.gross_profit == 800.0
    assert stmt_q1.operating_income == 300.0
    assert stmt_q1.net_income == 240.0
    assert stmt_q1.eps_diluted == 2.40

    # 2. Restatement Point-in-Time protection: restatement filed in August 2023
    stmt_q1_restated = XBRLTagResolver.parse_income_statement_pit(
        period_end="2023-03-31",
        facts={"Revenues": 1150.0, "GrossProfit": 750.0, "OperatingIncomeLoss": 280.0, "NetIncomeLoss": 220.0, "EarningsPerShareDiluted": 2.20},
        acceptance_datetime="2023-08-10T14:00:00-04:00",
    )

    all_stmts = [stmt_q1, stmt_q1_restated]
    # Query as of May 4 (before 16:00 ET cutoff rollover to May 5) -> empty
    assert len(XBRLTagResolver.filter_pit_statements(all_stmts, "2023-05-04")) == 0
    # Query as of June 1 -> sees original Q1 statement (revenue 1200), NOT the August restatement
    pit_june = XBRLTagResolver.filter_pit_statements(all_stmts, "2023-06-01")
    assert len(pit_june) == 1
    assert pit_june[0].revenue == 1200.0
    # Query as of September 1 -> sees restated Q1 statement (revenue 1150)
    pit_sept = XBRLTagResolver.filter_pit_statements(all_stmts, "2023-09-01")
    assert len(pit_sept) == 1
    assert pit_sept[0].revenue == 1150.0


def test_sec_redline_analyzer_difflib_and_decay() -> None:
    prev_1a = "Our supply chain is stable. We face normal market competition."
    curr_1a = "Our supply chain is stable. We received a DOJ subpoena and face material weakness investigation."
    prev_mda = "Revenue grew moderately during the quarter."
    curr_mda = "We achieved record revenue, margin expansion, and backlog growth and raised guidance."

    res = SecRedlineAnalyzer.score_filing_diff(
        prev_item1a=prev_1a,
        curr_item1a=curr_1a,
        prev_mda=prev_mda,
        curr_mda=curr_mda,
        acceptance_datetime="2023-05-10T15:00:00-04:00",
    )
    assert res.effective_date == "2023-05-10"
    assert res.item1a_risk_delta < 0.0  # Subpoena + material weakness -> negative risk delta
    assert res.mda_catalyst_score > 0.0  # Record revenue + margin expansion -> positive catalyst
    assert 0.0 <= res.similarity_ratio < 1.0

    # Check 15-day half-life decay
    d_risk, d_mda, elapsed = SecRedlineAnalyzer.compute_decayed_scores(res, "2023-05-25")
    assert elapsed == 15.0
    assert d_risk == pytest.approx(res.item1a_risk_delta * 0.5, abs=1e-4)
    assert d_mda == pytest.approx(res.mda_catalyst_score * 0.5, abs=1e-4)


def test_finbert_sentiment_engine_and_pit_rollover() -> None:
    engine = FinBertSentimentEngine(use_transformers=False)
    pos_score = engine.score_headline("NVIDIA beats earnings, raises guidance on record AI chip growth")
    neg_score = engine.score_headline("Company plunges after SEC subpoena and guidance cuts")
    assert pos_score > 0.4
    assert neg_score < -0.4

    headlines = [
        HeadlineObservation(
            headline="Company beats estimates and raises guidance",
            published_at="2023-06-09T14:30:00-04:00",  # Pre-close -> effective 2023-06-09
        ),
        HeadlineObservation(
            headline="Sudden bankruptcy filing announced after hours",
            published_at="2023-06-12T17:30:00-04:00",  # Post-close -> effective 2023-06-13
        ),
    ]

    # As of 2023-06-12 (3 days after 2023-06-09), the post-close June 12 headline is blocked!
    agg_score, age, mask = engine.aggregate_decayed_sentiment(headlines, as_of_date="2023-06-12")
    assert agg_score > 0.0
    assert age == 3.0
    assert mask == 1.0


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
    # Channel 9: News modality mask is 1.0 (active / Phase 3 Mode per PRD Section 5.10)
    assert tensor[9] == 1.0
    # Channels 10 and 11: Insider and 10-Q are missing (0.0 / Phase 2 Mode per PRD Section 5.10)
    assert tensor[10] == 0.0
    assert tensor[11] == 0.0


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

    # Must be treated as missing (mask = 0.0)
    assert tensor[0] == 0.0
    assert tensor[2] == 0.0
    assert tensor[9] == 0.0
    assert tensor[10] == 0.0


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

    # 3. Sentiment and price move in same direction (no divergence)
    d_plus3, d_minus3 = EventDecayKernel.compute_smart_money_divergence(
        sentiment=+0.80,
        residual_return=+0.03,
        volume=2000000,
        adv20=1000000,
    )
    assert d_plus3 == 0.0
    assert d_minus3 == 0.0

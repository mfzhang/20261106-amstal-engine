#!/usr/bin/env python3
"""
audit_specs.py - Automated Feedback Loop & Audit Suite for Amstal Specifications
Verifies:
1. File sync between root and amstal-engine/ (PRD.md, architecture.md)
2. Tensor dimension invariants and feature counts across all docs
3. JSON API contract schema validity and numerical range invariants
4. Mathematical formula consistency (Residualizer, Kelly, HMM, Triple Barrier)
5. Directory layout and component naming consistency
"""

import sys
import re
import json
from pathlib import Path

def get_repo_paths() -> tuple[Path, Path]:
    here = Path(__file__).resolve()
    for parent in [here.parent, here.parent.parent, here.parent.parent.parent]:
        if (parent / "amstal-engine").is_dir() and (parent / "PRD.md").is_file():
            return parent, parent / "amstal-engine"
    if here.parent.name == "scripts" and here.parent.parent.name == "amstal-engine":
        root = here.parent.parent.parent
        return root, root / "amstal-engine"
    cwd = Path.cwd()
    if (cwd / "amstal-engine").is_dir():
        return cwd, cwd / "amstal-engine"
    return cwd, Path("/path/to/missing/amstal-engine")

ROOT, ENGINE = get_repo_paths()

errors = []
warnings = []

def check(condition, msg):
    if not condition:
        errors.append(msg)

def warn(condition, msg):
    if not condition:
        warnings.append(msg)

print("=== AMSTAL SPECIFICATION AUDIT LOOP ===")

if not ENGINE.is_dir():
    errors.append(f"amstal-engine directory not found at {ENGINE}")

# --- Check 1: Root vs Engine File Synchronization ---
sync_files = ["PRD.md", "architecture.md"]
if (ROOT / "README.md").exists() or (ENGINE / "README.md").exists():
    sync_files.append("README.md")

for fname in sync_files:
    root_file = ROOT / fname
    engine_file = ENGINE / fname
    if not root_file.exists():
        errors.append(f"Missing root file: {fname}")
        continue
    if not engine_file.exists():
        errors.append(f"Missing engine file: {fname}")
        continue
    if root_file.read_text() != engine_file.read_text():
        errors.append(f"File mismatch between {root_file} and {engine_file}")

prd_text = (ENGINE / "PRD.md").read_text() if (ENGINE / "PRD.md").exists() else ""
arch_text = (ENGINE / "architecture.md").read_text() if (ENGINE / "architecture.md").exists() else ""
readme_text = (ENGINE / "README.md").read_text() if (ENGINE / "README.md").exists() else ""

docs_to_check = [("PRD.md", prd_text), ("architecture.md", arch_text)]
if readme_text:
    docs_to_check.append(("README.md", readme_text))

spec_docs = [("PRD.md", prd_text)]
if readme_text:
    spec_docs.append(("README.md", readme_text))

# --- Check 2: Stale References ---
stale_terms = ["Stockman", "stockman", "Arrow Flight", "8195"]
for term in stale_terms:
    for name, text in docs_to_check:
        if term in text:
            errors.append(f"Stale term '{term}' found in {name}")

# --- Check 3: Tensor Dimension Invariants ---
# Microstructure tensor: (Batch, 126, 15)
# LightGBM summary: (Batch, 105)
# Alt-data: (Batch, 12)
# Peer Graph: (Total_Nodes, 10)
# Fused vector: (Batch, 66) = 32 + 32 + 2
# HMM state: (Batch, 3)
# Head: 66 -> 32 -> 3
expected_tensors = [
    (r"\(Batch,\s*126,\s*15\)", "Microstructure tensor (Batch, 126, 15)"),
    (r"\(Batch,\s*105\)", "LightGBM multi-scale summary (Batch, 105)"),
    (r"\(Batch,\s*12\)", "Alt-data vector (Batch, 12)"),
    (r"\(Total_Nodes,\s*10\)", "Dynamic peer graph node features (Total_Nodes, 10)"),
    (r"\(B,\s*66\)", "Fused latent vector (B, 66)"),
    (r"\(Batch,\s*3\)", "HMM probabilities (Batch, 3)"),
    (r"Linear\s*\(\s*66\s*->\s*32\s*\)", "Classification head 66 -> 32"),
    (r"Linear\s*\(\s*32\s*->\s*3\s*\)", "Classification head 32 -> 3"),
]

for pattern, desc in expected_tensors:
    for name, text in docs_to_check:
        if not re.search(pattern, text):
            errors.append(f"Missing or mismatched tensor specification '{desc}' in {name}")

# --- Check 4: Feature Dictionary Channel Counts ---
# Sequence channels should be 0..14 (15 channels) parsed strictly from the Microstructure table
def extract_sequence_channels(doc_name: str, text: str) -> set[int]:
    indices = set()
    match = re.search(r"\|\s*Channel\s*\|\s*Feature Name\s*\|.*?\n(.*?)(?=\n###|\n---|\Z)", text, re.DOTALL | re.IGNORECASE)
    if not match:
        errors.append(f"Could not locate Microstructure table in {doc_name}")
        return indices
    table_section = match.group(1)
    row_matches = re.findall(r"\|\s*`([0-9\–\-]+)`\s*\|", table_section)
    for cell in row_matches:
        cell_clean = cell.replace("–", "-")
        if "-" in cell_clean:
            start_s, end_s = cell_clean.split("-", 1)
            for ch in range(int(start_s), int(end_s) + 1):
                indices.add(ch)
        else:
            indices.add(int(cell_clean))
    return indices

for name, text in spec_docs:
    seq_channels = extract_sequence_channels(name, text)
    expected_set = set(range(15))
    if seq_channels != expected_set:
        missing = expected_set - seq_channels
        surplus = seq_channels - expected_set
        if missing:
            errors.append(f"{name} Microstructure table missing sequence channels: {sorted(missing)}")
        if surplus:
            errors.append(f"{name} Microstructure table contains surplus sequence channels: {sorted(surplus)}")

# Alt-data channels: 12 features total parsed strictly from the Alt-Data table
def extract_alt_data_indices(doc_name: str, text: str) -> set[str]:
    indices = set()
    match = re.search(r"\|\s*Index\s*\|\s*Feature Name\s*\|\s*Source / Kernel\s*\|.*?\n(.*?)(?=\n###|\n---|\Z)", text, re.DOTALL | re.IGNORECASE)
    if not match:
        errors.append(f"Could not locate Alt-Data table in {doc_name}")
        return indices
    table_section = match.group(1)
    row_matches = re.findall(r"\|\s*`([0-9\–\-]+)`\s*\|", table_section)
    for cell in row_matches:
        indices.add(cell.replace("–", "-"))
    return indices

expected_alt_indices = {"0-1", "2-3", "4-5", "6", "7", "8", "9-11"}
for name, text in spec_docs:
    alt_indices = extract_alt_data_indices(name, text)
    if alt_indices != expected_alt_indices:
        missing = expected_alt_indices - alt_indices
        surplus = alt_indices - expected_alt_indices
        if missing:
            errors.append(f"{name} Alt-Data table missing feature indices: {sorted(missing)}")
        if surplus:
            errors.append(f"{name} Alt-Data table contains surplus feature indices: {sorted(surplus)}")

# Graph node features: 10 features total (0..9) parsed strictly from the Peer Graph table
def extract_graph_node_indices(doc_name: str, text: str) -> set[int]:
    indices = set()
    match = re.search(r"\|\s*Index\s*\|\s*Node Feature Description\s*\|.*?\n(.*?)(?=\n###|\n---|\Z)", text, re.DOTALL | re.IGNORECASE)
    if not match:
        errors.append(f"Could not locate Peer Graph table in {doc_name}")
        return indices
    table_section = match.group(1)
    row_matches = re.findall(r"\|\s*`([0-9]+)`\s*\|", table_section)
    for cell in row_matches:
        indices.add(int(cell))
    return indices

expected_graph_indices = set(range(10))
for name, text in spec_docs:
    graph_indices = extract_graph_node_indices(name, text)
    if graph_indices != expected_graph_indices:
        missing = expected_graph_indices - graph_indices
        surplus = graph_indices - expected_graph_indices
        if missing:
            errors.append(f"{name} Peer Graph table missing node indices: {sorted(missing)}")
        if surplus:
            errors.append(f"{name} Peer Graph table contains surplus node indices: {sorted(surplus)}")

# --- Check 5: JSON Schema Audit in PRD.md & README.md ---
json_blocks = re.findall(r"```json\s*(.*?)\s*```", prd_text, re.DOTALL)
check(len(json_blocks) >= 2, "Expected at least 2 JSON schema blocks in PRD.md")

for idx, block in enumerate(json_blocks):
    try:
        data = json.loads(block)
    except Exception as e:
        errors.append(f"Invalid JSON block #{idx+1} in PRD.md: {e}")
        continue

    # If it's the target output schema, check mathematical consistency
    if "signal_breakdown" in data:
        sb = data["signal_breakdown"]
        # All rank normalized features must be in [-1.0, +1.0]
        for key, val in sb.items():
            if "rank" in key or "skew" in key:
                if not (-1.0 <= val <= 1.0):
                    errors.append(f"JSON schema '{key}' value {val} is outside valid rank range [-1.0, +1.0]")
            if key == "closing_power_z":
                if val > 1.0:
                    errors.append(f"JSON schema 'closing_power_z' value {val} exceeds 1.0; cross-sectional rank normalization requires [-1.0, +1.0]")

        # Check calibrated probabilities
        if "calibration" in data and "calibrated_probabilities" in data["calibration"]:
            probs = data["calibration"]["calibrated_probabilities"]
            for pk, pv in probs.items():
                if not (0.0 <= pv <= 1.0):
                    errors.append(f"JSON schema calibrated probability '{pk}' value {pv} is outside [0.0, 1.0]")
            p_sum = sum(probs.values())
            if abs(p_sum - 1.0) > 0.001:
                errors.append(f"JSON schema calibrated probabilities do not sum to 1.0 (sum = {p_sum})")

        # Check macro regime probabilities
        if "macro_regime" in data and "filtered_probabilities" in data["macro_regime"]:
            reg_probs = data["macro_regime"]["filtered_probabilities"]
            for rk, rv in reg_probs.items():
                if not (0.0 <= rv <= 1.0):
                    errors.append(f"JSON schema HMM regime probability '{rk}' value {rv} is outside [0.0, 1.0]")
            r_sum = sum(reg_probs.values())
            if abs(r_sum - 1.0) > 0.001:
                errors.append(f"JSON schema HMM regime probabilities do not sum to 1.0 (sum = {r_sum})")

        # Check Kelly allocation cap
        if "execution_recommendation" in data:
            rec = data["execution_recommendation"]
            for alloc_key in ["recommended_portfolio_allocation_pct", "l1_normalized_portfolio_allocation_pct"]:
                if alloc_key in rec:
                    if rec[alloc_key] > 10.0:
                        errors.append(f"JSON schema allocation '{alloc_key}' ({rec[alloc_key]}%) exceeds single-stock ceiling of 10.0%")

# --- Check 6: Math Discrepancy Audit ---
# Check if README.md Kelly equation matches PRD.md
if "sum_{k" in readme_text and "b_k - c" in readme_text:
    errors.append("README.md contains old unexpanded Kelly summation formula 'sum_{k ... (b_k - c)' instead of explicit outcome payoffs")

# Check if alpha is subtracted in residualizer:
# Factor residualization should strip beta exposures: epsilon = r_i - (beta_mkt * r_spy + beta_sec * r_sec_perp)
# If alpha is subtracted: epsilon = r_i - (alpha + beta_mkt * r_spy + beta_sec * r_sec_perp), that removes idiosyncratic drift!
if r"\hat{\alpha}_{i, t-1}" in prd_text:
    warn(False, r"PRD.md subtracts stock intercept \hat{\alpha}_{i, t-1} in Stage 2 residualization; standard factor models preserve idiosyncratic drift by subtracting only factor exposures \hat{\beta}_{i, mkt} r_{SPY} + \hat{\beta}_{i, sec} r_{sec, \perp}")

# --- Check 7: Two-Tier Terminal Module Mapping ---
tier1_modules = ["GP", "PORT", "HMM", "DES", "FA"]
tier2_modules = ["SPLC", "EDGAR", "NEWS", "HP", "ANR"]
for mod in tier1_modules:
    if f"`{mod} <GO>`" not in prd_text:
        errors.append(f"Tier 1 module `{mod} <GO>` missing from PRD.md")
for mod in tier2_modules:
    if f"`{mod} <GO>`" not in prd_text:
        errors.append(f"Tier 2 module `{mod} <GO>` missing from PRD.md")

# --- Check 8: Python Code Syntax Validation in Documentation ---
import ast
import textwrap

for name, text in [("PRD.md", prd_text), ("architecture.md", arch_text), ("README.md", readme_text)]:
    py_blocks = re.findall(r"```python\s*(.*?)\s*```", text, re.DOTALL)
    for i, code in enumerate(py_blocks):
        clean_code = textwrap.dedent(code)
        try:
            ast.parse(clean_code)
        except SyntaxError as e:
            # If doc contains standalone placeholder lines with ..., try replacing only standalone ... lines with pass
            fallback_code = re.sub(r"^\s*\.\.\.\s*$", "pass", clean_code, flags=re.MULTILINE)
            try:
                ast.parse(fallback_code)
            except SyntaxError:
                errors.append(f"SyntaxError in Python code block #{i+1} in {name}: {e}")

# --- Check 9: Markdown Table of Contents Anchor Verification ---
def gfm_slug(header: str) -> str:
    """Exact GitHub Flavored Markdown slugification algorithm."""
    s = header.lower().strip()
    # Remove all chars except letters, numbers, spaces, hyphens, and underscores
    # Note: punctuation like & or ` or ( ) are removed, preserving adjacent spaces
    s = re.sub(r"[^\w\s\-]", "", s)
    # Each space (or run of spaces) becomes a hyphen
    s = s.replace(" ", "-")
    return s

for name, text in spec_docs:
    toc_links = re.findall(r"\[(.*?)\]\(#(.*?)\)", text)
    headers = re.findall(r"^#+\s+(.*?)$", text, re.MULTILINE)
    slug_counts = {}
    slugs = set()
    for h in headers:
        base = gfm_slug(h)
        if base in slug_counts:
            slug_counts[base] += 1
            slugs.add(f"{base}-{slug_counts[base]}")
        else:
            slug_counts[base] = 0
            slugs.add(base)

    for title, anchor in toc_links:
        if anchor not in slugs:
            errors.append(f"Broken TOC link in {name}: [{title}](#{anchor}) - target slug not found among {slugs}")

# --- Check 10: JSON API Contract Complete Field Validation ---
for name, text in docs_to_check:
    json_blocks = re.findall(r"```json\s*(.*?)\s*```", text, re.DOTALL)
    for idx, block in enumerate(json_blocks):
        try:
            data = json.loads(block)
        except json.JSONDecodeError as e:
            errors.append(f"Malformed JSON code block #{idx+1} in {name}: {e}")
            continue
        if "signal_breakdown" in data:
            sb = data["signal_breakdown"]
            required_keys = [
                "residual_momentum_20d_rank",
                "overnight_return_rank",
                "intraday_return_rank",
                "amihud_illiquidity_rank",
                "closing_power_rank",
                "decayed_finbert_sentiment",
                "decayed_form4_insider_cluster",
                "decayed_pead_shock",
                "decayed_10q_legal_risk_delta",
                "decayed_mda_catalyst_score",
                "peer_gat_spillover",
                "smart_money_divergence_pos",
                "smart_money_divergence_neg",
            ]
            for rk in required_keys:
                if rk not in sb:
                    errors.append(f"Missing required key '{rk}' in signal_breakdown in {name}")

print(f"\nAudit complete. Errors: {len(errors)}, Warnings: {len(warnings)}")
for e in errors:
    print(f" [FAIL] {e}")
for w in warnings:
    print(f" [WARN] {w}")

if errors:
    sys.exit(1)
else:
    print("\n [PASS] All checks passed successfully.")
    sys.exit(0)

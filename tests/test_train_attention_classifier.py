import sys
from pathlib import Path

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from scripts.train_attention_classifier import (  # noqa: E402
    TokenizedFeatureDataset,
    build_argparser,
    build_feature_groups,
)
from models.graph_classifier import GraphClassifier  # noqa: E402


def test_argparse_accepts_graph_flags():
    parser = build_argparser()
    args = parser.parse_args(
        [
            "--model-type",
            "graph",
            "--graph-hidden",
            "64",
            "--graph-heads",
            "2",
            "--graph-layers",
            "3",
            "--graph-dropout",
            "0.1",
        ]
    )
    assert args.model_type == "graph"
    assert args.graph_hidden == 64
    assert args.graph_heads == 2
    assert args.graph_layers == 3
    assert abs(args.graph_dropout - 0.1) < 1e-6


def test_tokenized_dataset_packs_and_pads_groups():
    df = pd.DataFrame(
        {
            "patient_id": ["p1", "p2"],
            "label": ["A", "B"],
            "label_enc": [0, 1],
            "LVEDV_ml": [100.0, 120.0],
            "LVESV_ml": [50.0, 60.0],
            "RVEDV_ml": [90.0, 95.0],
            "RVESV_ml": [40.0, 42.0],
            "MYO_th_mean_ED_mm": [6.0, 6.5],
            "LVEF_pct_robust": [55.0, 60.0],
        }
    )
    groups = build_feature_groups(df)
    ds = TokenizedFeatureDataset(df, groups, fit_stats=True)
    sample = ds[0]
    tokens = sample["tokens"]
    assert tokens.shape[0] == len(ds.group_order)
    assert tokens.shape[1] == ds.max_dim  # padded to max group size
    # Groups with a single feature should have zero padding in the tail
    for name, cols in groups.items():
        if len(cols) == 1:
            idx = ds.group_order.index(name)
            assert torch.allclose(tokens[idx, 1:], torch.zeros_like(tokens[idx, 1:]))
    # Forward pass through GAT should work
    model = GraphClassifier(token_dim=ds.max_dim, num_tokens=len(ds.group_order), num_classes=3)
    out = model(tokens.unsqueeze(0))
    assert out.shape == (1, 3)


def test_build_feature_groups_validates_input():
    df_empty = pd.DataFrame()
    try:
        build_feature_groups(df_empty)
        assert False, "Expected ValueError for empty dataframe"
    except ValueError:
        pass

    df_non_numeric = pd.DataFrame({"patient_id": ["p1"], "label": ["A"], "label_enc": [0], "foo": ["bar"]})
    try:
        build_feature_groups(df_non_numeric)
        assert False, "Expected ValueError for missing numeric features"
    except ValueError:
        pass

from __future__ import annotations

import numpy as np

from best_of_n_ebm.landscape import ToyEBMTransformer


def test_valid_sequence_has_no_global_violations() -> None:
    model = ToyEBMTransformer()
    rng = np.random.default_rng(7)
    prompt = model.sample_prompt(rng)
    seq = model.valid_sequence(prompt, rng)[None, :]
    measured = model.evaluate(prompt, seq)
    assert measured["violations"][0] == 0
    assert measured["valid"][0]
    assert measured["true_score"][0] == 1.0


def test_attention_weights_are_normalized() -> None:
    model = ToyEBMTransformer()
    rng = np.random.default_rng(8)
    prompt = model.sample_prompt(rng)
    batch = model.sample_candidates(prompt, 5, rng)
    attention = model.attention_weights(batch.tokens)
    np.testing.assert_allclose(attention.sum(axis=2), 1.0, atol=1e-8)


def test_artifacts_have_high_shortcut_mass_on_average() -> None:
    model = ToyEBMTransformer()
    rng = np.random.default_rng(9)
    prompt = model.sample_prompt(rng)
    batch = model.sample_candidates(prompt, 256, rng, near_valid_prob=0.35, artifact_prob=0.45)
    artifact = batch.proposal_type == "artifact"
    non_artifact = ~artifact
    assert artifact.any()
    assert batch.attention_shortcut[artifact].mean() > batch.attention_shortcut[non_artifact].mean()


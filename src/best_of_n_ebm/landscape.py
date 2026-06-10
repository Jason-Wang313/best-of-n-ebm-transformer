from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Prompt:
    """Prompt-specific global rule for a sequence completion problem."""

    offsets: np.ndarray
    bias: int


@dataclass(frozen=True)
class CandidateBatch:
    """Candidate sequences and all model/evaluation measurements."""

    tokens: np.ndarray
    proxy_energy: np.ndarray
    true_score: np.ndarray
    valid: np.ndarray
    violations: np.ndarray
    attention_shortcut: np.ndarray
    attention_entropy: np.ndarray
    repetition: np.ndarray
    local_attention_energy: np.ndarray
    global_penalty: np.ndarray
    proposal_type: np.ndarray

    def take(self, n: int) -> "CandidateBatch":
        return CandidateBatch(
            tokens=self.tokens[:n],
            proxy_energy=self.proxy_energy[:n],
            true_score=self.true_score[:n],
            valid=self.valid[:n],
            violations=self.violations[:n],
            attention_shortcut=self.attention_shortcut[:n],
            attention_entropy=self.attention_entropy[:n],
            repetition=self.repetition[:n],
            local_attention_energy=self.local_attention_energy[:n],
            global_penalty=self.global_penalty[:n],
            proposal_type=self.proposal_type[:n],
        )


class ToyEBMTransformer:
    """A deterministic Transformer-structured energy landscape.

    The true task is global: the second half of the sequence must be a
    prompt-conditioned mirror transform of the first half. The proxy energy is
    local and attention-mediated. It rewards low-cost local token interactions,
    including a deliberately misspecified artifact pocket made of special
    shortcut tokens. This lets Best-of-N expose a winner's-curse style failure
    without needing large models or labels during selection.
    """

    def __init__(
        self,
        length: int = 12,
        vocab: int = 16,
        dim: int = 16,
        seed: int = 314159,
    ) -> None:
        if length % 2 != 0:
            raise ValueError("length must be even")
        self.length = length
        self.half = length // 2
        self.vocab = vocab
        self.dim = dim
        self.artifact_tokens = np.arange(vocab - 4, vocab, dtype=np.int64)

        rng = np.random.default_rng(seed)
        self.token_emb = rng.normal(0.0, 0.7, size=(vocab, dim))
        self.pos_emb = rng.normal(0.0, 0.25, size=(length, dim))
        self.wq = rng.normal(0.0, 1.0 / np.sqrt(dim), size=(dim, dim))
        self.wk = rng.normal(0.0, 1.0 / np.sqrt(dim), size=(dim, dim))
        self.pair_cost = rng.normal(0.45, 0.08, size=(vocab, vocab))

        for token in range(vocab):
            self.pair_cost[token, token] = -0.42
        for left in self.artifact_tokens:
            for right in self.artifact_tokens:
                self.pair_cost[left, right] = -1.35

        self.distance_bias = 0.20
        self.global_weight = 0.62
        self.length_prior = 0.02

    def sample_prompt(self, rng: np.random.Generator) -> Prompt:
        offsets = rng.integers(0, self.vocab, size=self.half, dtype=np.int64)
        bias = int(rng.integers(0, self.vocab))
        return Prompt(offsets=offsets, bias=bias)

    def complete(self, first_half: np.ndarray, prompt: Prompt) -> np.ndarray:
        checksum = (int(first_half.sum()) + prompt.bias) % self.vocab
        mirror = first_half[::-1]
        return (mirror + prompt.offsets + checksum) % self.vocab

    def valid_sequence(self, prompt: Prompt, rng: np.random.Generator) -> np.ndarray:
        first = rng.integers(0, self.vocab - 4, size=self.half, dtype=np.int64)
        second = self.complete(first, prompt)
        return np.concatenate([first, second])

    def sample_candidates(
        self,
        prompt: Prompt,
        n: int,
        rng: np.random.Generator,
        near_valid_prob: float = 0.74,
        artifact_prob: float = 0.14,
    ) -> CandidateBatch:
        tokens = np.zeros((n, self.length), dtype=np.int64)
        proposal_type = np.empty(n, dtype="<U12")

        for i in range(n):
            draw = rng.random()
            if draw < near_valid_prob:
                seq = self.valid_sequence(prompt, rng)
                mutation_mask = rng.random(self.length) < 0.055
                if mutation_mask.any():
                    seq = seq.copy()
                    seq[mutation_mask] = rng.integers(
                        0, self.vocab, size=int(mutation_mask.sum())
                    )
                    proposal_type[i] = "near_valid"
                else:
                    proposal_type[i] = "valid"
            elif draw < near_valid_prob + artifact_prob:
                seq = self._artifact_sequence(rng)
                proposal_type[i] = "artifact"
            else:
                seq = rng.integers(0, self.vocab, size=self.length, dtype=np.int64)
                proposal_type[i] = "random"
            tokens[i] = seq

        measures = self.evaluate(prompt, tokens)
        return CandidateBatch(tokens=tokens, proposal_type=proposal_type, **measures)

    def evaluate(self, prompt: Prompt, tokens: np.ndarray) -> dict[str, np.ndarray]:
        tokens = np.asarray(tokens, dtype=np.int64)
        if tokens.ndim != 2 or tokens.shape[1] != self.length:
            raise ValueError(f"tokens must have shape (n, {self.length})")

        violations = self.global_violations(prompt, tokens)
        true_score = np.clip(1.0 - violations / self.half, 0.0, 1.0)
        valid = violations == 0
        global_penalty = violations / self.half

        attention = self.attention_weights(tokens)
        left = tokens[:, :, None]
        right = tokens[:, None, :]
        pair = self.pair_cost[left, right]
        local_attention_energy = (attention * pair).sum(axis=(1, 2)) / self.length

        artifact_mask = np.isin(left, self.artifact_tokens) & np.isin(
            right, self.artifact_tokens
        )
        repeat_mask = left == right
        shortcut_mass = attention * (artifact_mask + 0.35 * repeat_mask)
        attention_shortcut = shortcut_mass.sum(axis=(1, 2)) / self.length

        adjacency = tokens[:, 1:] == tokens[:, :-1]
        period_two = tokens[:, 2:] == tokens[:, :-2]
        repetition = 0.5 * adjacency.mean(axis=1) + 0.5 * period_two.mean(axis=1)

        entropy = -(attention * np.log(np.maximum(attention, 1e-12))).sum(axis=2)
        attention_entropy = entropy.mean(axis=1) / np.log(self.length)

        proxy_energy = (
            local_attention_energy
            + self.global_weight * global_penalty
            - 0.92 * attention_shortcut
            + self.length_prior * repetition
        )

        return {
            "proxy_energy": proxy_energy,
            "true_score": true_score,
            "valid": valid,
            "violations": violations,
            "attention_shortcut": attention_shortcut,
            "attention_entropy": attention_entropy,
            "repetition": repetition,
            "local_attention_energy": local_attention_energy,
            "global_penalty": global_penalty,
        }

    def global_violations(self, prompt: Prompt, tokens: np.ndarray) -> np.ndarray:
        first = tokens[:, : self.half]
        expected = np.array([self.complete(row, prompt) for row in first])
        observed = tokens[:, self.half :]
        return (expected != observed).sum(axis=1)

    def attention_weights(self, tokens: np.ndarray) -> np.ndarray:
        x = self.token_emb[tokens] + self.pos_emb[None, :, :]
        q = x @ self.wq
        k = x @ self.wk
        logits = q @ np.swapaxes(k, 1, 2) / np.sqrt(self.dim)
        positions = np.arange(self.length)
        dist = np.abs(positions[None, :] - positions[:, None])
        logits = logits - self.distance_bias * dist[None, :, :]
        logits = logits - logits.max(axis=2, keepdims=True)
        weights = np.exp(logits)
        weights /= weights.sum(axis=2, keepdims=True)
        return weights

    def _artifact_sequence(self, rng: np.random.Generator) -> np.ndarray:
        motif_length = int(rng.choice([2, 3, 4], p=[0.55, 0.30, 0.15]))
        motif = rng.choice(self.artifact_tokens, size=motif_length, replace=True)
        seq = np.resize(motif, self.length).astype(np.int64)
        if rng.random() < 0.35:
            idx = int(rng.integers(0, self.length))
            seq[idx] = int(rng.integers(0, self.vocab))
        return seq


from dataclasses import asdict, dataclass

import numpy as np

from protocol import (
    SIMULATOR_SECRET_FIELDS,
    PolicyView,
    PolicyVisibleDict,
    ScoringView,
    StreamManifestEntry,
    content_hash,
    derive_stream_id,
    make_manifest_entry,
)


@dataclass(frozen=True)
class Simcon:
    K: int = 10
    fami: int = 3
    w_f: float = 0.55
    a_f: float = 0.08
    sig: float = 5.30
    gamma: float = 1.5
    corr: float = 0.6
    noise: float = 0.6
    temp: float = 0.7
    dur_r: tuple[int, int] = (15, 45)
    blend: int = 5

    def __post_init__(self):
        if self.K < 2:
            raise ValueError("too less")
        if not 1 <= self.fami <= self.K:
            raise ValueError("fami out of bound")
        if self.sig <= 0:
            raise ValueError("sig must be pos")
        if self.gamma <= 0:
            raise ValueError("gamma +ve need")
        if not -1.0 < self.corr < 1.0:
            raise ValueError("-1 to +1")
        if self.noise < 0:
            raise ValueError("noise issue")
        if self.temp <= 0:
            raise ValueError("temp must be pos")
        if len(self.dur_r) != 2:
            raise ValueError("dur_r needs min, max")
        low, high = self.dur_r
        if low < 2 or high <= low:
            raise ValueError("it isn't legible")
        if not 0 <= self.blend <= low:
            raise ValueError("blend issue")


def _softmax(x, axis=-1): 
    """So that logits make sense in probability."""
    x = np.asarray(x, dtype=float)
    shifted = x - np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(shifted)
    return exp_x / exp_x.sum(axis=axis, keepdims=True)


def build_sim(cfg, rng):
    """Building S!"""
    family = np.arange(cfg.K) % cfg.fami
    rng.shuffle(family)
    same_family = family[:, None] == family[None, :]
    similarity = np.where(same_family, cfg.w_f, cfg.a_f).astype(float) 
    np.fill_diagonal(similarity, 1.0)
    return similarity, family


def _ar1_noise(rng, corr, std, T, K):
    """Generating AR(1) noise."""
    noise = np.zeros((T, K), dtype=float)
    if T == 0:
        return noise
    noise[0] = rng.normal(0.0, std, K)
    inno_scl = np.sqrt(max(1.0 - corr**2, 0.0))
    for t in range(1, T):
        epsilon_n = rng.normal(0.0, std, K)
        noise[t] = corr * noise[t - 1] + inno_scl * epsilon_n

    return noise


def _label_seq(cfg, n_signs, rng):
    if n_signs < 1:
        raise ValueError("insuff.")
    labels = [int(rng.integers(cfg.K))] # one sign lab
    for _ in range(1, n_signs):
        new = int(rng.integers(cfg.K))
        while new== labels[-1]:
            new= int(rng.integers(cfg.K))
        labels.append(new) # append list
    low, high = cfg.dur_r 
    durss = [
        int(rng.integers(low, high))
        for _ in range(n_signs)
    ]
    truth = []
    onsets = []
    seg_ends = []
    for label, duration in zip(labels, durss):
        onsets.append(len(truth))
        truth.extend([label] * duration)
        seg_ends.append(len(truth))
    # all except fin
    boundaries = seg_ends[:-1]
    return (np.asarray(truth, dtype=int), labels, durss, onsets, boundaries)


def _mean_logits(cfg, similarity, truth, labels, durss, onsets):
    """mean logit = alpha * s * S"""
    del onsets  
    T = len(truth)
    mean_logits = np.zeros((T, cfg.K), dtype=float)
    frame = 0
    for segment_index, (label, duration) in enumerate(
        zip(labels, durss)
    ):
        for within_segment in range(duration):
            progress = within_segment / max(duration - 1, 1)
            strength = progress**cfg.gamma
            current_logits = cfg.sig * strength * similarity[:, label]
            if (
                segment_index > 0
                and cfg.blend > 0
                and within_segment < cfg.blend
            ):
                previous_label = labels[segment_index - 1]
                blend_weight = within_segment / cfg.blend
                previous_logits = (
                    cfg.sig * similarity[:, previous_label]
                )
                current_logits = (
                    (1.0 - blend_weight) * previous_logits
                    + blend_weight * current_logits
                )
            mean_logits[frame] = current_logits
            frame += 1
    return mean_logits


def concat_stream(cfg, rng, n_signs=4):
    similarity, families = build_sim(cfg, rng)
    truth, labels, durss, onsets, boundaries = _label_seq(
        cfg, n_signs, rng
    )
    mean_logits = _mean_logits(
        cfg,
        similarity,
        truth,
        labels,
        durss,
        onsets,
    )
    correlated_noise = _ar1_noise(
        rng,
        cfg.corr,
        cfg.noise,
        len(truth),
        cfg.K,
    )
    logits = mean_logits + correlated_noise
    probabilities = _softmax(logits / cfg.temp).astype(np.float32)
    stream_id = content_hash({"p": probabilities.tolist(), "z": truth.tolist()})[:12]
    policy_visible = PolicyVisibleDict({"stream_id": stream_id, "p": probabilities})
    return {
        "stream_id": stream_id,
        "role": "synthetic_generator",
        "scenario": "abrupt-change",
        "seed": None,
        "policy_visible": policy_visible,
        "simulator_only": SIMULATOR_SECRET_FIELDS,
        "p": probabilities,
        "z": truth,
        "labels": labels,
        "durss": durss,
        "onsets": onsets,
        "bounds": boundaries,
        "families": families,
    }


def single_stream(cfg, rng):
    return concat_stream(cfg, rng, n_signs=1)


if __name__ == "__main__":
    cfg = Simcon()
    rng = np.random.default_rng(0)
    single = single_stream(cfg, rng)
    true_label = int(single["z"][0])
    length = len(single["z"])
    early = single["p"][: length // 3, true_label].mean()
    late = single["p"][-length // 3 :, true_label].mean()
    print(
        f"single: T={length}, true={true_label}, "
        f"P(true) early={early:.3f}, late={late:.3f}"
    )
    stream = concat_stream(cfg, rng, n_signs=4)
    frame_accuracy = (
        stream["p"].argmax(axis=1) == stream["z"]
    ).mean()
    print(
        f"continuous: T={len(stream['z'])}, "
        f"labels={stream['labels']}, "
        f"boundaries={stream['bounds']}"
    )
    print(f"frame-argmax accuracy={frame_accuracy:.3f}")
@dataclass(frozen=True, slots=True)
class GeneratedStream: 
    policy: PolicyView
    scoring: ScoringView
    manifest: StreamManifestEntry
    def __post_init__(self):
        identifiers = {
            self.policy.stream_id,
            self.scoring.stream_id,
            self.manifest.stream_id,
        }
        if len(identifiers) != 1:
            raise ValueError("generated stream views and manifest must share one ID")


def generate_stream(cfg, rng, *, role, scenario, seed, within_seed_index,
                    n_signs=4):
    """Generate unchanged evidence and project it across protocol boundaries."""
    raw = concat_stream(cfg, rng, n_signs=n_signs)
    frozen_config = {"simcon": asdict(cfg), "n_signs": int(n_signs)}
    generator_hash = content_hash(frozen_config)
    stream_id = derive_stream_id(
        role, scenario, seed, within_seed_index, generator_hash
    )
    policy = PolicyView(stream_id, raw["p"])
    truth = tuple(int(value) for value in raw["z"])
    scoring = ScoringView(
        stream_id=stream_id,
        truth=truth,
        labels=tuple(raw["labels"]),
        onsets=tuple(raw["onsets"]),
        bounds=tuple(raw["bounds"]),
        hierarchy_truth=tuple(frozenset((value,)) for value in truth),
    )
    manifest = make_manifest_entry(
        role=role,
        scenario=scenario,
        seed=seed,
        within_seed_index=within_seed_index,
        generator_config=frozen_config,
        policy_view=policy,
        scoring_view=scoring,
        simulator_only_field_names=SIMULATOR_SECRET_FIELDS,
    )
    return GeneratedStream(policy=policy, scoring=scoring, manifest=manifest)

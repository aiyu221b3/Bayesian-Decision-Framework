"""Shared deterministic configuration and requirement traceability for tests."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / ".kiro" / "specs" / "sequential-bayes-remediation"
CONFIG = json.loads((SPEC / "preservation_test_config.json").read_text(encoding="utf-8"))

NUMERIC_TOLERANCE = float(CONFIG["numeric_tolerance"])
BELIEF_SEED = int(CONFIG["seeds"]["belief"])
GENERATOR_SEED = int(CONFIG["seeds"]["generator"])
PROPERTY_SEEDS = tuple(int(seed) for seed in CONFIG["seeds"]["properties"])
HYPOTHESIS_SEED = int(CONFIG["seeds"]["hypothesis"])

_EXPLORE = "test_property_1_complete_sequential_bayes_remediation"
DEFECT_TRACEABILITY = {
    "1.1": (
        _EXPLORE,
        "test_bocpd_posterior_invariants",
        "test_bocpd_is_sensitive_to_surprising_evidence",
        "test_property_3_bocpd_recurrence_normalization_for_generated_inputs",
    ),
    "1.2": (_EXPLORE, "test_high_level_emits_once", "test_timer_alone_does_not_rearm"),
    "1.3": (
        _EXPLORE,
        "test_empty_and_single_have_same_keys",
        "test_pool_uses_counts_not_mean_rates",
        "test_property_5_generated_flat_metrics_have_stable_additive_counts_and_rates",
        "test_property_5_pool_sums_opportunities_before_deriving_rates",
    ),
    "1.4": (
        _EXPLORE,
        "test_boundary_miss_is_retained",
        "test_hit_to_miss_cannot_improve_result",
        "test_property_6_each_boundary_is_exactly_one_hit_or_right_censored_miss",
        "test_property_6_converting_a_hit_to_a_miss_cannot_improve_primary_metrics",
    ),
    "1.5": (
        _EXPLORE,
        "test_nonleaf_keeps_exact_identity",
        "test_leaf_id_permutation_preserves_tree_metrics",
        "test_property_7_structure_preserving_leaf_bijections_are_equivariant",
        "test_property_7_specificity_is_strictly_monotone_down_ancestor_chains",
    ),
    "1.6": (
        _EXPLORE,
        "test_property_8_generated_valid_manifests_freeze_before_one_paired_pass",
        "test_property_8_generated_invalid_manifests_abort",
        "test_property_8_duplicate_and_cross_role_overlap_abort_before_scoring",
        "test_property_8_order_mismatch_aborts_before_any_method_is_scored",
        "test_property_8_post_freeze_mutation_aborts_before_scoring",
        "test_property_8_second_final_pass_aborts_without_additional_scoring",
    ),
    "1.7": (
        _EXPLORE,
        "test_property_9_identical_visible_traces_ignore_all_hidden_variants",
        "test_property_9_every_simulator_secret_access_raises_protocol_violation",
        "test_property_9_generated_secret_access_attempts_always_abort",
        "test_supported_synthetic_smoke_has_all_methods_without_real_data_or_network",
    ),
    "1.8": (
        _EXPLORE,
        "test_property_11_generated_headline_reports_have_complete_replayable_provenance",
        "test_property_11_rejects_fewer_than_ten_predeclared_evaluation_seeds",
    ),
    "1.9": (_EXPLORE, "test_supported_python_and_dependency_versions_match_lock"),
    "1.10": (
        _EXPLORE,
        "test_property_12_generated_claims_have_one_category_and_adjacent_scope_metadata",
        "test_property_12_rejects_ambiguous_or_nonadjacent_claim_evidence",
    ),
    "1.11": (
        _EXPLORE,
        "test_property_12_generated_deferred_invocations_stop_before_import_or_path_access",
        "test_deferred_cli_exits_with_explicit_status_without_optional_imports",
    ),
}

PRESERVATION_TRACEABILITY = {
    "3.1": (
        "test_belief_remains_normalized_and_causal",
        "test_property_valid_posteriors_are_causal_finite_normalized_and_rho_zero_equivalent",
        "test_property_10_generated_ablation_and_posterior_invariants",
    ),
    "3.2": (
        "test_rho_zero_is_the_static_limit",
        "test_property_positive_rho_uses_the_documented_causal_forgetting_update",
        "test_property_9_identical_visible_traces_ignore_all_hidden_variants",
        "test_property_10_generated_ablation_and_posterior_invariants",
    ),
    "3.3": (
        "test_simulator_structural_invariants",
        "test_property_generator_replay_normalization_labels_lengths_onsets_and_bounds",
        "test_property_13_generated_generator_structural_preservation",
    ),
    "3.4": ("test_golden_first_qualifying_emission_time_and_identity_for_m0_through_mt",),
    "3.5": (
        "test_golden_flat_correctness_and_unit_cost_levenshtein_examples",
        "test_property_5_generated_flat_metrics_have_stable_additive_counts_and_rates",
        "test_property_5_pool_sums_opportunities_before_deriving_rates",
    ),
    "3.6": (
        "test_golden_hierarchy_membership_and_specificity_only",
        "test_property_7_structure_preserving_leaf_bijections_are_equivariant",
        "test_property_7_specificity_is_strictly_monotone_down_ancestor_chains",
    ),
    "3.7": (
        "test_golden_first_qualifying_emission_time_and_identity_for_m0_through_mt",
        "test_property_8_generated_valid_manifests_freeze_before_one_paired_pass",
        "test_property_8_order_mismatch_aborts_before_any_method_is_scored",
        "test_property_9_identical_visible_traces_ignore_all_hidden_variants",
        "test_supported_synthetic_smoke_has_all_methods_without_real_data_or_network",
        "test_property_11_generated_headline_reports_have_complete_replayable_provenance",
        "test_property_14_generated_scenario_method_outcomes_are_retained_exactly_once",
        "test_property_14_rejects_selective_omission_duplicates_and_status_relabeling",
    ),
    "3.8": (
        "test_property_9_identical_visible_traces_ignore_all_hidden_variants",
        "test_property_9_every_simulator_secret_access_raises_protocol_violation",
        "test_property_9_generated_secret_access_attempts_always_abort",
        "test_supported_synthetic_smoke_has_all_methods_without_real_data_or_network",
        "test_property_12_generated_deferred_invocations_stop_before_import_or_path_access",
        "test_deferred_import_is_standard_library_only_and_loads_no_pilot_module",
        "test_deferred_modules_are_not_imported_by_supported_synthetic_workflow",
        "test_deferred_scaffold_files_are_empty_or_guard_only",
    ),
    "3.9": (
        "test_beginner_friendly_explanatory_scope_is_preserved_without_claim_oracles",
        "test_property_12_generated_claims_have_one_category_and_adjacent_scope_metadata",
    ),
}

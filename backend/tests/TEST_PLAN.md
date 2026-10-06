# Backend test plan and traceability

Run from the repository root: `python -m pytest backend/tests`
Container simulation (no private research data): `GLUCOSE_TESTS_NO_RESEARCH_DATA=1 python -m pytest backend/tests`

| Files | Scope |
|---|---|
| `test_config.py` | settings from environment variables |
| `test_artifacts.py` | artifact integrity, versions, feature order, reproducibility, model card |
| `test_inference.py` | **causality**, **feature parity**, adapter, inference service |
| `test_api.py` | endpoints, validation, structured errors, start-up, CORS |
| `test_hardening.py` | rate limits, timeouts, upload limits, edge cases, logging, CORS |
| `test_pipeline_guards.py` | research-code drift, error paths, API contract, medical wording |

## Requirement → tests

| Requirement | Tests |
|---|---|
| Health endpoints | `test_api::test_health`, `test_pipeline_guards::test_responses_match_schemas` |
| Model loading once, hash + version verified, no fallback | `test_artifacts::test_loads_lightgbm`, `test_tampered_file_refused[5]`, `test_missing_*`, `test_unhashed_file_refused`, `test_no_fallback_to_other_model`, `test_library_version_mismatch_refused[4]`, `test_python_minor_mismatch_refused`, `test_api::test_startup_fails_without_valid_model` |
| Input validation (file, schema, numbers) | `test_api::test_upload_file_checks`, `test_upload_schema_errors`, `test_upload_bad_numbers_without_echoing_values[5]`, `test_hardening::test_malformed_csv[9]`, `test_nan_inf_and_non_numbers[8]`, `test_*_boundaries` |
| Timestamp handling | `test_api::test_upload_bad_timestamp[4]`, `test_hardening::test_timestamp_boundaries[8]`, `test_invalid_prediction_time[6]`, `test_pipeline_guards::test_timestamps_are_naive_iso8601`, `test_inference::test_prediction_time_must_be_on_grid` |
| Duplicate detection | `test_api::test_upload_duplicate_timestamp`, `test_hardening::test_duplicate_timestamps_per_signal` |
| Missing columns | `test_api::test_upload_schema_errors`, `test_inference::test_missing_columns` |
| Causal preprocessing (grid, Step A/B, gaps) | `test_inference::test_long_gap_inside_window_is_insufficient`, `test_short_gap_bridged_by_forward_fill`, `test_missing_current_glucose[6]`, `test_documented_event_bucket_difference` |
| **Causality (mandatory)** | `test_inference::test_causality_boundary_and_random_times` (118 times), `test_causality_real_ohio_data` (25), `test_placeholder_rows_affect_only_target`, `test_adapter_ignores_grid_rows_after_t`, `test_api::test_causality_end_to_end_via_api` |
| **Feature parity (mandatory)** | `test_inference::test_feature_parity_synthetic` (305 times), `test_feature_parity_real_ohio_file` (336), `test_context_window_does_not_change_features`, `test_artifacts::test_reproduces_recorded_research_predictions` |
| Feature generation / ordering | `test_artifacts::test_feature_list_equals_research_feature_order`, `test_pipeline_guards::test_feature_schema_mismatch`, `test_pipeline_constants_measured_from_create_features` |
| Training-serving skew guards | `test_pipeline_guards::test_research_code_unchanged[3]`, `test_heart_rate_is_never_available_and_imputed_like_training`, `test_inference::test_preprocessing_uses_transform_only`, `test_artifacts_unchanged_by_predictions`, `test_target_never_passed_to_model` |
| NaN / Inf detection | `test_inference::test_infinite_feature_rejected`, `test_pipeline_guards::test_unexpected_nan_feature`, `test_infinite_feature`, `test_nan_after_preprocessing`, `test_model_failure[3]` |
| Insufficient history | `test_inference::test_exact_minimum_history`, `test_api::test_upload_insufficient_history`, `test_hardening::test_insufficient_history_and_gaps` |
| Prediction output | `test_api::test_upload_success_matches_service`, `test_demo_prediction_matches_service`, `test_pipeline_guards::test_responses_match_schemas`, `test_single_forecast_no_recursion` |
| Demo endpoints | `test_api::test_demo_*`, `test_hardening::test_nonexistent_demo_patient[5]`, `test_missing_current_glucose_via_demo` |
| Upload endpoint | `test_api::test_upload_*`, `test_hardening::test_large_valid_input`, `test_malformed_multipart[6]` |
| Rate limits / timeouts / busy | `test_hardening::test_rate_limit*`, `test_forwarded_for_*`, `test_prediction_timeout`, `test_server_busy` |
| Upload size / compression / memory | `test_hardening::test_body_larger_than_limit_rejected_by_content_length`, `test_streamed_body_without_length_is_capped`, `test_file_just_over_limit_rejected_while_reading`, `test_compressed_*`, `test_json_body_limit_on_demo_endpoint`, `test_uploads_are_kept_in_memory`, `test_api::test_no_upload_persisted` |
| Privacy-safe errors and logs | `test_api::test_unexpected_error_is_generic`, `test_hardening::test_unexpected_exception_hidden`, `test_logs_are_structured_and_private`, `test_log_level_configurable` |
| CORS | `test_api::test_cors`, `test_production_rejects_wildcard_cors`, `test_hardening::test_production_cors_*`, `test_development_wildcard_is_logged` |
| Endpoint inventory / no medical claims | `test_pipeline_guards::test_endpoint_inventory`, `test_wrong_method_is_structured`, `test_no_medical_claims_in_public_text`, `test_api::test_model_endpoint` |

## Tests that need private research data (skipped in the container)
- `test_artifacts::test_reproduces_recorded_research_predictions` (HUPA-UCM data + stored predictions)
- `test_inference::test_causality_real_ohio_data`, `test_feature_parity_real_ohio_file` (OhioT1DM XML)

The synthetic causality and parity tests cover the same properties without private data.

## Not measured
Line coverage: no coverage tool is installed (`pytest-cov`/`coverage` absent); not added to keep dependencies unchanged.

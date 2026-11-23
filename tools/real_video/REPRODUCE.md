# Continuous video diagnostic reproduction

The aggregate CSV uses clip aliases. It contains threshold-response and motion-path counts for an unlabeled development batch. The scripts require locally authorized inputs and model files.

## Inputs

- A Windows installation of MATLAB R2024b with Deep Learning Toolbox and certutil. CPU inference uses an RGB uint8 image, single/255 normalization and foreground softmax channel 2.
- Python with NumPy, SciPy and Pillow. The measured setup was Python 3.7.0, NumPy 1.15.1, SciPy 1.1.0 and Pillow 5.2.0.
- A compatible research-code checkout with python/rail_motion_v3.py, python/rail_motion_conditional*.py and configs/algorithm_design_conditional_v3.json. The runner verifies the exact SHA-256 values in the protocol before fusion.
- The locked MobileNetV2–DeepLabv3+ seed 20261114 validation-selected MAT checkpoint. Verify its bytes against checkpoint_sha256 in the protocol template. The checkpoint payload is required locally.
- An input directory with the verified 256×256 RGB PNGs from three ordered clips and a frame_input_manifest.csv. Required columns are source_video, frame_index_zero_based, decoder_time_s, rgb_path and rgb_sha256. This batch uses 180 frames per clip and one additional stride-2 group for the last clip. Clip keys and input-relative paths use ordinary CSV-safe filenames.
- The frozen RGB-only v3 estimate trace per_frame_diagnostics.csv for the same ordered inputs; it also carries the prior frozen Small U-Net threshold counts. Required columns are source_video, stride, frame_index, previous_frame, decoder_time_s, edge_status, peak_boundary, peak_ncc, dx, dy, available_history_paths, valid_any_history_pixels, raw_positive_pixels, fused_positive_pixels, added_positive and removed_positive. Use an empty predecessor at each reset. An unavailable edge has empty dx and dy. The audit additionally reads analysis_summary.json with the trace's diagnostics_sha256.
- The prior 12-frame DeepLab scale-comparison cache for the continuity audit. Its prediction_manifest.csv uses video, frame, condition, input_path, probability_path and probability_sha256. The matching condition is deeplab14_320.

## Run

1. In an empty output directory copy protocol.template.json to protocol.json and fill its three clip keys with the matching source_video values. Keep the original ordering and aliases.
2. From MATLAB with the script on the path, invoke infer_continuous(runDir, inputDir, checkpointFile, expectedSHA256). MATLAB_PREFDIR can point to a writable directory with -nojvm -batch on Windows.
3. Run python -B analyze_continuous.py --run RUN_DIR --inputs INPUT_DIR --repo REPO_DIR.
4. Run python -B audit_continuous.py --run RUN_DIR --inputs INPUT_DIR --repo REPO_DIR --paired PAIRED_RUN_DIR. This verifies the saved outputs, prior 12 predictions, identical repeated input mappings and a stream replay prefix.

The main output aggregate_continuous.csv records foreground observations at 0.50 and the separate additions and removals relative to the single-frame predictions of the same segmentation model. Conditional writes count all pixels assigned the strong-fusion value, including those already above threshold under B. Conditional probability changes and B-to-conditional threshold flips are recorded separately. The fields tested_edges and rejected_edges describe transitions ending at the target frames in that row. For matched_even_targets at stride 1, a native transition's immediate predecessor remains the previous source frame. The matched rows reuse a subset of targets from the full groups.

The supplied aggregate CSV and report can be inspected directly. Exact numerical replay requires the local video-derived inputs, original estimate trace, prediction caches and checkpoint. The scope is unlabeled development behavior for these fixed clips.

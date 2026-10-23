# Scenario & Configuration Tuning Guide

| Preset File | Target Environment | Key Adjustments |
| :--- | :--- | :--- |
| `configs/default.yaml` | Balanced general-purpose video | `5 FPS` sampled, `auto` camera mode, `0.68` ReID threshold, `60s` return threshold |
| `configs/crowd.yaml` | High-density crowds, malls, stations | `8 FPS`, `min_observations: 3`, tighter NMS (`0.55`) & containment (`0.70`), `1.5s` grouping |
| `configs/low_light_cctv.yaml` | Nighttime, grainy, or IR CCTV | Lowered blur ref (`60.0`), lower detector conf (`0.30`), relaxed ReID (`0.62`), higher conf weight (`0.40`) |
| `configs/moving_handheld.yaml` | Smartphones, bodycams, drones | `10 FPS`, `moving` mode, relaxed spatial proximity (`0.30`), faster embedding refresh (`0.25s`) |

## Calibrating Key Thresholds
1. **Known people split into duplicate IDs:** Lower `reid.similarity_threshold` by `0.03` increments (`0.68` -> `0.65` -> `0.62`).
2. **Similar-looking people merged into one ID:** Increase `reid.similarity_threshold` to `0.72+` and raise `reid.ambiguous_margin` to `0.08`.
3. **Partial doorway appearances missed:** Lower `confirmation.min_crop_height` to `48` and increase `video.sample_fps` to `10.0`.
4. **Nighttime / soft footage rejected:** Lower `quality.blur_laplacian_ref` from `120.0` to `60.0`.

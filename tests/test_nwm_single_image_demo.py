import json
import math
import sys
import types

import numpy as np
import pytest
import torch
from PIL import Image

from demo_nwm_rollout import (
    FPS,
    _trajectory_points,
    encode_video,
    load_actions_file,
    make_preset_actions,
    render_contact_sheet,
    render_trajectory_panel,
    run_rollout,
    segments_to_actions,
)


def test_forward_preset_has_one_delta_per_4fps_frame() -> None:
    actions, metadata = make_preset_actions("forward", 16.0, 1.0, 7.5)

    assert actions.shape == (64, 3)
    np.testing.assert_allclose(actions[:, 0], 0.25)
    np.testing.assert_allclose(actions[:, 1:], 0.0)
    assert metadata["name"] == "forward"


def test_left_and_right_presets_turn_in_opposite_directions() -> None:
    left, _ = make_preset_actions("forward_then_left", 16.0, 1.0, 7.5)
    right, _ = make_preset_actions("forward_then_right", 16.0, 1.0, 7.5)

    assert left[:, 2].sum() == pytest.approx(math.pi / 2)
    assert right[:, 2].sum() == pytest.approx(-math.pi / 2)
    np.testing.assert_allclose(left[:, 0], right[:, 0])
    np.testing.assert_allclose(left[:, 1], -right[:, 1])


def test_four_second_turn_presets_turn_after_one_second() -> None:
    left, _ = make_preset_actions("forward_then_left", 4.0, 1.0, 7.5)
    right, _ = make_preset_actions("forward_then_right", 4.0, 1.0, 7.5)

    np.testing.assert_allclose(left[:4, 2], 0.0)
    assert np.all(left[4:, 2] > 0)
    assert np.all(right[4:, 2] < 0)
    np.testing.assert_allclose(left[:, 1], -right[:, 1])


def test_segments_support_body_frame_lateral_velocity() -> None:
    actions = segments_to_actions(
        [
            {
                "duration_seconds": 0.25,
                "forward_speed": 0.0,
                "lateral_speed": 2.0,
                "yaw_rate_deg": 0.0,
            }
        ]
    )

    np.testing.assert_allclose(actions, [[0.0, 0.5, 0.0]], atol=1e-7)


def test_custom_meter_actions_are_converted_to_waypoint_units(tmp_path) -> None:
    path = tmp_path / "actions.json"
    path.write_text(
        json.dumps(
            {
                "fps": FPS,
                "translation_unit": "meters",
                "waypoint_spacing_meters": 0.25,
                "yaw_unit": "degrees",
                "actions": [[0.25, 0.0, 90.0]],
            }
        ),
        encoding="utf-8",
    )

    actions, metadata = load_actions_file(path, expected_frames=1)

    np.testing.assert_allclose(actions, [[1.0, 0.0, math.pi / 2]], rtol=1e-6)
    assert metadata["input_translation_unit"] == "meters"


def test_paper_sheet_selects_2_through_16_seconds(tmp_path) -> None:
    first = Image.new("RGB", (32, 32), (255, 0, 0))
    generated = [Image.new("RGB", (32, 32), (index, 0, 0)) for index in range(64)]
    output = tmp_path / "sheet.png"

    trajectory = Image.new("RGB", (32, 32), (255, 255, 255))
    seconds = render_contact_sheet(trajectory, first, generated, output)

    assert seconds == [float(value) for value in range(2, 17)]
    assert output.is_file()
    assert Image.open(output).size == (224 * 17, 262)


def test_four_second_sheet_includes_each_one_second_snapshot(tmp_path) -> None:
    first = Image.new("RGB", (32, 32), (255, 0, 0))
    generated = [Image.new("RGB", (32, 32), (index, 0, 0)) for index in range(16)]
    output = tmp_path / "sheet.png"
    trajectory = Image.new("RGB", (32, 32), (255, 255, 255))

    seconds = render_contact_sheet(trajectory, first, generated, output)

    assert seconds == [1.0, 2.0, 3.0, 4.0]
    assert Image.open(output).size == (224 * 6, 262)


def test_action_trajectory_is_yellow_on_white_with_fixed_origin() -> None:
    actions, _ = make_preset_actions("forward_then_left", 16.0, 1.0, 7.5)

    panel = render_trajectory_panel(actions)
    points = _trajectory_points(actions, panel.size)

    assert panel.size == (224, 224)
    assert points[0] == (112, 170)
    assert points[16][1] < points[0][1]  # forward is up
    assert points[-1][0] < points[16][0]  # positive-y/left turn is left
    pixels = np.asarray(panel)
    assert np.all(pixels[0, 0] == 255)
    yellowish = (pixels[..., 0] > 220) & (pixels[..., 1] > 160) & (pixels[..., 2] < 100)
    assert yellowish.sum() > 100


def test_complete_rollout_video_uses_generated_frames(tmp_path) -> None:
    frames = tmp_path / "frames"
    frames.mkdir()
    for index in range(1, 5):
        Image.new("RGB", (32, 32), (index * 20, 0, 0)).save(
            frames / f"frame_{index:04d}.png"
        )
    output = tmp_path / "rollout.mp4"

    encode_video(frames, output, frame_count=4)

    assert output.is_file()
    assert output.stat().st_size > 0


def test_rollout_decodes_normalizes_and_reencodes_pixel_context(monkeypatch) -> None:
    seen_contexts = []

    def fake_forward(_models, context, _action, **_kwargs):
        seen_contexts.append(context.detach().clone())
        return torch.full((1, 3, 2, 2), 0.75, device=context.device)

    monkeypatch.setitem(
        sys.modules,
        "isolated_nwm_infer",
        types.SimpleNamespace(model_forward_wrapper=fake_forward),
    )
    monkeypatch.setitem(
        sys.modules,
        "misc",
        types.SimpleNamespace(get_normalize=lambda _mean, _std: lambda image: image * 2 - 1),
    )

    generated = run_rollout(
        model=object(),
        diffusion=object(),
        tokenizer=object(),
        first_image=torch.zeros(3, 2, 2),
        actions=np.zeros((2, 3), dtype=np.float32),
        context_size=2,
        device=torch.device("cpu"),
        seed=0,
        image_mean=[0.5, 0.5, 0.5],
        image_std=[0.5, 0.5, 0.5],
    )

    assert len(generated) == 2
    torch.testing.assert_close(seen_contexts[0], torch.zeros(1, 2, 3, 2, 2))
    torch.testing.assert_close(seen_contexts[1][:, -1], torch.full((1, 3, 2, 2), 0.5))


def test_latent_rollout_encodes_initial_context_only_once(monkeypatch) -> None:
    seen_contexts = []

    def fake_forward(_models, context, _action, **kwargs):
        assert kwargs["skip_tokenizer"] is True
        seen_contexts.append(context.detach().clone())
        value = 0.25 * len(seen_contexts)
        return torch.full((1, 1, 2, 2), value, device=context.device)

    class FakeTokenizer:
        def __init__(self):
            self.encode_calls = 0

        def encode(self, images):
            self.encode_calls += 1
            return images[:, :1]

        def decode(self, latents, denormalize=True):
            assert denormalize is True
            return latents.repeat(1, 3, 1, 1)

    monkeypatch.setitem(
        sys.modules,
        "isolated_nwm_infer",
        types.SimpleNamespace(model_forward_wrapper=fake_forward),
    )
    monkeypatch.setitem(
        sys.modules,
        "misc",
        types.SimpleNamespace(get_normalize=lambda _mean, _std: lambda image: image),
    )
    tokenizer = FakeTokenizer()
    generated = run_rollout(
        model=object(),
        diffusion=object(),
        tokenizer=tokenizer,
        first_image=torch.zeros(3, 2, 2),
        actions=np.zeros((2, 3), dtype=np.float32),
        context_size=2,
        device=torch.device("cpu"),
        seed=0,
        image_mean=[0.5, 0.5, 0.5],
        image_std=[0.5, 0.5, 0.5],
        feedback_mode="latent",
    )

    assert len(generated) == 2
    assert tokenizer.encode_calls == 1
    torch.testing.assert_close(seen_contexts[0], torch.zeros(1, 2, 1, 2, 2))
    torch.testing.assert_close(seen_contexts[1][:, -1], torch.full((1, 1, 2, 2), 0.25))

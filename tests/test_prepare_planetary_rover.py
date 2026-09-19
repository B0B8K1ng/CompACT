"""CPU-only contracts for the real Mars + Moon evaluation dataset builder."""

import copy
import io
import json
import math
import pickle

import numpy as np
from PIL import Image

from misc import get_delta_np
from scripts.prepare_planetary_rover import (
    MarsLocalizer,
    ce4_rover_yaw,
    circular_std_degrees,
    parse_mars_label,
    process_source_image,
    scale_moon_evaluation_positions,
    segment_mars_frames,
    select_mars_index,
    sha256_file,
    validate_dataset,
)


def _places_row(frame, site, drive, pose, northing, easting):
    return {
        "frame": frame,
        "site": str(site),
        "drive": str(drive),
        "pose": str(pose),
        "northing": str(northing),
        "easting": str(easting),
    }


def test_parse_mars_telemetry_pose_and_rover_camera_geometry(tmp_path):
    half_sqrt = math.sqrt(0.5)
    label = tmp_path / "navcam.xml"
    label.write_text(
        f"""<?xml version="1.0"?>
        <Product xmlns:g="urn:geometry" xmlns:m="urn:mission">
          <m:spacecraft_clock_start>700000000.25</m:spacecraft_clock_start>
          <g:Coordinate_Space_Definition>
            <local_identifier>ROVER_NAV_FRAME_7_42_3_0_0_0_0_0_0_0</local_identifier>
            <local_identifier>ROVER_NAV_FRAME_7_42_3_0_0_0_0_0_0_0_TELEMETRY</local_identifier>
            <g:Coordinate_Space_Indexed>
              <g:coordinate_space_frame_type>ROVER_NAV_FRAME</g:coordinate_space_frame_type>
              <g:Coordinate_Space_Index><g:index_id>SITE</g:index_id><g:index_value_number>7</g:index_value_number></g:Coordinate_Space_Index>
              <g:Coordinate_Space_Index><g:index_id>DRIVE</g:index_id><g:index_value_number>42</g:index_value_number></g:Coordinate_Space_Index>
              <g:Coordinate_Space_Index><g:index_id>POSE</g:index_id><g:index_value_number>3</g:index_value_number></g:Coordinate_Space_Index>
              <g:solution_id>TELEMETRY</g:solution_id>
            </g:Coordinate_Space_Indexed>
            <g:Vector_Origin_Offset>
              <g:x_position>1.25</g:x_position><g:y_position>-2.5</g:y_position><g:z_position>0.75</g:z_position>
            </g:Vector_Origin_Offset>
            <g:Quaternion_Plus_Direction>
              <g:qcos>{half_sqrt}</g:qcos><g:qsin1>0</g:qsin1><g:qsin2>0</g:qsin2><g:qsin3>{half_sqrt}</g:qsin3>
            </g:Quaternion_Plus_Direction>
          </g:Coordinate_Space_Definition>
          <g:Derived_Geometry>
            <local_identifier_reference>ROVER_NAV_FRAME_7_42_3_0_0_0_0_0_0_0</local_identifier_reference>
            <g:instrument_azimuth>-1.25</g:instrument_azimuth>
            <g:instrument_elevation>2.5</g:instrument_elevation>
          </g:Derived_Geometry>
        </Product>
        """,
        encoding="utf-8",
    )

    parsed = parse_mars_label(label)

    assert (parsed["site"], parsed["drive"], parsed["pose"]) == (7, 42, 3)
    np.testing.assert_allclose(parsed["origin_site"], [1.25, -2.5, 0.75])
    assert math.isclose(parsed["yaw_rad"], math.pi / 2, abs_tol=1e-12)
    assert parsed["instrument_azimuth_deg"] == -1.25
    assert parsed["instrument_elevation_deg"] == 2.5
    assert parsed["sclk"] == 700000000.25


def test_mars_localizer_uses_exact_places_or_local_similarity():
    telemetry = [
        _places_row("SITE", 1, -1, -1, 100.0, 200.0),
        _places_row("ROVER", 1, 0, 2, 100.0, 200.0),
        _places_row("ROVER", 1, 10, 2, 110.0, 200.0),
    ]
    best_interp = [
        _places_row("ROVER", 1, 0, -1, 1000.0, 2000.0),
        _places_row("ROVER", 1, 10, -1, 1000.0, 2020.0),
    ]
    localizer = MarsLocalizer(telemetry, best_interp)

    exact, exact_provenance = localizer.localize(
        {"site": 1, "drive": 0, "origin_site": [999.0, 999.0, 0.0]}
    )
    inferred, inferred_provenance = localizer.localize(
        {"site": 1, "drive": 5, "origin_site": [5.0, 0.0, 0.0]}
    )

    np.testing.assert_allclose(exact, [1000.0, 2000.0])
    assert exact_provenance["method"] == "exact_best_interp_rmc"
    np.testing.assert_allclose(inferred, [1000.0, 2010.0], atol=1e-12)
    assert inferred_provenance["lower_best_interp_rmc"] == [1, 0]
    assert inferred_provenance["upper_best_interp_rmc"] == [1, 10]


def test_ce4_official_panorama_recovers_one_rover_heading():
    # Five official left-eye labels from CLEP sequence 0362.  The camera pans,
    # while the recovered rover attitude must remain fixed.
    observations = [
        (-172.891418, [-0.365342, 0.930332, -0.031733]),
        (-160.124512, [-0.562010, 0.826658, -0.027935]),
        (-147.225784, [-0.732460, 0.680415, -0.023182]),
        (-134.129288, [-0.867669, 0.496829, -0.017661]),
        (-121.142654, [-0.957190, 0.289220, -0.011775]),
    ]
    headings = [
        ce4_rover_yaw(np.asarray(center), mast_yaw) for mast_yaw, center in observations
    ]

    assert circular_std_degrees(headings) < 0.003
    np.testing.assert_allclose(headings, math.radians(-75.6669278), atol=5.1e-5)


def test_mars_segmentation_enforces_site_time_translation_and_yaw_limits():
    def frame(index, *, site=1, x=None, sclk=None, yaw=0.0):
        return {
            "name": str(index),
            "site": site,
            "position": [float(index if x is None else x), 0.0],
            "sclk": float(index if sclk is None else sclk),
            "yaw": yaw,
        }

    frames = [
        frame(0, yaw=math.radians(179)),
        frame(1, yaw=math.radians(-179)),  # wrapped yaw step is only two degrees
        frame(2, x=7.0),  # translation > 5 m
        frame(3, x=8.0, site=2),  # site change
        frame(4, x=9.0, site=2, sclk=3.0),  # non-positive clock gap
        frame(5, x=10.0, site=2, sclk=1_000_000.0),  # clock gap > threshold
        frame(6, x=11.0, site=2, sclk=1_000_001.0, yaw=math.radians(121)),
    ]

    assert [len(part) for part in segment_mars_frames(frames)] == [2, 1, 1, 1, 1, 1]


def test_mars_selection_requires_an_official_published_release():
    released = {
        "atlas_id": (
            "atlas:pds4:mars_2020:perseverance:/mars2020_navcam_ops_raw/"
            "data/sol/00065/ids/edr/ncam/"
            "NLM_0065_0672718757_551EDR_N0032059TRAV12002_00_2LLJ02.IMG::8"
        ),
        "source": {
            "archive": {
                "name": "NLM_0065_0672718757_551EDR_N0032059TRAV12002_00_2LLJ02.IMG"
            },
            "gather": {
                "landed_missions": {"site_instrument_azimuth": 0.1},
                "pds_archive": {
                    "related": {
                        "label": {"uri": "atlas:/label.xml"},
                        "browse": {"uri": "atlas:/browse.png"},
                    }
                },
            },
        },
    }
    unversioned = copy.deepcopy(released)
    unversioned["atlas_id"] = unversioned["atlas_id"].removesuffix("::8")

    assert select_mars_index([released, unversioned]) == [released]
    assert select_mars_index([released, unversioned], require_release=False) == [
        released,
        unversioned,
    ]


def test_lunar_position_scale_is_reversible_and_matches_mars_step():
    frames = [
        {"position": [10.0, 20.0], "name": "a"},
        {"position": [12.0, 20.0], "name": "b"},
        {"position": [16.0, 20.0], "name": "c"},
    ]

    scaled, transform = scale_moon_evaluation_positions(frames, 1.0)
    positions = np.asarray([frame["position"] for frame in scaled])
    raw = positions / transform["scale"] + np.asarray(transform["origin_xy_m"])

    assert transform["scale"] == 1.0 / 3.0
    np.testing.assert_allclose(raw, [frame["position"] for frame in frames])
    np.testing.assert_allclose(
        np.median(np.linalg.norm(np.diff(positions, axis=0), axis=1)), 1.0
    )
    assert all("raw_position_xy_m" in frame for frame in scaled)
    assert not transform["images_or_poses_interpolated"]


def test_lunar_unsigned_lsb2_uses_fixed_ten_to_eight_bit_mapping():
    values = np.asarray([[0, 4], [1020, 1023]], dtype="<u2")
    jpeg, processing = process_source_image(
        {
            "source": "ce4_yutu2_pcam",
            "data_type": "UnsignedLSB2",
            "lines": 2,
            "samples": 2,
        },
        values.tobytes(),
    )

    with Image.open(io.BytesIO(jpeg)) as image:
        assert image.size == (224, 224)
        assert image.mode == "RGB"
    assert processing["source_dn_min"] == 0
    assert processing["source_dn_max"] == 1023
    assert processing["display_conversion"] == "uint8 = uint16_dn >> 2"
    assert not processing["scene_dependent_stretch"]


def test_wrapped_delta_yaw_is_opt_in_and_uses_shortest_angle():
    actions = np.asarray(
        [[0.0, 0.0, math.radians(179.0)], [1.0, 0.0, math.radians(-179.0)]]
    )

    legacy = get_delta_np(actions)
    wrapped = get_delta_np(actions, wrap_yaw=True)

    assert math.isclose(legacy[1, 2], math.radians(-358.0), abs_tol=1e-12)
    assert math.isclose(wrapped[1, 2], math.radians(2.0), abs_tol=1e-12)


def test_validate_dataset_checks_compatible_layout_and_provenance(tmp_path):
    output = tmp_path / "planetary_rover"
    trajectory = output / "mars_test_0000"
    trajectory.mkdir(parents=True)
    count = 68
    position = np.column_stack([np.arange(count, dtype=float), np.zeros(count)])
    yaw = np.linspace(-0.1, 0.1, count)
    with (trajectory / "traj_data.pkl").open("wb") as stream:
        pickle.dump({"position": position, "yaw": yaw}, stream, protocol=4)
    image = Image.new("RGB", (224, 224), (10, 20, 30))
    for index in range(count):
        image.save(trajectory / f"{index}.jpg")
    processed_sha256 = sha256_file(trajectory / "0.jpg")
    rows = [
        {
            "frame": index,
            "source": "mars2020_perseverance_navcam",
            "position": position[index].tolist(),
            "yaw": float(yaw[index]),
            "label_url": "https://pds.example/label.xml",
            "label_sha256": "a" * 64,
            "image_url": "https://pds.example/image.png",
            "source_image_sha256": "b" * 64,
            "processed_jpeg_sha256": processed_sha256,
        }
        for index in range(count)
    ]
    (trajectory / "frame_metadata.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    split = tmp_path / "traj_names.txt"
    split.write_text("mars_test_0000\n", encoding="utf-8")

    report = validate_dataset(output, split)

    assert report["dataset"] == "planetary_rover"
    assert report["trajectories"] == 1
    assert report["frames"] == 68
    assert report["anchors_context4_horizon64"] == 1
    assert report["source_frames"] == {"mars2020_perseverance_navcam": 68}

import numpy as np

from acoustics.svp_perturbation import (
    SEABED_DEPTH_M,
    SETTINGS,
    mode1_shape,
    named_perturbation,
    perturb_profile,
)

# Cast-like profile: warm thermocline over a deep pressure gradient of 0.0165 m/s/m.
DEPTH = np.arange(0.0, 5401.0, 5.0)
C = 1500.0 + 40.0 * np.exp(-DEPTH / 500.0) + 0.0165 * DEPTH


def test_zero_amplitude_returns_reference_profile():
    np.testing.assert_array_equal(perturb_profile(DEPTH, C), C)
    np.testing.assert_array_equal(perturb_profile(DEPTH, C, 0.0, 0.0), C)


def test_mode_shape_vanishes_at_surface_and_seabed_and_peaks_at_one():
    phi = mode1_shape(DEPTH)

    assert phi[0] == 0.0
    assert np.all(phi[DEPTH >= SEABED_DEPTH_M] == 0.0)
    assert phi.max() == 1.0
    assert 500.0 < DEPTH[np.argmax(phi)] < 2000.0


def test_heave_leaves_surface_and_seabed_unchanged():
    c_perturbed = perturb_profile(DEPTH, C, heave_amplitude_m=25.0)

    assert c_perturbed[0] == C[0]
    np.testing.assert_array_equal(
        c_perturbed[DEPTH >= SEABED_DEPTH_M], C[DEPTH >= SEABED_DEPTH_M]
    )
    assert np.abs(c_perturbed - C).max() > 0.1


def test_deep_water_changes_by_much_less_than_the_thermocline():
    delta_c = perturb_profile(DEPTH, C, heave_amplitude_m=25.0) - C

    assert np.abs(delta_c[DEPTH > 3000]).max() < 0.02 * np.abs(delta_c).max()


def test_mixed_layer_term_is_confined_to_the_mixed_layer():
    delta_c = perturb_profile(DEPTH, C, mixed_layer_dT=-0.3) - C

    assert delta_c[0] == -0.75
    assert np.all(delta_c[DEPTH >= 60.0] == 0.0)


def test_named_settings_match_their_parameters():
    for name, parameters in SETTINGS.items():
        np.testing.assert_array_equal(
            named_perturbation(name, DEPTH, C), perturb_profile(DEPTH, C, **parameters)
        )


def test_invalid_inputs_raise():
    for call in (
        lambda: perturb_profile(DEPTH, C[:-1]),
        lambda: named_perturbation("extreme", DEPTH, C),
    ):
        try:
            call()
        except ValueError:
            continue
        raise AssertionError("expected ValueError")

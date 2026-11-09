"""Shared small pieces of the PILOT: the image-only illumination rule (an UNCALIBRATED baseline, definitions ours) and the period mapping.
Labels are build-time only; none of this is an inference input."""
ILLUM = ("dark", "lit_night", "twilight", "day", "glare")
TOD = ("night", "dawn_dusk", "day")                      # rough time of day, as the PI sees it
SUN_TO_TOD = {"night": "night", "twilight": "dawn_dusk", "low_sun": "day", "day": "day"}
ILLUM_TO_TOD = {"dark": "night", "lit_night": "night", "twilight": "dawn_dusk", "day": "day", "glare": "day"}


def illum_rule(luma_mean, luma_p95, sky_luma, headlight_frac, lit_frac):
    """Image-only illumination from Tier 1 statistics (0..255 luma, fractions). Thresholds are a first guess, not fitted to anything.
    glare: a large blown-out area under a bright sky; day: bright sky; sky below 60 is night (lit_night when lamps or lit area are present, else dark);
    in between, twilight when the road is not dark, else night."""
    if headlight_frac >= 0.10 and sky_luma >= 150:
        return "glare"
    if sky_luma >= 120:
        return "day"
    if sky_luma < 60:
        return "lit_night" if (lit_frac >= 0.05 or luma_p95 >= 160) else "dark"
    if luma_mean >= 40:
        return "twilight"
    return "lit_night" if (lit_frac >= 0.05 or luma_p95 >= 160) else "dark"

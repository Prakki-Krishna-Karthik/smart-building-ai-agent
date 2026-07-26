"""Unit tests for EnergyPlus output parsing using representative sample files."""

from pathlib import Path

from src.simulator.output_parser import EnergyPlusOutputParser


SAMPLE_CSV = """Date/Time,Electricity:Facility [J](Hourly),HVAC Electricity [J](Hourly),Lighting Electricity [J](Hourly),Equipment Electricity [J](Hourly),Environment:Site Outdoor Air Drybulb Temperature [C](Hourly),Zone One:Zone Mean Air Temperature [C](Hourly),Zone One:Zone Air Relative Humidity [%](Hourly),Zone One:Zone Thermal Comfort Fanger Model PMV [](Hourly),Zone One:Zone Thermal Comfort Fanger Model PPD [%](Hourly),Zone One:People Occupant Count [people](Hourly),Zone One:HVAC Operating State [](Hourly),Zone Two:Zone Mean Air Temperature [C](Hourly),Zone Two:People Occupant Count [people](Hourly)
01/01 01:00:00,100,40,20,30,5,21,45,0.1,5,2,On,20,0
01/01 02:00:00,120,50,25,35,6,22,50,0.2,6,3,Off,21,0
"""


def write_standard_outputs(directory: Path) -> None:
    """Write a representative successful EnergyPlus output set."""
    directory.mkdir()
    (directory / "eplusout.csv").write_text(SAMPLE_CSV, encoding="utf-8")
    (directory / "eplusout.err").write_text("** Warning ** Warmup convergence took extra iterations\n", encoding="utf-8")
    (directory / "eplusout.end").write_text(
        "EnergyPlus Completed Successfully-- 01/01 02:00\nElapsed Time = 00:00:12.50\n",
        encoding="utf-8",
    )
    (directory / "eplusout.eso").write_text("! sample ESO header\n1,2,3\n2,4,5\n", encoding="utf-8")


def test_parser_extracts_building_state(tmp_path: Path) -> None:
    """The parser should extract energy, thermal, comfort, and building data."""
    output = tmp_path / "run-001"
    write_standard_outputs(output)

    state = EnergyPlusOutputParser().parse(output)

    assert state.simulation_success is True
    assert len(state.simulation_warnings) == 1
    assert state.simulation_errors == ()
    assert state.simulation_duration_seconds == 12.5
    assert state.energy.total_electricity_consumption == 220.0
    assert state.energy.hvac_electricity == 90.0
    assert state.energy.lighting_electricity == 45.0
    assert state.energy.equipment_electricity == 65.0
    assert state.thermal.zone_temperatures == {"Zone One": 22.0, "Zone Two": 21.0}
    assert state.thermal.outdoor_temperature == 6.0
    assert state.thermal.zone_humidity == {"Zone One": 50.0}
    assert state.comfort.pmv == {"Zone One": 0.2}
    assert state.comfort.ppd == {"Zone One": 6.0}
    assert state.zone_names == ("Zone One", "Zone Two")
    assert state.occupied_zones == ("Zone One",)
    assert state.hvac_operating_state == {"Zone One:HVAC Operating State": "Off"}
    assert state.eso_available is True
    assert state.eso_record_count == 2


def test_parser_handles_missing_columns_and_failed_run(tmp_path: Path) -> None:
    """Missing optional data should become typed defaults, not parser failures."""
    output = tmp_path / "run-002"
    output.mkdir()
    (output / "eplusout.csv").write_text("Date/Time,Some Unrelated Metric\n01/01 01:00:00,3\n", encoding="utf-8")
    (output / "eplusout.err").write_text("** Severe  ** A required object is invalid\n** Fatal  ** Simulation aborted\n", encoding="utf-8")
    (output / "eplusout.end").write_text("EnergyPlus terminated before completion\n", encoding="utf-8")

    state = EnergyPlusOutputParser().parse(output)

    assert state.simulation_success is False
    assert len(state.simulation_errors) == 2
    assert state.energy.total_electricity_consumption is None
    assert state.thermal.zone_temperatures == {}
    assert state.comfort.pmv == {}
    assert state.zone_names == ()
    assert state.eso_available is False


def test_parser_handles_completely_empty_output_directory(tmp_path: Path) -> None:
    """A parser call immediately after an incomplete run should still return state."""
    state = EnergyPlusOutputParser().parse(tmp_path)

    assert state.simulation_success is None
    assert state.simulation_warnings == ()
    assert state.simulation_errors == ()
    assert state.source_files == ()


def test_zero_severity_summary_and_missing_end_do_not_fail_successful_process(tmp_path: Path) -> None:
    """A zero-severity ERR summary and absent END file are not failures."""
    (tmp_path / "eplusout.err").write_text(
        "Program Version,EnergyPlus, Version 26.1.0\n"
        "EnergyPlus Completed Successfully-- 0 Warning; 0 Severe Errors.\n",
        encoding="utf-8",
    )

    state = EnergyPlusOutputParser().parse(tmp_path, return_code=0)

    assert state.simulation_success is True
    assert state.simulation_warnings == ()
    assert state.simulation_errors == ()


def test_genuine_fatal_diagnostic_fails_even_with_zero_process_code(tmp_path: Path) -> None:
    """A real Severe/Fatal diagnostic remains a validation failure."""
    (tmp_path / "eplusout.err").write_text(
        "** Severe ** Invalid object in input\n** Fatal ** Simulation aborted\n",
        encoding="utf-8",
    )

    state = EnergyPlusOutputParser().parse(tmp_path, return_code=0)

    assert state.simulation_success is False
    assert len(state.simulation_errors) == 2


def test_nonzero_process_code_fails_without_diagnostic_file(tmp_path: Path) -> None:
    """A non-zero EnergyPlus process code is always unsuccessful."""
    state = EnergyPlusOutputParser().parse(tmp_path, return_code=1)

    assert state.simulation_success is False
    assert state.simulation_errors == ()

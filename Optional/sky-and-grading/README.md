# Optional skyboxes and colour grading

This asset-free kit builds an optional MPQ on your machine. It adds zone
skyboxes for Burning Steppes, Blasted Lands, and Mount Hyjal; project-authored
per-zone colour grading for Elwynn Forest, Duskwood, Westfall, and Redridge;
and, with `--wf-colours`, a moderated import of night colours from your own WoW
Forever installation.

The repository contains only scripts and benilla's own grading LUTs/table. The
Blizzard models, textures, and lighting tables are read from installations you
own and appear only in your local build directory.

## Requirements

- Windows and Python 3.8 or newer.
- A World of Warcraft 1.12.1 client and its `Data` directory.
- For skyboxes, a local WoW Forever `_classic_beta_` or current retail install.
  Point the command at the shared World of Warcraft root (the directory with
  `.build.info` and `Data`), or at `_classic_beta_` itself.
- Git and the external `wowdev/pywowlib` checkout pinned below. It is not
  redistributed here.
- 64-bit StormLib 9.40 (`StormLib.dll`) from the official
  [StormLib releases](https://github.com/ladislav-zezula/StormLib/releases/tag/v9.40).
  Download `stormlib_dll.zip` and use its x64 DLL. StormLib is not redistributed.
- Internet access on the first run for the public wow-listfile and public TACT
  key list, unless `--listfile` and `--tact-keys` name local copies.

## Install, one command per step

Run these in PowerShell. Change the example paths for your machine.

1. Clone the exact pywowlib revision used by the converter.

   ```powershell
   git clone https://github.com/wowdev/pywowlib.git C:\Tools\pywowlib
   ```

2. Pin it to the tested commit.

   ```powershell
   git -C C:\Tools\pywowlib checkout 55276dc5c2195da7fe136638a2a59716622f8c65
   ```

3. Install the Python modules needed by the M2 reader. pywowlib remains an
   external checkout; this installs dependencies only.

   ```powershell
   py -3 -m pip install bidict==0.23.1 multimethod numpy
   ```

4. Download StormLib 9.40's `stormlib_dll.zip` from the release linked above,
   extract the 64-bit `StormLib.dll`, and place it at
   `C:\Tools\StormLib\StormLib.dll`.

5. Close the game and build the candidate. The tool never writes to either
   input installation.

   ```powershell
   py -3 .\tools\build_sky_patch.py --wow-forever "C:\Games\World of Warcraft" --client-data "C:\Games\WoW-1.12.1\Data" --pywowlib "C:\Tools\pywowlib" --stormlib "C:\Tools\StormLib\StormLib.dll" --out "C:\Temp\benilla-sky"
   ```

6. Verify an isolated Data copy before installation. `C:\Temp\WoW-test\Data`
   should be a copy or hard-link mirror in which only `patch-Z.mpq` is an
   ordinary copy; replace that copy with the newly built archive first.

   ```powershell
   py -3 .\tools\verify_client_chain.py "C:\Temp\WoW-test\Data" --stage "C:\Temp\benilla-sky\stage" --stormlib "C:\Tools\StormLib\StormLib.dll"
   ```

7. Back up the live target archive if one exists.

   ```powershell
   Copy-Item "C:\Games\WoW-1.12.1\Data\patch-Z.mpq" "C:\Games\WoW-1.12.1\Data\patch-Z.before-benilla.mpq.bak"
   ```

8. Copy the verified candidate into the client.

   ```powershell
   Copy-Item -Force "C:\Temp\benilla-sky\patch-Z.MPQ" "C:\Games\WoW-1.12.1\Data\patch-Z.mpq"
   ```

If no `patch-Z.mpq` existed, skip step 7. The build report's `archive_base`
field records whether the builder preserved an existing patch-Z.

### Variants

Build only the project-owned grading table and LUTs; this needs no modern WoW
install or pywowlib:

```powershell
py -3 .\tools\build_sky_patch.py --grading-only --client-data "C:\Games\WoW-1.12.1\Data" --stormlib "C:\Tools\StormLib\StormLib.dll" --out "C:\Temp\benilla-grading"
```

Add moderated WoW Forever night colours. LightData is extracted directly from
your installation; no derived LightData is shipped here:

```powershell
py -3 .\tools\build_sky_patch.py --wow-forever "C:\Games\World of Warcraft" --client-data "C:\Games\WoW-1.12.1\Data" --pywowlib "C:\Tools\pywowlib" --stormlib "C:\Tools\StormLib\StormLib.dll" --wf-colours --out "C:\Temp\benilla-sky-wf"
```

## Enable in game

Open **Options → Advanced Graphics** and enable **Zone Skyboxes** and
**Colour Grading**. The equivalent console commands are:

```text
/console zoneSkyboxes 1
/console colorGrading 1
```

Use `0` instead of `1` to disable either effect without removing the MPQ.

## MPQ load order: read before copying

All five mutually dependent lighting DBCs must win from the same, highest
letter patch that carries any of them. The default is `patch-Z.mpq`. If the
client already has patch-Z, the builder copies that archive and overlays the
new files so unrelated members survive; install the resulting candidate as a
replacement, not as a second archive.

Do not blindly rename the output. A later-mounted archive that also contains
`Light.dbc`, `LightParams.dbc`, `LightIntBand.dbc`, `LightFloatBand.dbc`, or
`LightSkybox.dbc` can split the table set and break the sky. Some custom clients
use punctuation after ordinary letters; in that case merge this stage into the
actual last patch or remove the conflicting DBC copies, then run
`verify_client_chain.py --require-winner ANY`. A conventional patch-Z setup
should keep the default strict check.

The extended four-field `LightSkybox.dbc` is for benilla. Do not use this patch
with an unmodified stock 1.12 executable that expects the two-field table.

## Uninstall

Close the game. If the build preserved an existing patch-Z, restore the backup:

```powershell
Move-Item -Force "C:\Games\WoW-1.12.1\Data\patch-Z.before-benilla.mpq.bak" "C:\Games\WoW-1.12.1\Data\patch-Z.mpq"
```

If there was no prior patch-Z and this kit created it, remove only that exact
archive:

```powershell
Remove-Item -LiteralPath "C:\Games\WoW-1.12.1\Data\patch-Z.mpq"
```

Deleting a pre-existing patch-Z would remove unrelated custom content, so use
`build-report.json` and your backup to distinguish these cases.

## Troubleshooting

- **pywowlib revision mismatch:** run the checkout command from step 2. The
  builder deliberately rejects another revision so M2 layout behaviour cannot
  drift silently.
- **StormLib not found:** pass `--stormlib` or set `STORMLIB_PATH` to the x64
  DLL. A 32-bit DLL cannot load into 64-bit Python.
- **No CASC build/config:** point `--wow-forever` at the World of Warcraft root
  with `.build.info` and `Data`, or at its `_classic_beta_` child.
- **Missing/encrypted CASC block:** update the public TACT key file or rerun
  without a stale `--tact-keys` copy. The tool uses public keys only.
- **Unexpected source schema/clone start:** another addon has already changed
  the winning lighting DBCs. Remove that conflict or merge deliberately; the
  tool refuses to guess.
- **Sky does not appear:** enable `zoneSkyboxes`, fully restart the client, and
  run `verify_client_chain.py` against the actual Data directory.
- **Bands fail:** run `check_bands.py <Data directory> --stormlib <DLL>`. Correct
  clone keys are `(P-1)*18+b+1` and `(P-1)*6+b+1`.
- **Output already exists:** choose a fresh `--out`, or pass `--force` only when
  replacing the candidate archive in that output directory is intended.

## Credits and rights

WoW Forever and retail skybox art is © Blizzard Entertainment and is extracted
locally by the user from their own installation. No Blizzard art or client DBC
is distributed here.

The colour-grading technique is based on WarcraftXL's `wxl-retail-grading` work
by iThorgrim, used with attribution. The grading LUTs and `MonkeyZoneGrade.dbc`
in `data/` are original benilla project data, licensed MIT OR Apache-2.0.

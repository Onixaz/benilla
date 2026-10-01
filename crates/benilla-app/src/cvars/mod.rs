//! The CVar registry, host side: the reference's engine-side table (`ConsoleVar.cpp`, 0xc4-byte
//! records of name, value, default, latch slot and the change callback handed to `CVar::Register`,
//! `0x63db90`). [`Cvars`] holds every registered row's live value, survives a VM replacement and is
//! what `config.toml` is composed from; the script VM keeps a mirror for Lua's synchronous
//! `GetCVar`/`SetCVar` ([`benilla_ui::script::UiScript::seed_cvars`]) whose writes queue back here.
//!
//! - [`REGISTERED`] holds only vars something reads, a host knob or a Lua consumer. A row's default
//!   is the reference's, and [`table::Registered::reference`] says where it stands against it.
//! - The change callback is a Bevy observer on [`CvarChanged`], beside the knob it writes; the
//!   registry applies nothing itself.
//! - A latched row's write is staged in [`Row::pending`] until [`Cvars::commit_latched`], the
//!   reference's `0x639ec0` inside `RestartGx`. A stage never committed is dropped at exit, as the
//!   reference's `SaveConfig` writes only the applied value (`rec+0x20`).
//! - Boot folds `benilla-config/config.toml` in and fires the observers before anything after
//!   [`CvarLoad`]; each frame drains the VM's writes in and pushes the registry's out; a dirty
//!   registry saves after one quiet second or at exit, writing only values off their default and
//!   keeping unknown keys verbatim.
//!
//! Env overrides (`WOW_UI_SCALE`, `WOW_FARCLIP`, …) win for the session and never touch the file
//! ([`Cvars::own_for_session`]).

use std::collections::{BTreeMap, HashMap, HashSet};
use std::time::Instant;

use bevy::prelude::*;

use crate::ui_script::VmMemo;
use benilla_ui::script::{SeededCvar, UiScript};

mod table;
pub(crate) use table::{registered_pairs, REGISTERED};
#[cfg(test)]
use table::{Reference, Registered};

/// `config.toml`: a `[cvars]` table of `Name = "value"` strings, sorted so every save is stable.
#[derive(serde::Serialize, serde::Deserialize, Default)]
struct LocalConfig {
    #[serde(default)]
    cvars: BTreeMap<String, String>,
}

// ─── The registry ────────────────────────────────────────────────────────────────────────────

/// An accepted move of a CVar's applied value: the reference's change callback, as a Bevy event.
/// Not fired for a no-op write, a staged latched value, or a session override.
#[derive(Event, Clone, Debug, PartialEq, Eq)]
pub(crate) struct CvarChanged {
    /// The registered spelling (`MasterVolume`).
    pub(crate) name: String,
    pub(crate) old: String,
    pub(crate) new: String,
}

impl CvarChanged {
    /// Case-insensitive, like every lookup the client makes.
    pub(crate) fn is(&self, name: &str) -> bool {
        self.name.eq_ignore_ascii_case(name)
    }

    /// The lowercased name, as observers' `match` arms spell it.
    pub(crate) fn key(&self) -> String {
        self.name.to_ascii_lowercase()
    }

    /// The new value as a number; `0` on a string row, whose observer reads [`Self::new`]. The
    /// registry refuses an unparseable write to a numeric row ([`Cvars::set`]).
    pub(crate) fn num(&self) -> f32 {
        self.new.trim().parse().unwrap_or(0.0)
    }

    /// The new value as the client's flag: int-parse, then `!= 0`.
    pub(crate) fn flag(&self) -> bool {
        self.num() != 0.0
    }
}

/// One row of the live registry: the parts of the reference's `CVar` record benilla keeps.
#[derive(Clone, Debug)]
pub(crate) struct Row {
    /// The registered spelling.
    pub(crate) name: String,
    pub(crate) default: String,
    /// The applied value: what `GetCVar` answers and what the file is composed from.
    pub(crate) value: String,
    /// A latched row's staged value (`rec+0x38`), applied by [`Cvars::commit_latched`].
    pub(crate) pending: Option<String>,
    /// The reference's flag bit1.
    pub(crate) latched: bool,
    /// Declared by an addon's `RegisterCVar`: persisted like any row and re-seeded into every later
    /// VM, so the addon's re-declaration is a no-op.
    pub(crate) addon: bool,
}

impl Row {
    /// A row whose default parses as a number refuses a write that does not ([`Cvars::set`]).
    fn numeric(&self) -> bool {
        self.default.trim().parse::<f32>().is_ok()
    }
}

/// What a write did.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum SetOutcome {
    /// No such row (warned).
    Unknown,
    /// A numeric row and a value that does not parse: refused, the applied value stands (warned).
    Refused,
    /// Already the applied value (or already the staged one).
    Unchanged,
    /// A latched row: staged, not applied; nothing fires until the commit.
    Staged,
    /// Applied: a [`CvarChanged`] is queued for the next flush and the config is dirty.
    Changed,
}

/// The engine-side CVar table.
#[derive(Resource)]
pub(crate) struct Cvars {
    rows: Vec<Row>,
    /// Lowercased name → row.
    index: HashMap<String, usize>,
    /// The file's `[cvars]` entries in their own spelling, the merge base of every save.
    file: BTreeMap<String, String>,
    /// Lowercased names the session owns rather than the player: the env levers, and `gxApi`, the
    /// render backend. Never saved; the file's entry is left as found.
    session_owned: HashSet<String>,
    /// Accepted moves not yet triggered, flushed by [`sync_cvars`], the boot load, the session-edge
    /// fold, or a caller of [`Cvars::take_events`].
    events: Vec<CvarChanged>,
    /// Host-side writes the VM's mirror has not seen; cleared by a seed.
    outbox: Vec<(String, String)>,
    /// A change since the last save; `last_change` drives the one-quiet-second debounce.
    dirty: bool,
    last_change: Option<Instant>,
}

impl Default for Cvars {
    fn default() -> Self {
        let mut cvars = Self {
            rows: Vec::with_capacity(REGISTERED.len()),
            index: HashMap::with_capacity(REGISTERED.len()),
            file: BTreeMap::new(),
            session_owned: HashSet::new(),
            events: Vec::new(),
            outbox: Vec::new(),
            dirty: false,
            last_change: None,
        };
        for r in REGISTERED {
            cvars.insert_row(Row {
                name: r.name.to_string(),
                default: r.default.to_string(),
                value: r.default.to_string(),
                pending: None,
                latched: r.latched,
                addon: false,
            });
        }
        cvars
    }
}

impl Cvars {
    fn insert_row(&mut self, row: Row) {
        let key = row.name.to_ascii_lowercase();
        debug_assert!(
            !self.index.contains_key(&key),
            "{}: registered twice",
            row.name
        );
        self.index.insert(key, self.rows.len());
        self.rows.push(row);
    }

    fn slot(&self, name: &str) -> Option<usize> {
        self.index.get(&name.to_ascii_lowercase()).copied()
    }

    /// The row, matched case-insensitively.
    pub(crate) fn row(&self, name: &str) -> Option<&Row> {
        self.slot(name).map(|i| &self.rows[i])
    }

    /// Every row, in registration order.
    pub(crate) fn rows(&self) -> impl Iterator<Item = &Row> {
        self.rows.iter()
    }

    pub(crate) fn get(&self, name: &str) -> Option<&str> {
        self.row(name).map(|r| r.value.as_str())
    }

    pub(crate) fn num(&self, name: &str) -> Option<f32> {
        self.get(name).and_then(|v| v.trim().parse().ok())
    }

    /// The applied value as the client's flag (int-parse, `!= 0`).
    pub(crate) fn flag(&self, name: &str) -> Option<bool> {
        self.num(name).map(|v| v != 0.0)
    }

    pub(crate) fn default_of(&self, name: &str) -> Option<&str> {
        self.row(name).map(|r| r.default.as_str())
    }

    pub(crate) fn is_session_owned(&self, name: &str) -> bool {
        self.session_owned.contains(&name.to_ascii_lowercase())
    }

    /// The persisted `checkAddonVersion` the addon load walk gates on; registered on.
    pub(crate) fn addon_version_check(&self) -> bool {
        self.flag("checkAddonVersion").unwrap_or(true)
    }

    fn touch(&mut self) {
        self.dirty = true;
        self.last_change = Some(Instant::now());
    }

    /// Remove a `config.toml` entry no row claims, a setting retired from the table, and return
    /// its value; the next save writes the file without it.
    pub(crate) fn retire_file_entry(&mut self, name: &str) -> Option<String> {
        if self.index.contains_key(&name.to_ascii_lowercase()) {
            return None;
        }
        let key = self
            .file
            .keys()
            .find(|k| k.eq_ignore_ascii_case(name))?
            .clone();
        let value = self.file.remove(&key);
        self.touch();
        value
    }

    /// A write from either side of the VM boundary. A `from_vm` write is not echoed back, but a
    /// refusal is, because the mirror already stored the refused value.
    fn write(&mut self, name: &str, value: &str, from_vm: bool) -> SetOutcome {
        let Some(i) = self.slot(name) else {
            warn!("cvar {name}: not registered — write ignored");
            return SetOutcome::Unknown;
        };
        let row = &mut self.rows[i];
        if row.numeric() && value.trim().parse::<f32>().is_err() {
            warn!(
                "cvar {}: unparseable value '{value}' refused (still {:?})",
                row.name, row.value
            );
            if from_vm {
                self.outbox.push((row.name.clone(), row.value.clone()));
            }
            return SetOutcome::Refused;
        }
        if row.latched {
            // The reference's `Set 0x63df50` on flag bit1 stores `latchedValue` and skips
            // `InternalSet`: no dirty, no callback. The stage lives here, not in the mirror, which
            // only ever learns applied values.
            let staged = (value != row.value).then(|| value.to_string());
            if row.pending == staged {
                return SetOutcome::Unchanged;
            }
            row.pending = staged;
            return if row.pending.is_some() {
                SetOutcome::Staged
            } else {
                SetOutcome::Unchanged // the stage cleared: the boundary has nothing to do
            };
        }
        if row.value == value {
            return SetOutcome::Unchanged;
        }
        let old = std::mem::replace(&mut row.value, value.to_string());
        let name = row.name.clone();
        self.events.push(CvarChanged {
            name: name.clone(),
            old,
            new: value.to_string(),
        });
        if !from_vm {
            self.outbox.push((name, value.to_string()));
        }
        self.touch();
        SetOutcome::Changed
    }

    /// A host-side write (minimap zoom, camera views, the remembered character, the tip cursor):
    /// mirrored into the VM, persisted and observed.
    pub(crate) fn set(&mut self, name: &str, value: &str) -> SetOutcome {
        let outcome = self.write(name, value, false);
        if name.eq_ignore_ascii_case("lightingQuality") {
            apply_lighting_preset(self, value);
        }
        if name.eq_ignore_ascii_case("graphicsQuality") {
            apply_graphics_preset(self, value);
        }
        outcome
    }

    /// A write the VM's mirror already made, drained from its change queue.
    pub(crate) fn set_from_vm(&mut self, name: &str, value: &str) -> SetOutcome {
        let outcome = self.write(name, value, true);
        // Apply at the write's position in the queue: a later member edit must win,
        // including a re-selection of the currently displayed preset.
        if name.eq_ignore_ascii_case("lightingQuality") {
            apply_lighting_preset(self, value);
        }
        if name.eq_ignore_ascii_case("graphicsQuality") {
            apply_graphics_preset(self, value);
        }
        outcome
    }

    /// MONKEY (presets): **the first boot's Graphics Preset** — a player whose `config.toml`
    /// names no `graphicsQuality` (a new player, or one from before the ladder) gets the
    /// [`GRAPHICS_DEFAULT`] column written over every governed row the file does not carry and the
    /// session does not own. Rows the file carries are the player's and stay; the lighting rung
    /// is left alone, because High on that ladder IS the registered defaults
    /// (`the_high_preset_is_the_registered_defaults`). Returns how many rows moved.
    pub(crate) fn seed_graphics_preset(&mut self) -> usize {
        let carried = |cvars: &Self, k: &str| {
            cvars.file.keys().any(|f| f.eq_ignore_ascii_case(k)) || cvars.is_session_owned(k)
        };
        if carried(self, "graphicsQuality") {
            return 0;
        }
        let col = graphics_column(GRAPHICS_DEFAULT).expect("the default is a rung");
        let mut moved = 0;
        for (k, values) in GRAPHICS_PRESETS {
            if k.eq_ignore_ascii_case("lightingQuality") || carried(self, k) {
                continue;
            }
            if self.set(k, values[col]) == SetOutcome::Changed {
                moved += 1;
            }
        }
        moved
    }

    /// MONKEY (followups): **re-apply the saved presets to rows the file does not carry.** A saved
    /// rung name implies every governed row matched it when the file was written (the label is
    /// re-derived every frame), and a row at its registered default is not saved. So a governed
    /// row ABSENT from the file is one that was registered after the save (or held a rung value
    /// equal to its default): it takes the saved rung's value, not its own registered default.
    /// Rows the file carries and session-owned rows are never touched; `Custom` (or no saved
    /// name) writes nothing. The lighting rung is the file's own `lightingQuality`, or else the
    /// saved Graphics rung's. Returns how many rows moved.
    pub(crate) fn reapply_saved_presets(&mut self) -> usize {
        let carried = |cvars: &Self, k: &str| {
            cvars.file.keys().any(|f| f.eq_ignore_ascii_case(k)) || cvars.is_session_owned(k)
        };
        let saved = |cvars: &Self, k: &str| {
            if cvars.is_session_owned(k) {
                return None;
            }
            cvars
                .file
                .iter()
                .find(|(f, _)| f.eq_ignore_ascii_case(k))
                .map(|(_, v)| v.clone())
        };
        let mut moved = 0;
        let col = saved(self, "graphicsQuality").and_then(|v| graphics_column(v.trim()));
        if let Some(col) = col {
            for (k, values) in GRAPHICS_PRESETS {
                if k.eq_ignore_ascii_case("lightingQuality") || carried(self, k) {
                    continue;
                }
                if self.set(k, values[col]) == SetOutcome::Changed {
                    moved += 1;
                }
            }
        }
        let lighting = saved(self, "lightingQuality").or_else(|| {
            let col = col?;
            GRAPHICS_PRESETS
                .iter()
                .find(|(k, _)| k.eq_ignore_ascii_case("lightingQuality"))
                .map(|(_, values)| values[col].to_string())
        });
        let members = lighting.and_then(|name| {
            LIGHTING_PRESETS
                .iter()
                .find(|(n, _)| n.eq_ignore_ascii_case(name.trim()))
                .map(|(_, members)| *members)
        });
        for (k, v) in members.unwrap_or(&[]) {
            if carried(self, k) {
                continue;
            }
            if self.set(k, v) == SetOutcome::Changed {
                moved += 1;
            }
        }
        moved
    }

    /// Follow a value the engine already applied, for a second spelling of one knob
    /// (`WorldDetail`/`frillDensity`): the row moves, persists and reaches the mirror, but no
    /// observer fires, since a full write would queue an event that lands a flush late and wins
    /// stale. Returns whether the row moved.
    pub(crate) fn mirror(&mut self, name: &str, value: &str) -> bool {
        let Some(i) = self.slot(name) else {
            warn!("cvar {name}: not registered — mirror ignored");
            return false;
        };
        let row = &mut self.rows[i];
        if row.value == value {
            return false;
        }
        row.value = value.to_string();
        row.pending = None;
        self.outbox.push((row.name.clone(), value.to_string()));
        self.touch();
        true
    }

    /// The latch boundary `RestartGx` crosses: the reference's `0x639ec0..0x639f5a`, which calls
    /// `CVar::Commit` (`0x63e060`) on the fourteen gx records `[0xc4ea90]` … `[0xc4eab4]`. Each
    /// staged `gx*` value is applied, fires, persists and reaches the mirror; returns how many
    /// moved.
    ///
    /// Only `gx*` rows: `SoundBufferSize`'s register site (`0x4571ca`) discards its record, so
    /// nothing commits it and its stage is lost at exit, as in the reference. The reference runs a
    /// gx callback at `SetCVar` time (`0x63df50`); ours fire here, before the device rebuild reads
    /// them.
    pub(crate) fn commit_latched(&mut self) -> usize {
        let mut moved = 0;
        for row in &mut self.rows {
            if !row.name.starts_with("gx") {
                continue;
            }
            let Some(staged) = row.pending.take() else {
                continue;
            };
            if staged == row.value {
                continue;
            }
            let old = std::mem::replace(&mut row.value, staged.clone());
            self.events.push(CvarChanged {
                name: row.name.clone(),
                old,
                new: staged.clone(),
            });
            self.outbox.push((row.name.clone(), staged));
            moved += 1;
        }
        if moved > 0 {
            self.touch();
        }
        moved
    }

    /// The session owns this row: it is never saved and the file's entry is left alone. `value` is
    /// what it answers this session; `None` marks it without moving it. No observer fires: the knob
    /// already read the env.
    pub(crate) fn own_for_session(&mut self, name: &str, value: Option<&str>) {
        let key = name.to_ascii_lowercase();
        self.session_owned.insert(key.clone());
        let Some(value) = value else {
            return;
        };
        let Some(&i) = self.index.get(&key) else {
            warn!("cvar {name}: not registered — session value ignored");
            return;
        };
        let row = &mut self.rows[i];
        if row.value != value {
            row.value = value.to_string();
            row.pending = None;
            self.outbox.push((row.name.clone(), value.to_string()));
        }
    }

    /// An addon's `RegisterCVar`: a row of its own at the file's value for the name, else at the
    /// declared default. A name already registered is a no-op.
    pub(crate) fn learn_addon_row(&mut self, name: &str, default: &str) {
        if self.slot(name).is_some() {
            return;
        }
        let saved = self
            .file
            .iter()
            .find(|(k, _)| k.eq_ignore_ascii_case(name))
            .map(|(_, v)| v.clone());
        self.insert_row(Row {
            name: name.to_string(),
            default: default.to_string(),
            value: saved.unwrap_or_else(|| default.to_string()),
            pending: None,
            latched: false,
            addon: true,
        });
    }

    /// Fold the file in: a known, player-owned key becomes its row's applied value and fires (the
    /// reference's `Register` on a record `Config.wtf` created calls the callback); an unknown key
    /// is kept for the save and warned; a session-owned one is skipped. Nothing dirties.
    fn load_file(&mut self, file: BTreeMap<String, String>) {
        for (name, value) in &file {
            let key = name.to_ascii_lowercase();
            let Some(&i) = self.index.get(&key) else {
                warn!("config: unknown cvar '{name}' — preserved, not applied");
                continue;
            };
            if self.session_owned.contains(&key) {
                info!("config: {name} is owned by this session, not the file (file value kept)");
                continue;
            }
            let row = &mut self.rows[i];
            if row.numeric() && value.trim().parse::<f32>().is_err() {
                warn!("config: {name}: unparseable value '{value}' ignored");
                continue;
            }
            if row.value == *value {
                continue;
            }
            let old = std::mem::replace(&mut row.value, value.clone());
            self.events.push(CvarChanged {
                name: row.name.clone(),
                old,
                new: value.clone(),
            });
        }
        self.file = file;
    }

    /// The file's entries no row claims: a newer build's keys, or an addon's not yet registered.
    /// Handed to the VM as its saved base, so an addon's `RegisterCVar` starts at the saved value.
    pub(crate) fn orphans(&self) -> Vec<(String, String)> {
        self.file
            .iter()
            .filter(|(k, _)| !self.index.contains_key(&k.to_ascii_lowercase()))
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect()
    }

    /// The whole table as a VM's mirror is seeded from it.
    pub(crate) fn vm_seed(&self) -> Vec<SeededCvar> {
        self.rows
            .iter()
            .map(|r| SeededCvar {
                name: r.name.clone(),
                value: r.value.clone(),
                default: r.default.clone(),
                latched: r.latched,
            })
            .collect()
    }

    /// Read before taking, so a quiet frame never deref-muts the registry.
    pub(crate) fn has_events(&self) -> bool {
        !self.events.is_empty()
    }

    /// The accepted moves since the last flush, for the caller to trigger.
    pub(crate) fn take_events(&mut self) -> Vec<CvarChanged> {
        std::mem::take(&mut self.events)
    }

    fn has_outbox(&self) -> bool {
        !self.outbox.is_empty()
    }

    fn take_outbox(&mut self) -> Vec<(String, String)> {
        std::mem::take(&mut self.outbox)
    }

    /// The file to save: the previous file as the merge base, each row off its default at its
    /// applied value (a staged one is not the player's yet), each at its default removed;
    /// session-owned and unknown keys untouched.
    fn compose(&self) -> BTreeMap<String, String> {
        let mut out = self.file.clone();
        for row in &self.rows {
            let key = row.name.to_ascii_lowercase();
            if self.session_owned.contains(&key) {
                continue;
            }
            // Match any existing entry case-insensitively so a hand-edited spelling doesn't fork.
            let existing = out
                .keys()
                .find(|k| k.eq_ignore_ascii_case(&row.name))
                .cloned();
            if row.value == row.default {
                if let Some(k) = existing {
                    out.remove(&k);
                }
            } else {
                out.insert(
                    existing.unwrap_or_else(|| row.name.clone()),
                    row.value.clone(),
                );
            }
        }
        out
    }

    /// A registry already holding one stored value, as if `config.toml` said so.
    #[cfg(test)]
    pub(crate) fn with_value(name: &str, value: &str) -> Self {
        let mut cvars = Self::default();
        cvars.load_file(BTreeMap::from([(name.to_string(), value.to_string())]));
        cvars.events.clear();
        cvars
    }
}

/// How long a dirty config waits before saving: long enough to coalesce a slider drag.
const SAVE_QUIET: std::time::Duration = std::time::Duration::from_secs(1);

/// The startup fold of `config.toml` into the registry and the knobs ([`load_config`]). A set
/// because the world camera reads `gxMultisample` once at spawn, so `setup_player` must order after
/// it; without the constraint that order is the executor's choice.
#[derive(SystemSet, Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) struct CvarLoad;

pub(crate) struct CvarPlugin;

impl Plugin for CvarPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<Cvars>()
            .add_systems(
                Startup,
                (load_config, publish_filter_policy)
                    .chain()
                    .in_set(CvarLoad),
            )
            // After the tick, so a `SetCVar` from this frame reaches the registry and its observers
            // before the frame's drains; `video::drain_restart_gx` orders after this so the commit
            // finds the stage.
            .add_systems(Update, sync_cvars.after(crate::ui_script::UiInput))
            // In `Last`, after every `Update` writer of the registry, so the save reads the frame's
            // settled state; before the exit flush, which finds it clean on a quiet exit frame.
            .add_systems(
                Last,
                save_config_when_quiet.before(crate::shutdown::OnAppExit),
            );
        // The exit flush runs on the exit edge: the close button's `AppExit` is written in
        // `PostUpdate`, so an `Update` save never sees it.
        crate::shutdown::on_app_exit(app, save_config_on_exit.into_configs());
    }
}

// ─── MONKEY (advanced graphics): the lighting preset ladder ──────────────────────────────────

/// **The one place the presets are written down** — the Advanced Graphics page's `lightingQuality`
/// ladder, as (name, members) pairs in the order [`derive_lighting_quality`] tries them.
///
/// A preset is a NAME FOR A SET OF ROWS, not a knob: applying one writes each member through the
/// ordinary registry path ([`Cvars::set`]), so every observer, clamp, bridge and save that a
/// hand-typed `/console set` would run, runs. That is why this table lives beside the registry
/// rather than in the options window's Lua: the page is one of four ways to reach it (the others
/// being `/console`, the lighting debug panel, and a hand-edited `config.toml`), and a table
/// carried by the window would be a table three of them cannot see.
///
/// **A preset names only the members it decides.** `Off` says nothing about `shadowMapSize` — with
/// both shadow lanes off there is no map to size — and Low says nothing about the torch counts,
/// which its `interiorShadows 0` has already made inert. Derivation matches on exactly the members
/// a preset lists, so a row nobody's preset mentions never pushes the ladder to `Custom`.
///
/// Order matters twice: the ladder reads Off → High for a human, and the first preset whose whole
/// member list matches wins. The four are mutually exclusive on `characterShadows` /
/// `worldShadows` / `exteriorShadows`, so no value can answer to two of them — a property
/// `every_preset_derives_back_to_its_own_name` holds rather than a reader has to check.
pub(crate) const LIGHTING_PRESETS: &[(&str, &[(&str, &str)])] = &[
    // Everything this whole system invented, off — the client as it rendered before any of it
    // existed. Not "cheap": OFF, so a machine that cannot afford the lanes can say so in one
    // click, and so a bug report can be split from a taste report in one click too.
    (
        "Off",
        &[
            ("characterShadows", "0"),
            ("worldShadows", "0"),
            ("interiorLight", "0"),
            ("interiorShadows", "0"),
            ("exteriorShadows", "0"),
            ("spellLightGain", "0"),
            ("waterQuality", "0"),
            // MONKEY (volumetric fog): Off preset restores the original atmosphere.
            ("volumetricFog", "0"),
            // MONKEY (post): no HDR lift or post pass is the byte-identical baseline.
            ("bloom", "0"),
            ("sunShafts", "0"),
            ("colorGrading", "0"),
            ("lavaLightGain", "0"),
            ("fireLightGain", "0"),
            ("nightGain", "1.0"),
            ("interiorGain", "1.0"),
            ("moonShadowStrength", "0"),
            ("fireFlicker", "0"),
            // MONKEY (wind): Off is the exact zero-displacement path.
            ("foliageWind", "0"),
        ],
    ),
    // Character silhouettes and lit rooms, and nothing that costs a second shadow pass: no world
    // lane, no torch maps indoors or out, no moon term, and the smallest sun map on the ladder.
    (
        "Low",
        &[
            ("characterShadows", "1"),
            ("worldShadows", "0"),
            ("shadowMapSize", "1024"),
            ("interiorLight", "1"),
            ("interiorShadows", "0"),
            ("exteriorShadows", "0"),
            ("spellLightGain", "1"),
            ("waterQuality", "1"),
            // MONKEY (volumetric fog): preset atmosphere uses the default cheap tier.
            ("volumetricFog", "1"),
            // MONKEY (post): quarter-resolution halo.
            ("bloom", "1"),
            ("sunShafts", "0"),
            ("colorGrading", "0"),
            ("lavaLightGain", "1"),
            ("fireLightGain", "1"),
            ("nightGain", "0.45"),
            ("interiorGain", "0.5"),
            ("moonShadowStrength", "0"),
            ("fireFlicker", "1"),
            // MONKEY (wind): Low animates the denser grass lane only.
            ("foliageWind", "1"),
        ],
    ),
    // Both sun lanes, indoor torch shadows at half the residency, and the moon term — but no
    // OUTDOOR torch shadows, which are the one lane whose cost is paid in the open world where the
    // sun lanes are already paying.
    (
        "Medium",
        &[
            ("characterShadows", "1"),
            ("worldShadows", "1"),
            ("shadowMapSize", "2048"),
            ("interiorLight", "1"),
            ("interiorShadows", "1"),
            ("interiorShadowCasters", "6"),
            ("interiorShadowDynamic", "2"),
            ("exteriorShadows", "0"),
            ("spellLightGain", "1"),
            ("waterQuality", "1"),
            // MONKEY (volumetric fog): preset atmosphere uses the default cheap tier.
            ("volumetricFog", "1"),
            // MONKEY (post): quarter-resolution halo.
            ("bloom", "1"),
            ("sunShafts", "0"),
            ("colorGrading", "0"),
            ("lavaLightGain", "1"),
            ("fireLightGain", "1"),
            ("nightGain", "0.45"),
            ("interiorGain", "0.5"),
            ("moonShadowStrength", "0.35"),
            ("fireFlicker", "1"),
            // MONKEY (wind): Medium and High include classified trees.
            ("foliageWind", "2"),
        ],
    ),
    // **High IS the shipped default**, member for member — which is a property, not a coincidence:
    // `the_high_preset_is_the_registered_defaults` welds the two, so a fresh `benilla-config`
    // reads "High" on this row rather than "Custom".
    (
        "High",
        &[
            ("characterShadows", "1"),
            ("worldShadows", "1"),
            ("shadowMapSize", "2048"),
            ("interiorLight", "1"),
            ("interiorShadows", "1"),
            ("interiorShadowCasters", "12"),
            ("interiorShadowDynamic", "4"),
            ("exteriorShadows", "1"),
            ("spellLightGain", "1"),
            // Mirror reflections stay opt-in until their cost is measured.
            ("waterQuality", "1"),
            // MONKEY (volumetric fog): preset atmosphere uses the default cheap tier.
            ("volumetricFog", "1"),
            // MONKEY (post): half-resolution halo.
            ("bloom", "2"),
            ("sunShafts", "1"),
            ("colorGrading", "1"),
            ("lavaLightGain", "1"),
            ("fireLightGain", "1"),
            ("nightGain", "0.45"),
            ("interiorGain", "0.5"),
            ("moonShadowStrength", "0.35"),
            ("fireFlicker", "1"),
            // MONKEY (wind): High preset = grass plus classified trees.
            ("foliageWind", "2"),
        ],
    ),
    // MONKEY (presets): High plus every lane at its maximum — the 4096 sun map, all sixteen
    // resident torch maps with eight of them tracking movers, mirror-reflection water and the
    // High fog march. Disjoint from High on `shadowMapSize`, so derivation never confuses them.
    (
        "Ultra",
        &[
            ("characterShadows", "1"),
            ("worldShadows", "1"),
            ("shadowMapSize", "4096"),
            ("interiorLight", "1"),
            ("interiorShadows", "1"),
            ("interiorShadowCasters", "16"),
            ("interiorShadowDynamic", "8"),
            ("exteriorShadows", "1"),
            ("spellLightGain", "1"),
            ("waterQuality", "2"),
            ("volumetricFog", "2"),
            ("bloom", "2"),
            ("sunShafts", "1"),
            ("colorGrading", "1"),
            ("lavaLightGain", "1"),
            ("fireLightGain", "1"),
            ("nightGain", "0.45"),
            ("interiorGain", "0.5"),
            ("moonShadowStrength", "0.35"),
            ("fireFlicker", "1"),
            ("foliageWind", "2"),
        ],
    ),
];

/// What the ladder shows when the members match no preset. Not a preset: selecting it writes
/// nothing, and the next derivation puts back whatever the members really say.
pub(crate) const LIGHTING_CUSTOM: &str = "Custom";

/// Two registry values are the SAME setting when they are the same number — `"0"`, `"0.00"` and
/// `"0.0"` all mean a lane that is off, and a preset whose member reads `"1"` must not be called
/// `Custom` because a slider wrote `"1.00"`. Falls back to a case-insensitive string compare for
/// the rows that hold words.
fn same_value(a: &str, b: &str) -> bool {
    match (a.trim().parse::<f32>(), b.trim().parse::<f32>()) {
        (Ok(x), Ok(y)) => (x - y).abs() < 1e-4,
        _ => a.eq_ignore_ascii_case(b),
    }
}

/// **The name the members currently spell** — the first preset every one of whose members matches
/// the live registry, or [`LIGHTING_CUSTOM`]. This is the only thing `lightingQuality` is ever
/// allowed to say, which is what keeps the row from going stale behind a `/console` write.
pub(crate) fn derive_lighting_quality(cvars: &Cvars) -> &'static str {
    for (name, members) in LIGHTING_PRESETS {
        // MONKEY (reviewfix-a): a session-owned row (an env lever) is not the player's choice and
        // must not turn the saved label Custom; it is skipped, as the seed skips it.
        let matched = members
            .iter()
            .filter(|(k, _)| !cvars.is_session_owned(k))
            .all(|(k, v)| cvars.get(k).is_some_and(|live| same_value(live, v)));
        if matched {
            return name;
        }
    }
    LIGHTING_CUSTOM
}

/// Write one preset's members. Ordinary registry writes: observers, clamps and the save all run.
/// Unknown names (including `Custom`) write nothing and say so by answering `false`.
pub(crate) fn apply_lighting_preset(cvars: &mut Cvars, name: &str) -> bool {
    let Some((_, members)) = LIGHTING_PRESETS
        .iter()
        .find(|(n, _)| n.eq_ignore_ascii_case(name))
    else {
        return false;
    };
    for (k, v) in *members {
        cvars.set(k, v);
    }
    true
}

/// Derive after all ordered writes, before the mirror/observer flush. File loading bypasses
/// `set`, so saved members remain authoritative even when the saved preset name disagrees.
fn lighting_quality(cvars: &mut Cvars) {
    let derived = derive_lighting_quality(cvars);
    // A derived label is a mirror, never another request to apply a preset.
    cvars.mirror("lightingQuality", derived);
    // MONKEY (presets): after the lighting label, which the graphics ladder reads as a member.
    let graphics = derive_graphics_quality(cvars);
    cvars.mirror("graphicsQuality", graphics);
}

// ─── MONKEY (presets): the Graphics Preset ladder ─────────────────────────────────────────────

/// The Graphics Preset's rungs, in the order [`derive_graphics_quality`] tries them and the
/// column order of [`GRAPHICS_PRESETS`].
pub(crate) const GRAPHICS_PRESET_NAMES: [&str; 5] = ["Classic", "Low", "Medium", "High", "Ultra"];

/// The rung a player with no saved preset boots into ([`Cvars::seed_graphics_preset`]).
pub(crate) const GRAPHICS_DEFAULT: &str = "High";

/// MONKEY (reviewfix-a): **the rows where the first-boot seed leaves the reference** — the
/// [`GRAPHICS_DEFAULT`] column value a new player actually boots with, against the reference's
/// own default. The registered row keeps its `same(...)` (the registry default IS the
/// reference's; a capture or a test, which never seeds, runs at it), so this list is where the
/// deviation of the *seeded* boot value is declared, `Deviates`-style: `(row, reference, why)`.
/// `seeded_column_deviates_only_where_declared` walks the whole column against the reference
/// rows, both ways.
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) const SEEDED_DEVIATIONS: &[(&str, &str, &str)] = &[(
    "farclip",
    "350",
    "owner decision: the High preset is the first-boot default, and High's view distance is the \
     reference's own slider maximum, 777 yd; Classic keeps 350",
)];

/// **The one place the Graphics Preset is written down**: one line per governed row, one column
/// per rung of [`GRAPHICS_PRESET_NAMES`]. `lightingQuality` is itself a row — writing its rung
/// writes [`LIGHTING_PRESETS`]' members — so every lighting, water, fog, post and wind row the
/// lighting ladder owns is governed here without a second copy. The other rows are disjoint from
/// that ladder, which is what lets both labels be right at once.
///
/// Classic is the reference client: every lane off, `farclip` at its registered 350. High is
/// each lane's documented High (LIGHTING.md, WATER.md); Ultra is High at every maximum, and the
/// only rung past the reference's 777 yd (`FARCLIP_RANGE`); Low and Medium keep the practically
/// free lanes (sky, dither, modern fog, wet surfaces) and leave the costly ones to High.
///
/// A row that lands later is ONE line here, and only once it is registered (an unregistered
/// member would never match, and the ladder would read Custom forever).
#[rustfmt::skip]
pub(crate) const GRAPHICS_PRESETS: &[(&str, [&str; 5])] = &[
    //                        Classic  Low    Medium    High    Ultra
    ("lightingQuality",      ["Off",  "Low", "Medium", "High", "Ultra"]),
    ("farclip",              ["350",  "350", "477",    "777",  "1497"]),
    ("skyQuality",           ["0",    "1",   "1",      "2",    "2"]),
    ("skyDither",            ["0",    "1",   "1",      "1",    "1"]),
    ("fogModel",             ["0",    "1",   "1",      "1",    "1"]),
    ("rainSurfaces",         ["0",    "1",   "1",      "1",    "1"]),
    ("torchTerrainShadows",  ["0",    "0",   "0",      "1",    "1"]),
    ("daylightWindowSplit",  ["0",    "1",   "1",      "1",    "1"]),
    // MONKEY (integration): the three rows that landed in round 3. Low keeps them off (AO and
    // the lamp halos are GPU passes; the zone skybox is a look change), Medium takes the Low tiers.
    ("ambientOcclusion",     ["0",    "0",   "1",      "2",    "2"]),
    ("zoneSkyboxes",         ["0",    "0",   "1",      "1",    "1"]),
    ("lampFog",              ["0",    "0",   "1",      "2",    "2"]),
    // GFX (volumetric light) / (moonlight): the shafts are a GPU pass (Medium = the cheap tier);
    // moonlight is a few ALU per fragment, so every rung past Classic takes it.
    ("volumetricLight",      ["0",    "0",   "1",      "2",    "2"]),
    ("moonLight",            ["0",    "1",   "1",      "1",    "1"]),
];

/// The column of a rung, matched case-insensitively; `None` for `Custom` or anything else.
fn graphics_column(name: &str) -> Option<usize> {
    GRAPHICS_PRESET_NAMES
        .iter()
        .position(|n| n.eq_ignore_ascii_case(name))
}

/// The rung every governed row currently spells, or [`LIGHTING_CUSTOM`] (the one "Custom" both
/// ladders show). Reads `lightingQuality`'s DERIVED label, so run it after that is mirrored.
pub(crate) fn derive_graphics_quality(cvars: &Cvars) -> &'static str {
    for (col, name) in GRAPHICS_PRESET_NAMES.iter().enumerate() {
        // MONKEY (reviewfix-a): session-owned rows are skipped, as in the lighting ladder above.
        let matched = GRAPHICS_PRESETS
            .iter()
            .filter(|(k, _)| !cvars.is_session_owned(k))
            .all(|(k, v)| cvars.get(k).is_some_and(|live| same_value(live, v[col])));
        if matched {
            return name;
        }
    }
    LIGHTING_CUSTOM
}

/// Write one rung's column through [`Cvars::set`]; `false` (and nothing written) for an unknown
/// name, `Custom` included.
pub(crate) fn apply_graphics_preset(cvars: &mut Cvars, name: &str) -> bool {
    let Some(col) = graphics_column(name) else {
        return false;
    };
    for (k, values) in GRAPHICS_PRESETS {
        cvars.set(k, values[col]);
    }
    true
}

/// What the environment took for this session and its value, read off the knobs the env levers
/// seeded (`RenderScale::default()` reads `$WOW_RENDER_SCALE`, …). A lever whose resource is absent
/// still marks its row, so the file cannot apply over the env.
fn session_values(world: &World) -> Vec<(&'static str, Option<String>)> {
    let set = |k: &str| std::env::var_os(k).is_some();
    let flag = |b: bool| if b { "1" } else { "0" }.to_string();
    let mut out: Vec<(&'static str, Option<String>)> = Vec::new();
    if set("WOW_UI_SCALE") {
        let v = world.get_resource::<crate::ui_script::UiScaleCvar>();
        out.push(("uiScale", v.map(|s| s.0.to_string())));
    }
    if set("WOW_FARCLIP") {
        let v = world.get_resource::<benilla_world::view::ViewDistance>();
        out.push(("farclip", v.map(|s| s.farclip.to_string())));
    }
    // The clutter lever takes both spellings of its knob; an off-grid multiplier seeds off-grid.
    if set("WOW_CLUTTER_DENSITY") {
        let v = world.get_resource::<benilla_world::clutter::ClutterConfig>();
        out.push(("WorldDetail", v.map(|c| (c.density - 1.0).to_string())));
        out.push(("frillDensity", v.map(|c| c.frill_density().to_string())));
    }
    if crate::video::novsync_env() {
        let v = world.get_resource::<crate::video::VideoConfig>();
        out.push(("gxVSync", v.map(|c| flag(c.vsync))));
    }
    let tex = world.get_resource::<benilla_assets::TexFilterSetting>();
    if set("WOW_TRILINEAR") {
        out.push(("trilinear", tex.map(|t| flag(t.trilinear))));
    }
    if set("WOW_ANISO") {
        out.push(("anisotropic", tex.map(|t| t.aniso.to_string())));
    }
    // `$WOW_WIN`, a capture scenario or an instrumented run owns the window geometry.
    if crate::video::windowed_env() {
        let v = world.get_resource::<crate::video::VideoConfig>();
        out.push((
            "gxWindow",
            v.map(|c| flag(c.display == crate::video::DisplayMode::Windowed)),
        ));
        out.push(("gxMaximize", v.map(|c| flag(c.maximize))));
        out.push((
            "gxResolution",
            v.map(|c| format!("{}x{}", c.windowed.x, c.windowed.y)),
        ));
    }
    // The multisampling and render-scale levers: an instrument run must not pin 4× into the file.
    if set("WOW_MSAA") {
        let v = world.get_resource::<benilla_world::view::MsaaSetting>();
        out.push(("gxMultisample", v.map(|m| m.samples.to_string())));
    }
    if set("WOW_RENDER_SCALE") {
        let v = world.get_resource::<crate::world_backdrop::RenderScale>();
        out.push(("renderScale", v.map(|r| r.0.to_string())));
    }
    // `$WOW_HOST` is the session's realmlist, which a test run must never write into the file.
    if set("WOW_HOST") {
        let v = world.get_resource::<crate::realmlist::Realmlist>();
        out.push((
            crate::realmlist::CVAR_REALMLIST,
            v.map(|r| r.address().to_string()),
        ));
    }
    out
}

/// Startup: mark what the session owns, fold `config.toml` in (no file means all defaults) and fire
/// the observers now, so everything after [`CvarLoad`] finds its knob written. The VM does not
/// exist yet; [`sync_cvars`] seeds its mirror.
fn load_config(world: &mut World) {
    let session = session_values(world);
    let stored = stored_config();
    let stored_kind = match &stored {
        StoredConfig::Absent => StoredKind::Absent,
        StoredConfig::Table(_) => StoredKind::Table,
        StoredConfig::Bad(_) => StoredKind::Bad,
    };
    let events = {
        let mut cvars = world.resource_mut::<Cvars>();
        for (name, value) in session {
            cvars.own_for_session(name, value.as_deref());
        }
        // `gxApi` is the render backend, a fact about the machine that `sync_cvars` pushes live, so
        // it is never persisted.
        cvars.own_for_session("gxApi", None);
        match stored {
            StoredConfig::Absent => {} // no file, capture without a fixture, or no install
            StoredConfig::Bad(msg) => {
                // A malformed file is kept: nothing loads, and nothing saves over it until a
                // change.
                warn!("{msg}");
            }
            StoredConfig::Table(table) => {
                cvars.load_file(table);
                // MONKEY (followups): rows registered after the save follow the saved preset.
                cvars.reapply_saved_presets();
            }
        }
        // MONKEY (presets): a player's own run (never a capture, never a malformed file) boots
        // into the default Graphics Preset over whatever its file leaves unsaid.
        if seeds_graphics_preset(&stored_kind) {
            cvars.seed_graphics_preset();
        }
        cvars.take_events()
    };
    for event in events {
        world.trigger(event);
    }
}

/// MONKEY (presets): which [`StoredConfig`] arm a boot took, kept past the table's move.
enum StoredKind {
    Absent,
    Table,
    Bad,
}

/// MONKEY (presets): whether this boot seeds the default Graphics Preset — a run that reads and
/// saves a player's `config.toml` and found it absent or well-formed. A capture (hermetic, or on
/// an explicit fixture) keeps the registered defaults so its A/B stays exact, and a malformed
/// file is left for the player rather than papered over.
fn seeds_graphics_preset(kind: &StoredKind) -> bool {
    let players_file = std::env::var_os("WOW_CAPTURE").is_none() && config_read_path().is_some();
    players_file && matches!(kind, StoredKind::Absent | StoredKind::Table)
}

/// What the one read of `config.toml` found.
enum StoredConfig {
    /// No file, no install, or a capture without an explicit fixture — use registered defaults.
    Absent,
    Table(BTreeMap<String, String>),
    /// Unreadable or malformed; the message is carried because this read runs before `LogPlugin`
    /// exists.
    Bad(String),
}

/// Read `config.toml`, for [`load_config`] and for the primary window in [`crate::run`], which
/// needs `gxWindow`/`gxResolution` before it exists. Not cached: a process-wide cache would answer
/// every test from the first one's file.
fn stored_config() -> StoredConfig {
    let Some(path) = config_read_path() else {
        return StoredConfig::Absent; // capture without a fixture, or no install — session-only
    };
    let text = match std::fs::read_to_string(&path) {
        Ok(t) => t,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return StoredConfig::Absent,
        Err(e) => return StoredConfig::Bad(format!("config: cannot read {}: {e}", path.display())),
    };
    match toml::from_str::<LocalConfig>(&text) {
        Ok(cfg) => StoredConfig::Table(cfg.cvars),
        Err(e) => StoredConfig::Bad(format!(
            "config: {} is malformed ({e}) — running on defaults",
            path.display()
        )),
    }
}

/// MONKEY (integration): a capture may opt into one explicit, read-only CVar fixture.
/// MONKEY (reviewfix-a): through its own variable, `WOW_CAPTURE_CVARS=<config.toml path>`, not
/// `BENILLA_HOME` (the general local-state override a player shell may carry, which would load
/// that player's settings into a baseline). Without it a capture reads no config at all. All
/// other local state stays hermetic and [`save_config`] still sees no path.
fn config_read_path() -> Option<std::path::PathBuf> {
    if std::env::var_os("WOW_CAPTURE").is_some() {
        return std::env::var_os("WOW_CAPTURE_CVARS").map(std::path::PathBuf::from);
    }
    crate::local_state::config_path()
}

/// One CVar as `config.toml` holds it, before the `App` exists: for the primary window, which is
/// built with its display mode resolved. Everything else reads [`Cvars::get`].
pub(crate) fn boot_cvar(name: &str) -> Option<String> {
    match stored_config() {
        StoredConfig::Table(t) => t
            .into_iter()
            .find(|(k, _)| k.eq_ignore_ascii_case(name))
            .map(|(_, v)| v),
        StoredConfig::Absent | StoredConfig::Bad(_) => None,
    }
}

/// Per frame: seed a new VM's mirror, drain its registrations and writes into the registry, push
/// the registry's writes into the mirror, and trigger every accepted move.
pub(crate) fn sync_cvars(
    script: Option<NonSendMut<UiScript>>,
    mut cvars: ResMut<Cvars>,
    adapter: Option<Res<bevy::render::renderer::RenderAdapterInfo>>,
    msaa_formats: Option<Res<benilla_world::view::MsaaFormats>>,
    mut seeded: Local<VmMemo<bool>>,
    mut commands: Commands,
) {
    // The live render backend; absent headless, where the registered `""` stands.
    if let Some(adapter) = adapter.as_deref() {
        let backend = adapter.backend.to_str();
        if cvars.get("gxApi") != Some(backend) {
            cvars.own_for_session("gxApi", Some(backend));
        }
    }
    let Some(mut script) = script else {
        lighting_quality(&mut cvars);
        // Nothing to mirror into; a later VM is seeded from the table, which carries it all.
        if cvars.has_outbox() {
            cvars.take_outbox();
        }
        if cvars.has_events() {
            for event in cvars.take_events() {
                commands.trigger(event);
            }
        }
        return;
    };
    // The VM's writes before the seed: an addon's `SetCVar` at load is already queued, and seeding
    // first would overwrite the mirror with the older value. Registrations before writes, since an
    // addon declares a row and sets it together.
    let registrations = script.take_cvar_registrations();
    let changes = script.take_cvar_changes();
    if !registrations.is_empty() || !changes.is_empty() {
        for (name, default) in registrations {
            cvars.learn_addon_row(&name, &default);
        }
        for (name, value) in changes {
            cvars.set_from_vm(&name, &value);
        }
    }
    lighting_quality(&mut cvars);
    if seeded.claim(&script) {
        // The file's unclaimed entries first, so an addon's `RegisterCVar` starts at the saved
        // value; then the table.
        script.set_cvar_saved_base(cvars.orphans());
        script.seed_cvars(cvars.vm_seed());
        if cvars.has_outbox() {
            cvars.take_outbox(); // the seed just carried everything
        }
        // The Video dropdown's menu: what this device accepts, enumerated by
        // `view::MsaaSupportPlugin`.
        script.set_multisample_formats(
            msaa_formats
                .as_deref()
                .map(|f| {
                    f.formats
                        .iter()
                        .map(|&(color_bits, depth_bits, samples)| {
                            benilla_ui::script::MultisampleFormat {
                                color_bits,
                                depth_bits,
                                samples,
                            }
                        })
                        .collect()
                })
                .unwrap_or_default(),
        );
        // `GetVideoCaps`, the seven values `OptionsFrame_Load` destructures. Shaders, trilinear,
        // anisotropy and the hardware cursor are always there (`crate::cursor` composites the
        // reference's cursors into an OS cursor); `max_anisotropy` is reported raw because
        // `OptionsFrame.lua:122` matches it against `ANISOTROPIC_VALUES`. Triple buffering is
        // false: wgpu owns the surface's buffering, so the stock frame hides check button 13
        // (`OptionsFrame.lua:163-165`).
        script.set_video_caps(benilla_ui::script::VideoCaps {
            anisotropic: true,
            pixel_shaders: true,
            vertex_shaders: true,
            trilinear: true,
            triple_buffering: false,
            max_anisotropy: *benilla_assets::ANISO_RANGE.end(),
            hardware_cursor: true,
        });
    }
    if cvars.has_outbox() {
        for (name, _) in cvars.take_outbox() {
            // A preset may have queued this before a newer VM member edit. Publish
            // the final registry value, never that older queued snapshot.
            if let Some(value) = cvars.get(&name) {
                script.set_cvar_host(&name, value);
            }
        }
    }
    if cvars.has_events() {
        for event in cvars.take_events() {
            commands.trigger(event);
        }
    }
}

/// Fold the dying VM's last writes into the registry. Called from
/// [`crate::ui_script::end_ui_session`] after the shutdown events (a `PLAYER_LOGOUT` handler may
/// `SetCVar`, and the reference keeps that write) and before the VM is replaced, so a final-frame
/// write is not lost to the next VM's seed. The observers fire here.
pub(crate) fn fold_dying_vm_cvars(world: &mut World) {
    // No registry: a test or stripped world without the plugin.
    if !world.contains_resource::<Cvars>() {
        return;
    }
    let (registrations, changes) = {
        let Some(mut script) = world.get_non_send_resource_mut::<UiScript>() else {
            return;
        };
        (script.take_cvar_registrations(), script.take_cvar_changes())
    };
    let events = {
        let mut cvars = world.resource_mut::<Cvars>();
        for (name, default) in registrations {
            cvars.learn_addon_row(&name, &default);
        }
        for (name, value) in changes {
            cvars.set_from_vm(&name, &value);
        }
        lighting_quality(&mut cvars);
        if cvars.has_outbox() {
            cvars.take_outbox(); // the VM this was for is going away
        }
        if cvars.has_events() {
            cvars.take_events()
        } else {
            Vec::new()
        }
    };
    for event in events {
        world.trigger(event);
    }
}

/// The file's header comment.
const HEADER: &str = "\
# benilla local config — CVar values that moved off their defaults.
# Managed by the client; hand edits are read on next launch and preserved on save.
";

/// Dirty and one quiet second since the last change: save, so a crash loses at most that second.
fn save_config_when_quiet(mut cvars: ResMut<Cvars>) {
    if cvars.dirty && cvars.last_change.is_none_or(|t| t.elapsed() >= SAVE_QUIET) {
        write_config(&mut cvars);
    }
}

/// The exit frame: save whatever is still dirty, quiet second or not.
fn save_config_on_exit(mut cvars: ResMut<Cvars>) {
    if cvars.dirty {
        write_config(&mut cvars);
    }
}

/// Rewrite `config.toml` atomically from the registry, so a session with no VM saves what it
/// changed.
fn write_config(cvars: &mut Cvars) {
    let Some(path) = crate::local_state::config_path() else {
        cvars.dirty = false; // hermetic/session-only: nothing to write, stop retrying
        return;
    };
    let file = cvars.compose();
    let body = toml::to_string(&LocalConfig {
        cvars: file.clone(),
    })
    .expect("string map serializes");
    match crate::local_state::write_atomic(&path, &format!("{HEADER}{body}")) {
        Ok(()) => {
            cvars.file = file;
            cvars.dirty = false;
        }
        Err(e) => {
            warn!("config: cannot write {}: {e}", path.display());
            cvars.dirty = false; // don't retry every frame into the same error
        }
    }
}

/// Freeze the texture filter policy for the process and log the mode, so a player's log says which
/// mode the run was in. Its own system after [`load_config`], so it publishes whatever the file
/// held.
fn publish_filter_policy(filter: Res<benilla_assets::TexFilterSetting>) {
    benilla_assets::publish_tex_filter(*filter);
    let mode = filter.mode();
    let name = match mode {
        3 => "bilinear + nearest-mip select, aniso off",
        4 => "trilinear, aniso off",
        _ => "trilinear + aniso",
    };
    info!(
        "texture filter: mode {mode} ({name}) — trilinear={} anisotropic={}",
        u8::from(filter.trilinear),
        filter.aniso
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chat_bubble::BubbleConfig;
    use crate::minimap::MinimapZoom;
    use crate::nameplates::NameConfig;
    use crate::player::camera::{FollowConfig, FollowStyle, LookConfig, FOLLOW_SPEED_RANGE};
    use crate::player::camera_zoom::ZoomLimit;
    use crate::portrait::PaneRate;
    use crate::sound::SoundConfig;
    use crate::target::ClickConfig;
    use crate::ui_loot::LootConfig;
    use crate::ui_script::{UiScaleCvar, DEFAULT_UI_SCALE};
    use crate::video::VideoConfig;
    use crate::world_backdrop::{RenderScale, RENDER_SCALE_RANGE};
    use benilla_ui::widget::MINIMAP_ZOOM_LEVELS;
    use benilla_world::clutter::ClutterConfig;
    use benilla_world::view::{MsaaSetting, ViewDistance, FARCLIP_RANGE, MSAA_RANGE};

    /// Every row's [`Reference`] column holds, both ways: a `Same` row that drifted fails, and so
    /// does a `Deviates` or `Overridden` row back in agreement. Numbers parse-compare, so "1" and
    /// "1.0" agree.
    #[test]
    fn defaults_stand_where_the_reference_column_says() {
        /// Numeric when both parse, textual otherwise.
        fn agrees(ours: &str, theirs: &str) -> bool {
            match (ours.parse::<f32>(), theirs.parse::<f32>()) {
                (Ok(a), Ok(b)) => a == b,
                _ => ours == theirs,
            }
        }
        for row in REGISTERED {
            let name = row.name;
            match &row.reference {
                Reference::Same(value) => assert!(
                    agrees(row.default, value),
                    "{name}: the row claims the reference registers {value:?} and we ship the \
                     same, but our default is {:?}. If the reference really does differ, this is \
                     a `deviates` row and owes a reason.",
                    row.default,
                ),
                Reference::Deviates { value, why } => {
                    assert!(
                        !agrees(row.default, value),
                        "{name}: a `deviates` row that no longer deviates — our {:?} IS the \
                         reference's. Demote it to `same`; a stale deviation hides that we are \
                         faithful again.",
                        row.default,
                    );
                    assert!(!why.trim().is_empty(), "{name}: a deviation owes a reason");
                }
                Reference::Overridden { registered, why } => {
                    assert!(
                        !agrees(row.default, registered),
                        "{name}: an `overridden` row whose default is just the registered string \
                         {registered:?} — that is `same`, and saying otherwise buries a real \
                         override behind a false one.",
                    );
                    assert!(
                        !why.trim().is_empty(),
                        "{name}: an override owes its mechanism"
                    );
                }
                Reference::Ours(why) => assert!(
                    !why.trim().is_empty(),
                    "{name}: a CVar the reference does not have owes the reason it exists",
                ),
            }
        }
    }

    #[test]
    fn the_options_that_leave_the_reference_are_this_list_and_no_other() {
        let mut names: Vec<&str> = REGISTERED
            .iter()
            .filter(|r| matches!(r.reference, Reference::Deviates { .. }))
            .map(|r| r.name)
            .collect();
        names.sort_unstable();
        assert_eq!(
            names,
            vec![
                "SoundReverb",
                "frillDensity",
                "gxApi",
                "gxColorBits",
                "gxDepthBits",
                "gxResolution",
                "realmList",
                "weatherDensity",
            ],
        );
    }

    /// Every numeric registered default equals the code constant it mirrors; the string rows are
    /// checked in [`the_string_valued_cvars_are_the_realm_and_the_windowed_size`].
    #[test]
    fn registered_defaults_mirror_the_code_truths() {
        let d: BTreeMap<&str, f32> = REGISTERED
            .iter()
            .filter_map(|r| r.default.parse::<f32>().ok().map(|f| (r.name, f)))
            .collect();
        let sound = SoundConfig::default();
        assert_eq!(d["MasterVolume"], sound.master);
        assert_eq!(d["SoundVolume"], sound.sfx);
        assert_eq!(d["MusicVolume"], sound.music);
        assert_eq!(d["AmbienceVolume"], sound.ambience);
        assert_eq!(d["MasterSoundEffects"] != 0.0, sound.enabled);
        assert_eq!(d["EnableMusic"] != 0.0, sound.music_enabled);
        assert_eq!(d["EnableAmbience"] != 0.0, sound.ambience_enabled);
        assert_eq!(d["EnableErrorSpeech"] != 0.0, sound.error_speech);
        assert_eq!(
            d["Sound_EnableSoundWhenGameIsInBG"] != 0.0,
            sound.background_sound
        );
        assert!(
            !sound.background_sound,
            "the reference goes quiet in the background and offers no way out"
        );
        assert_eq!(d["SoundReverb"] != 0.0, sound.reverb);
        assert_eq!(d["SoundOutputLimiter"] != 0.0, sound.limiter);
        assert!(sound.limiter, "the output limiter ships on");
        assert!(!sound.reverb, "zone reverb ships off");
        assert_eq!(d["uiScale"], DEFAULT_UI_SCALE);
        // `ViewDistance::default()` reads `$WOW_FARCLIP`, so this is the env-less literal.
        assert_eq!(d["farclip"], 350.0);
        assert_eq!(d["nearclip"], benilla_world::view::NEARCLIP_DEFAULT);
        assert_eq!(
            d["nearclip"],
            ViewDistance::default().nearclip,
            "the registered default and the resource's own must be one number"
        );
        // `MsaaSetting::default()` reads `$WOW_MSAA`, so this is the env-less literal.
        assert_eq!(d["gxMultisample"], 1.0);
        assert_eq!(
            d["deselectOnClick"] != 0.0,
            ClickConfig::default().deselect_on_click
        );
        assert_eq!(
            d["mouseInvertPitch"] != 0.0,
            LookConfig::default().invert_pitch
        );
        assert_eq!(d["mousespeed"], LookConfig::default().sensitivity);
        assert_eq!(d["cameraDistanceMax"], ZoomLimit::default().distance_max());
        assert_eq!(d["cameraDistanceMaxFactor"], ZoomLimit::default().factor());
        let follow = FollowConfig::default();
        assert_eq!(
            d["cameraSmoothStyle"],
            follow.style.cvar().parse::<f32>().unwrap()
        );
        assert_eq!(
            d["cameraSmoothTrackingStyle"],
            follow.tracking_style.cvar().parse::<f32>().unwrap()
        );
        assert_eq!(d["cameraYawSmoothSpeed"], follow.yaw_speed);
        assert_eq!(FollowStyle::default(), FollowStyle::Smart);
        assert_eq!(d["autoLootDefault"] != 0.0, LootConfig::default().auto_loot);
        assert_eq!(
            d["showLootSpam"] != 0.0,
            LootConfig::default().show_loot_spam
        );
        assert_eq!(
            d["guildMemberNotify"] != 0.0,
            crate::ui_guild::GuildMemberNotify::default().0
        );
        assert_eq!(d["guildMemberNotify"], 0.0, "the binary registers \"0\"");
        assert_eq!(
            d["BlockTrades"] != 0.0,
            crate::ui_trade::BlockTrades::default().0
        );
        assert_eq!(d["BlockTrades"], 0.0, "an unset BlockTrades allows trades");
        let names = NameConfig::default();
        assert_eq!(d["UnitNamePlayer"] != 0.0, names.player);
        assert_eq!(d["UnitNameNPC"] != 0.0, names.npc);
        assert_eq!(d["UnitNameOwn"] != 0.0, names.own);
        assert_eq!(d["UnitNamePlayerGuild"] != 0.0, names.player_guild);
        assert!(
            names.player && !names.npc && !names.own && names.player_guild,
            "the binary registers UnitNamePlayer \"1\", NPC \"0\", Own \"0\", \
             PlayerGuild \"1\""
        );
        let camera_opts = crate::player::camera_dynamics::CameraOptions::default();
        assert_eq!(d["cameraPivot"] != 0.0, camera_opts.pivot);
        assert!(camera_opts.pivot, "the binary registers cameraPivot \"1\"");
        assert_eq!(
            d["cameraWaterCollision"] != 0.0,
            camera_opts.water_collision
        );
        assert!(
            camera_opts.pivot && camera_opts.water_collision,
            "the binary registers cameraPivot and cameraWaterCollision both \"1\""
        );
        assert_eq!(d["cameraTerrainTilt"] != 0.0, camera_opts.terrain_tilt);
        assert!(
            !camera_opts.terrain_tilt,
            "the binary registers cameraTerrainTilt \"0\""
        );
        assert_eq!(
            d["cameraGroundSmoothSpeed"],
            camera_opts.ground_smooth_speed
        );
        assert_eq!(d["cameraTerrainTiltTimeMin"], camera_opts.tilt_time_min);
        assert_eq!(d["cameraTerrainTiltTimeMax"], camera_opts.tilt_time_max);
        assert_eq!(d["cameraBobbing"] != 0.0, camera_opts.bobbing);
        assert!(
            !camera_opts.bobbing && !camera_opts.terrain_tilt,
            "the binary registers cameraBobbing and cameraTerrainTilt both \"0\""
        );
        assert_eq!(d["cameraBobbingLRAmplitude"], camera_opts.bob_lr_amplitude);
        assert_eq!(d["cameraBobbingUDAmplitude"], camera_opts.bob_ud_amplitude);
        assert_eq!(d["cameraBobbingFrequency"], camera_opts.bob_frequency);
        assert_eq!(d["cameraBobbingSmoothSpeed"], camera_opts.bob_smooth_speed);
        assert_eq!(d["cameraPivotDXMax"], camera_opts.pivot_dx_max);
        assert_eq!(d["cameraPivotDYMin"], camera_opts.pivot_dy_min);
        assert_eq!(
            d["cameraTargetSmoothSpeed"],
            camera_opts.target_smooth_speed
        );
        assert_eq!(
            d["weatherDensity"],
            f32::from(benilla_world::weather::WeatherState::default().weather_density)
        );
        // `ClutterConfig::default()` reads `$WOW_CLUTTER_DENSITY`; stop 1 is its env-less ×2.
        assert_eq!(d["WorldDetail"], 1.0);
        assert_eq!(
            d["frillDensity"],
            (d["WorldDetail"] + 1.0) * benilla_formats::FRILL_DENSITY as f32
        );
        let bubbles = BubbleConfig::default();
        assert_eq!(d["ChatBubbles"] != 0.0, bubbles.all);
        assert_eq!(d["ChatBubblesParty"] != 0.0, bubbles.party);
        assert!(bubbles.all && !bubbles.party, "the binary's own pair");
        let zoom = MinimapZoom::default();
        assert_eq!(d["minimapZoom"], f32::from(zoom.outdoor));
        assert_eq!(d["minimapInsideZoom"], f32::from(zoom.inside));
        assert_eq!(zoom.outdoor, benilla_ui::widget::MINIMAP_DEFAULT_ZOOM);
        // `video::tests` welds `VideoConfig::vsync` to the window's boot present mode.
        assert_eq!(d["gxVSync"] != 0.0, VideoConfig::default().vsync);
        // ── MONKEY (lighting): the weld for the dynamic light + shadow system's whole row set ──
        //
        // **All 34, as one census, with the count asserted.** Every one of them lands on
        // `VideoConfig` and is read from there per frame (`shadow_core::update_shadows`,
        // `dynamic_interior::bridge`, `torch_shadow`), so a registered default that drifts from
        // the struct's literal is a setting that reads one way in `config.toml` and renders
        // another — invisible until someone compares a fresh install against a configured one.
        // Spelling the table out rather than asserting a favourite handful is what makes "added
        // a row, forgot its weld" fail HERE: the length check below is the gate.
        let shadows = VideoConfig::default();
        let flag = |b: bool| if b { 1.0 } else { 0.0 };
        // MONKEY (volumetric fog): include the atmospheric tier in this fixed-size default table.
        let lighting: [(&str, f32); 41] = [
            // GFX (volumetric light) / (moonlight): the three new rows.
            ("volumetricLight", shadows.volumetric_light as f32),
            ("volumetricLightStrength", shadows.volumetric_light_strength),
            ("moonLight", shadows.moon_light),
            ("waterQuality", shadows.water_quality as f32),
            // MONKEY (volumetric fog): weld registry and renderer defaults.
            ("volumetricFog", shadows.volumetric_fog as f32),
            // MONKEY (lampfog): opt-in point-light fog.
            ("lampFog", shadows.lamp_fog as f32),
            // MONKEY (post): weld registry and renderer defaults.
            ("bloom", shadows.bloom as f32),
            ("sunShafts", flag(shadows.sun_shafts)),
            ("colorGrading", flag(shadows.color_grading)),
            ("lavaLightGain", shadows.lava_light_gain),
            // The two sun lanes and the cascade they share.
            ("worldShadows", flag(shadows.world_shadows)),
            ("characterShadows", flag(shadows.character_shadows)),
            ("shadowDistance", shadows.shadow_distance),
            // MONKEY (sun shadow perf): the five cost dials.
            ("shadowMapSize", shadows.shadow_map_size as f32),
            ("shadowFilter", shadows.shadow_filter as f32),
            ("characterShadowRate", shadows.character_shadow_rate as f32),
            ("worldShadowRate", shadows.world_shadow_rate as f32),
            ("shadowCasterReach", shadows.shadow_caster_reach),
            // MONKEY (moon shadows): the night lane's own shadow darkness.
            ("moonShadowStrength", shadows.moon_shadow_strength),
            // MONKEY (dynamic interiors): the room lane's switch and its light law.
            ("interiorLight", flag(shadows.interior_light)),
            ("interiorAmbient", shadows.interior_ambient),
            ("interiorFill", shadows.interior_fill),
            ("interiorExposure", shadows.interior_exposure),
            ("interiorAttenScale", shadows.interior_atten_scale),
            ("interiorRoomGate", flag(shadows.interior_room_gate)),
            ("interiorDebug", shadows.interior_debug as f32),
            // MONKEY (torch shadows): the cube-map lane, indoors and out.
            ("interiorShadows", flag(shadows.interior_shadows)),
            ("exteriorShadows", flag(shadows.exterior_shadows)),
            // MONKEY (daylight: terrain torch casters)
            ("torchTerrainShadows", flag(shadows.torch_terrain_shadows)),
            (
                "interiorShadowCasters",
                shadows.interior_shadow_casters as f32,
            ),
            (
                "interiorShadowDynamic",
                shadows.interior_shadow_dynamic as f32,
            ),
            (
                "interiorShadowEntityRate",
                shadows.interior_shadow_entity_rate as f32,
            ),
            ("interiorShadowSoft", shadows.interior_shadow_soft),
            ("torchShadowStrength", shadows.torch_shadow_strength),
            // MONKEY (darkness gains) / (enclosed day floor) / (bake floor): the four level dials.
            ("nightGain", shadows.night_gain),
            ("interiorGain", shadows.interior_gain),
            ("interiorDaylight", shadows.interior_daylight),
            ("interiorBakeFloor", shadows.interior_bake_floor),
            // MONKEY (fire GO lights) / (spellLightGain) / (flame flicker): the invented lanes.
            ("fireLightGain", shadows.fire_light_gain),
            ("spellLightGain", shadows.spell_light_gain),
            ("fireFlicker", shadows.fire_flicker),
        ];
        for (name, want) in lighting {
            assert_eq!(d[name], want, "{name}: registered default left the knob");
        }
        // …and the census is the VideoConfig-backed row set. A name here that nothing registers
        // would weld against a row the client does not have; the length is the other half — 38
        // VideoConfig rows, 38 welds (MONKEY lampfog: + lampFog). MONKEY (wind): `foliageWind` is a world resource bridge.
        let welded: std::collections::BTreeSet<&str> = lighting.iter().map(|(n, _)| *n).collect();
        // MONKEY (volumetric fog): the atmospheric tier joins the default-consumer weld.
        assert_eq!(welded.len(), 41, "the lighting lane welds 41 distinct rows");
        for name in &welded {
            assert!(
                REGISTERED.iter().any(|r| r.name == *name),
                "{name}: welded but not registered"
            );
        }
        // Three of them are CALIBRATED rather than chosen, so the weld alone is not enough — it
        // would pass just as happily if a tuning session left the struct's literal wherever it was
        // last dragged in the debug panel. These pin the measured numbers themselves.
        assert_eq!(d["interiorGain"], 0.5, "MONKEY (lighting debug panel)");
        // MONKEY (bake floor): the inn's black door band measured up to 0.108 x tex with
        // candle-lit surfaces moving under +10 % — only true at this number.
        assert_eq!(d["interiorBakeFloor"], 0.12);
        assert_eq!(d["spellLightGain"], 1.0, "the spell lane ships neutral");
        assert_eq!(d["waterQuality"], 1.0, "mirror reflections stay opt-in");
        assert_eq!(d["lavaLightGain"], 1.0);
        // MONKEY (ao): registry and renderer agree, and the lane ships Off.
        assert_eq!(d["ambientOcclusion"], shadows.ambient_occlusion as f32);
        assert_eq!(d["ambientOcclusion"], 0.0, "ambient occlusion is opt-in");
        // `lightingQuality` and MONKEY (wind) `foliageWind` deliberately are not in that census:
        // neither is a `VideoConfig` knob. The former names the rows above; its own weld is
        // `the_high_preset_is_the_registered_defaults`. The latter bridges directly to the
        // world-owned `FoliageWind` resource in `monkey_gfx`.
        // ── end MONKEY (lighting) ─────────────────────────────────────────────────────────────
        assert_eq!(d["boothHalfRate"] != 0.0, PaneRate::default().half);
        // Every visual golden assumes a 1:1 backdrop.
        assert_eq!(d["renderScale"], 1.0);
    }

    /// The FFX pass's three switches reach it one by one (`ffx`, `ffxGlow`, `ffxDeath`).
    #[test]
    fn the_ffx_switches_reach_the_pass() {
        use benilla_world::ffx_glow::FfxSwitches;
        let mut app = cvar_app();
        let on = FfxSwitches::default();
        assert_eq!(*res::<FfxSwitches>(&app), on);
        apply(&mut app, "ffxGlow", "0");
        assert_eq!(*res::<FfxSwitches>(&app), FfxSwitches { glow: false, ..on });
        apply(&mut app, "ffxglow", "1");
        apply(&mut app, "ffxDeath", "0");
        assert_eq!(
            *res::<FfxSwitches>(&app),
            FfxSwitches { death: false, ..on }
        );
        apply(&mut app, "ffxDeath", "1");
        apply(&mut app, "ffx", "0");
        assert_eq!(
            *res::<FfxSwitches>(&app),
            FfxSwitches {
                master: false,
                ..on
            }
        );
    }

    /// `gxMaximize` registers latched (flags 3): a write is staged and moves nothing until the
    /// `RestartGx` commit, where the window knob takes it.
    #[test]
    fn gx_maximize_waits_for_the_restart_commit() {
        let mut app = cvar_app();
        let outcome = app
            .world_mut()
            .resource_mut::<Cvars>()
            .set("gxMaximize", "1");
        assert_eq!(outcome, SetOutcome::Staged);
        let events = app.world_mut().resource_mut::<Cvars>().take_events();
        assert!(events.is_empty(), "a staged write fires no callback");
        assert!(!res::<VideoConfig>(&app).maximize);
        assert_eq!(app.world_mut().resource_mut::<Cvars>().commit_latched(), 1);
        let events = app.world_mut().resource_mut::<Cvars>().take_events();
        for event in events {
            app.world_mut().trigger(event);
        }
        assert!(res::<VideoConfig>(&app).maximize, "the commit applies it");
    }

    // ── MONKEY (advanced graphics): the preset ladder ─────────────────────────────────────────

    /// A registry holding nothing but the registered defaults — the state a fresh
    /// `benilla-config` boots into, which is what every claim below is measured against.
    fn fresh_registry() -> Cvars {
        Cvars::default()
    }

    /// **The round trip, for every rung**: apply a preset, derive, and get that preset's own name
    /// back. This is the property the whole ladder rests on — without it the dropdown would show
    /// `Custom` the instant after the player picked something, which is the exact failure the
    /// "derived, never stored stale" rule exists to prevent.
    #[test]
    fn every_preset_derives_back_to_its_own_name() {
        for (from, _) in LIGHTING_PRESETS {
            for (name, _) in LIGHTING_PRESETS {
                let mut cvars = fresh_registry();
                apply_lighting_preset(&mut cvars, from);
                cvars.set("shadowDistance", "120");
                cvars.set("interiorShadowSoft", "2.5");
                assert!(apply_lighting_preset(&mut cvars, name), "{name}: applied");
                assert_eq!(derive_lighting_quality(&cvars), *name, "{from} -> {name}");
                assert_eq!(cvars.get("shadowDistance"), Some("120"));
                assert_eq!(cvars.get("interiorShadowSoft"), Some("2.5"));
                // MONKEY (volumetric fog): Off must remove atmosphere as well as enhanced water.
                for member in [
                    "fireLightGain",
                    "waterQuality",
                    "volumetricFog",
                    "lavaLightGain",
                    "nightGain",
                    "interiorGain",
                ] {
                    let expected = if *name == "Off" {
                        if matches!(
                            member,
                            "fireLightGain" | "waterQuality" | "volumetricFog" | "lavaLightGain"
                        ) {
                            "0"
                        } else {
                            "1.0"
                        }
                    } else if *name == "Ultra" && matches!(member, "waterQuality" | "volumetricFog")
                    {
                        "2" // MONKEY (presets): Ultra's two maxima over the defaults.
                    } else {
                        cvars.default_of(member).unwrap()
                    };
                    assert!(
                        same_value(cvars.get(member).unwrap(), expected),
                        "{name}/{member}"
                    );
                }
            }
        }
    }

    /// **High is the shipped default, member for member.** A fresh config must read "High" on this
    /// row rather than "Custom" — otherwise the very first thing a player sees on the page is a
    /// word that says their settings are a hand-made combination when they have touched nothing.
    #[test]
    fn the_high_preset_is_the_registered_defaults() {
        let cvars = fresh_registry();
        assert_eq!(derive_lighting_quality(&cvars), "High");
        assert_eq!(
            cvars.get("lightingQuality"),
            Some("High"),
            "the registered default agrees with what a fresh registry derives"
        );
        // …and the table is not merely CONSISTENT with the defaults, it IS them: a member whose
        // registered default moved without its preset entry moving would still derive "High"
        // above (both sides move together), so the weld is checked against the row's own default.
        let (_, members) = LIGHTING_PRESETS
            .iter()
            .find(|(n, _)| *n == "High")
            .expect("the High rung");
        for (k, v) in *members {
            assert!(
                same_value(cvars.default_of(k).expect("a registered member"), v),
                "{k}: the High preset says {v}, the registry's default says {:?}",
                cvars.default_of(k)
            );
        }
    }

    /// **One member off the rung is `Custom`** — for every rung, and for every member of it, which
    /// is the half that keeps the derivation from being a lucky match on a favourite row.
    #[test]
    fn one_changed_member_derives_custom() {
        for (name, members) in LIGHTING_PRESETS {
            for (k, v) in *members {
                let mut cvars = fresh_registry();
                apply_lighting_preset(&mut cvars, name);
                // Somewhere else on the row's own scale: the flags flip, the numbers move by one.
                let moved = match v.trim().parse::<f32>() {
                    Ok(x) if x == 0.0 => "1".to_string(),
                    Ok(x) if x == 1.0 => "0".to_string(),
                    Ok(x) => (x + 1.0).to_string(),
                    Err(_) => format!("{v}x"),
                };
                assert_eq!(cvars.set(k, &moved), SetOutcome::Changed, "{name}/{k}");
                assert_eq!(
                    derive_lighting_quality(&cvars),
                    LIGHTING_CUSTOM,
                    "{name}: {k} moved to {moved} and the ladder still says {name}"
                );
            }
        }
    }

    /// A value that says the same NUMBER is the same setting: the options window's sliders write
    /// `"1"`, a hand-edited config may say `"1.0"`, and `%.2f` formatting elsewhere says `"1.00"`.
    /// A ladder that compared strings would call every one of those `Custom`.
    #[test]
    fn the_ladder_compares_numbers_not_spellings() {
        let mut cvars = fresh_registry();
        apply_lighting_preset(&mut cvars, "High");
        for spelling in ["1.0", "1.00", "1"] {
            cvars.set("spellLightGain", spelling);
            assert_eq!(derive_lighting_quality(&cvars), "High", "{spelling}");
        }
        cvars.set("moonShadowStrength", "0.350");
        assert_eq!(derive_lighting_quality(&cvars), "High");
    }

    /// `Custom` is not a rung: picking it writes nothing, so the members — and therefore the name
    /// the next derivation lands on — are exactly where they were.
    #[test]
    fn picking_custom_writes_nothing() {
        let mut cvars = fresh_registry();
        apply_lighting_preset(&mut cvars, "Medium");
        assert!(!apply_lighting_preset(&mut cvars, LIGHTING_CUSTOM));
        assert!(!apply_lighting_preset(&mut cvars, "Epic"));
        assert_eq!(derive_lighting_quality(&cvars), "Medium");
    }

    // ── MONKEY (presets): the Graphics Preset ladder ──────────────────────────────────────────

    /// Every governed row is registered and names no row the lighting ladder already owns — the
    /// disjointness that lets both labels be right at once.
    #[test]
    fn graphics_rows_are_registered_and_disjoint_from_the_lighting_ladder() {
        assert_eq!(GRAPHICS_PRESETS.len(), 13, "add the row count with the row");
        let cvars = fresh_registry();
        for (k, values) in GRAPHICS_PRESETS {
            assert!(cvars.get(k).is_some(), "{k}: not registered");
            for (rung, members) in LIGHTING_PRESETS {
                assert!(
                    members.iter().all(|(m, _)| !m.eq_ignore_ascii_case(k)),
                    "{k} is also a member of lighting {rung}"
                );
            }
            if *k == "lightingQuality" {
                for v in values {
                    assert!(
                        LIGHTING_PRESETS.iter().any(|(n, _)| n == v),
                        "{v}: a lighting rung"
                    );
                }
            }
        }
        // The rungs are distinct columns, so no two can derive to the same name.
        for a in 0..GRAPHICS_PRESET_NAMES.len() {
            for b in a + 1..GRAPHICS_PRESET_NAMES.len() {
                assert!(
                    GRAPHICS_PRESETS
                        .iter()
                        .any(|(_, v)| !same_value(v[a], v[b])),
                    "{} and {} are the same column",
                    GRAPHICS_PRESET_NAMES[a],
                    GRAPHICS_PRESET_NAMES[b]
                );
            }
        }
    }

    /// **Preset → rows → the same preset**, from every rung to every rung, through the ordinary
    /// `set` path — with the two rows no preset decides left where the player put them.
    #[test]
    fn every_graphics_preset_derives_back_to_its_own_name() {
        for from in GRAPHICS_PRESET_NAMES {
            for name in GRAPHICS_PRESET_NAMES {
                let mut cvars = fresh_registry();
                cvars.set("graphicsQuality", from);
                cvars.set("shadowDistance", "120");
                cvars.set("interiorShadowSoft", "2.5");
                cvars.set("graphicsQuality", name);
                lighting_quality(&mut cvars);
                assert_eq!(derive_graphics_quality(&cvars), name, "{from} -> {name}");
                assert_eq!(cvars.get("graphicsQuality"), Some(name));
                let col = graphics_column(name).unwrap();
                assert_eq!(
                    cvars.get("lightingQuality"),
                    Some(GRAPHICS_PRESETS[0].1[col])
                );
                assert_eq!(cvars.get("shadowDistance"), Some("120"));
                assert_eq!(cvars.get("interiorShadowSoft"), Some("2.5"));
            }
        }
    }

    /// Classic is the reference client: the lighting ladder's Off, every programme lane off and
    /// `farclip` at the reference's registered 350.
    #[test]
    fn the_classic_preset_is_every_lane_off_at_the_reference_view_distance() {
        let mut cvars = fresh_registry();
        assert!(apply_graphics_preset(&mut cvars, "Classic"));
        assert_eq!(cvars.get("lightingQuality"), Some("Off"));
        assert_eq!(cvars.get("farclip"), cvars.default_of("farclip"));
        for (k, values) in GRAPHICS_PRESETS {
            if !matches!(*k, "lightingQuality" | "farclip") {
                assert!(same_value(values[0], "0"), "{k}: Classic leaves it on");
            }
        }
        // Ultra is the one rung past the reference's 777, and inside the extended clamp.
        let ultra = GRAPHICS_PRESETS
            .iter()
            .find(|(k, _)| *k == "farclip")
            .unwrap()
            .1[4];
        let ultra: f32 = ultra.parse().unwrap();
        assert!(ultra > 777.0 && FARCLIP_RANGE.contains(&ultra));
    }

    /// **Editing one governed row is Custom** — for every rung and every row of it, the lighting
    /// rung included (moved by one of ITS members, as the page's lighting rows would).
    #[test]
    fn one_changed_graphics_row_derives_custom() {
        for name in GRAPHICS_PRESET_NAMES {
            let col = graphics_column(name).unwrap();
            for (k, values) in GRAPHICS_PRESETS {
                let mut cvars = fresh_registry();
                apply_graphics_preset(&mut cvars, name);
                let (row, moved) = if *k == "lightingQuality" {
                    let to = if values[col] == "Off" { "1" } else { "0" };
                    ("characterShadows", to.to_string())
                } else {
                    let v: f32 = values[col].parse().unwrap();
                    let to = if *k == "farclip" {
                        v + 60.0
                    } else if v == 0.0 {
                        1.0
                    } else {
                        0.0
                    };
                    (*k, to.to_string())
                };
                assert_eq!(cvars.set(row, &moved), SetOutcome::Changed, "{name}/{row}");
                lighting_quality(&mut cvars);
                assert_eq!(
                    cvars.get("graphicsQuality"),
                    Some(LIGHTING_CUSTOM),
                    "{name}/{row}"
                );
                // …and picking the rung again puts every row back.
                assert!(apply_graphics_preset(&mut cvars, name));
                lighting_quality(&mut cvars);
                assert_eq!(
                    cvars.get("graphicsQuality"),
                    Some(name),
                    "{name}/{row} re-pick"
                );
            }
        }
        let mut cvars = fresh_registry();
        assert!(!apply_graphics_preset(&mut cvars, LIGHTING_CUSTOM));
        assert!(!apply_graphics_preset(&mut cvars, "Epic"));
    }

    /// **A new player boots into High**: the seed writes the High column over a registry with no
    /// file, and the registered default of the label agrees, so it never reaches `config.toml`.
    #[test]
    fn a_fresh_player_is_seeded_to_the_high_graphics_preset() {
        assert_eq!(
            fresh_registry().default_of("graphicsQuality"),
            Some(GRAPHICS_DEFAULT)
        );
        let mut cvars = fresh_registry();
        assert!(cvars.seed_graphics_preset() > 0);
        lighting_quality(&mut cvars);
        assert_eq!(cvars.get("graphicsQuality"), Some("High"));
        assert_eq!(cvars.get("lightingQuality"), Some("High"));
        assert_eq!(cvars.get("farclip"), Some("777"));
        assert!(
            !cvars.compose().contains_key("graphicsQuality"),
            "the default label is not saved"
        );
        // A second boot over what the first one saved seeds nothing.
        let mut again = fresh_registry();
        again.load_file(cvars.compose());
        assert_eq!(again.seed_graphics_preset(), 0);
        lighting_quality(&mut again);
        assert_eq!(again.get("graphicsQuality"), Some("High"));
    }

    /// MONKEY (followups): a saved Graphics rung fills the governed rows the file lacks (a row
    /// registered after the save) with the rung's value; rows the file carries stay; Custom and
    /// no saved name write nothing.
    #[test]
    fn a_saved_graphics_preset_fills_rows_missing_from_the_file() {
        // An Ultra save made before `ambientOcclusion` / `lampFog` existed.
        let file = BTreeMap::from([
            ("graphicsQuality".to_string(), "Ultra".to_string()),
            ("lightingQuality".to_string(), "Ultra".to_string()),
            ("farclip".to_string(), "1497".to_string()),
            ("zoneSkyboxes".to_string(), "0".to_string()), // the player's own later edit
        ]);
        let mut cvars = fresh_registry();
        cvars.load_file(file);
        assert!(cvars.reapply_saved_presets() > 0);
        assert_eq!(cvars.get("ambientOcclusion"), Some("2"));
        assert_eq!(cvars.get("lampFog"), Some("2"));
        assert_eq!(cvars.get("farclip"), Some("1497"));
        assert_eq!(cvars.get("zoneSkyboxes"), Some("0"), "a saved row stays");
        // The lighting members follow Ultra too.
        assert_eq!(cvars.get("shadowMapSize"), Some("4096"));
        assert_eq!(cvars.get("volumetricFog"), Some("2"));

        // Medium without a saved lighting rung: the Graphics rung's lighting rung applies.
        let mut medium = fresh_registry();
        medium.load_file(BTreeMap::from([(
            "graphicsQuality".to_string(),
            "Medium".to_string(),
        )]));
        medium.reapply_saved_presets();
        assert_eq!(medium.get("ambientOcclusion"), Some("1"));
        assert_eq!(medium.get("farclip"), Some("477"));
        lighting_quality(&mut medium);
        assert_eq!(medium.get("lightingQuality"), Some("Medium"));
        assert_eq!(medium.get("graphicsQuality"), Some("Medium"));

        // Custom writes nothing; neither does a file with no preset name.
        for file in [
            BTreeMap::from([("graphicsQuality".to_string(), LIGHTING_CUSTOM.to_string())]),
            BTreeMap::from([("skyQuality".to_string(), "1".to_string())]),
        ] {
            let mut custom = fresh_registry();
            custom.load_file(file);
            assert_eq!(custom.reapply_saved_presets(), 0);
            assert_eq!(
                custom.get("ambientOcclusion"),
                custom.default_of("ambientOcclusion")
            );
        }

        // A session-owned row is left to the session.
        let mut owned = fresh_registry();
        owned.own_for_session("farclip", Some("500"));
        owned.load_file(BTreeMap::from([(
            "graphicsQuality".to_string(),
            "Ultra".to_string(),
        )]));
        owned.reapply_saved_presets();
        assert_eq!(owned.get("farclip"), Some("500"));
        assert_eq!(owned.get("ambientOcclusion"), Some("2"));
    }

    /// MONKEY (followups): a saved Lighting Quality rung fills the lighting members the file
    /// lacks; members the file carries stay; Custom writes nothing.
    #[test]
    fn a_saved_lighting_preset_fills_members_missing_from_the_file() {
        let mut cvars = fresh_registry();
        cvars.load_file(BTreeMap::from([
            ("lightingQuality".to_string(), "Ultra".to_string()),
            ("volumetricFog".to_string(), "1".to_string()),
        ]));
        assert!(cvars.reapply_saved_presets() > 0);
        assert_eq!(cvars.get("shadowMapSize"), Some("4096"));
        assert_eq!(cvars.get("interiorShadowCasters"), Some("16"));
        assert_eq!(
            cvars.get("volumetricFog"),
            Some("1"),
            "a saved member stays"
        );
        // No Graphics rung saved: the graphics rows keep their defaults.
        assert_eq!(
            cvars.get("ambientOcclusion"),
            cvars.default_of("ambientOcclusion")
        );

        let mut off = fresh_registry();
        off.load_file(BTreeMap::from([(
            "lightingQuality".to_string(),
            "Off".to_string(),
        )]));
        off.reapply_saved_presets();
        assert_eq!(off.get("worldShadows"), Some("0"));
        lighting_quality(&mut off);
        assert_eq!(off.get("lightingQuality"), Some("Off"));

        let mut custom = fresh_registry();
        custom.load_file(BTreeMap::from([(
            "lightingQuality".to_string(),
            LIGHTING_CUSTOM.to_string(),
        )]));
        assert_eq!(custom.reapply_saved_presets(), 0);
    }

    /// The seed never overrides the player: a saved preset stops it outright, a saved row keeps
    /// its value, and a session-owned row (an env lever) is left to the session.
    #[test]
    fn the_graphics_seed_leaves_saved_and_session_rows_alone() {
        let mut saved = fresh_registry();
        saved.load_file(BTreeMap::from([(
            "graphicsQuality".into(),
            "Classic".into(),
        )]));
        assert_eq!(saved.seed_graphics_preset(), 0);
        assert_eq!(saved.get("skyQuality"), saved.default_of("skyQuality"));

        let mut row = fresh_registry();
        row.load_file(BTreeMap::from([("skyQuality".into(), "1".into())]));
        row.own_for_session("farclip", Some("500"));
        row.seed_graphics_preset();
        lighting_quality(&mut row);
        assert_eq!(row.get("skyQuality"), Some("1"));
        assert_eq!(row.get("farclip"), Some("500"));
        assert_eq!(row.get("fogModel"), Some("1"));
        assert_eq!(row.get("graphicsQuality"), Some(LIGHTING_CUSTOM));
    }

    /// Ultra's `farclip` survives the real observer (the extended clamp), and every governed
    /// row's rung values do too.
    #[test]
    fn graphics_rows_pass_their_observers_unclamped() {
        // MONKEY (reviewfix-a): every governed row, every rung — not only `farclip`. Each value
        // is driven through the real setter and read back off the resource its renderer reads.
        let mut app = cvar_app();
        for (col, name) in GRAPHICS_PRESET_NAMES.iter().enumerate() {
            for (k, values) in GRAPHICS_PRESETS {
                if *k == "lightingQuality" {
                    continue; // its members are welded by the lighting-preset census test
                }
                let value = values[col];
                apply(&mut app, k, "-1");
                apply(&mut app, k, value);
                let video = res::<VideoConfig>(&app);
                let flag = |b: bool| if b { 1.0 } else { 0.0 };
                let applied = match *k {
                    "farclip" => res::<ViewDistance>(&app).farclip,
                    "skyQuality" => video.sky_quality as f32,
                    "skyDither" => flag(res::<benilla_world::ffx_glow::SkyDither>(&app).0),
                    "fogModel" => {
                        flag(res::<benilla_world::lighting::FogModelSetting>(&app).modern())
                    }
                    "rainSurfaces" => flag(res::<benilla_world::weather::RainSurfaces>(&app).0),
                    "torchTerrainShadows" => flag(video.torch_terrain_shadows),
                    "daylightWindowSplit" => flag(video.daylight_window_split),
                    "ambientOcclusion" => video.ambient_occlusion as f32,
                    "zoneSkyboxes" => flag(res::<benilla_world::skybox::ZoneSkyboxes>(&app).0),
                    "lampFog" => video.lamp_fog as f32,
                    // GFX (volumetric light) / (moonlight)
                    "volumetricLight" => video.volumetric_light as f32,
                    "moonLight" => video.moon_light,
                    _ => panic!("add the observer readback for {k}"),
                };
                assert_eq!(applied, value.parse::<f32>().unwrap(), "{name}/{k}");
            }
        }
    }

    /// Every member every rung names is a REGISTERED row — a typo'd member would otherwise make
    /// its whole preset underivable (a `cvars.get` that answers `None` never matches), and the
    /// only symptom would be a dropdown stuck on `Custom`.
    #[test]
    fn every_preset_member_is_registered_and_inside_its_observers_clamp() {
        let mut app = cvar_app();
        for (name, members) in LIGHTING_PRESETS {
            for (k, value) in *members {
                assert!(
                    REGISTERED.iter().any(|r| r.name == *k),
                    "{name}: member {k} is not registered"
                );
                // Drive the real setter: a clamp, integer conversion or map-size snap
                // must leave every preset value unchanged.
                apply(&mut app, k, "-1");
                apply(&mut app, k, value);
                let video = res::<VideoConfig>(&app);
                let applied = match *k {
                    "characterShadows" => video.character_shadows as u32 as f32,
                    "worldShadows" => video.world_shadows as u32 as f32,
                    "interiorLight" => video.interior_light as u32 as f32,
                    "interiorShadows" => video.interior_shadows as u32 as f32,
                    "exteriorShadows" => video.exterior_shadows as u32 as f32,
                    "shadowMapSize" => video.shadow_map_size as f32,
                    "interiorShadowCasters" => video.interior_shadow_casters as f32,
                    "interiorShadowDynamic" => video.interior_shadow_dynamic as f32,
                    "spellLightGain" => video.spell_light_gain,
                    "waterQuality" => video.water_quality as f32,
                    // MONKEY (volumetric fog): verify presets reach the renderer.
                    "volumetricFog" => video.volumetric_fog as f32,
                    // MONKEY (post): the post lane's live tier.
                    "bloom" => video.bloom as f32,
                    "sunShafts" => video.sun_shafts as u32 as f32,
                    "colorGrading" => video.color_grading as u32 as f32,
                    // MONKEY (integration): wind is bridged directly to its world resource rather
                    // than through VideoConfig, but it is still a preset member and needs the same
                    // real-observer clamp weld as every VideoConfig-backed row above.
                    "foliageWind" => res::<benilla_world::wind::FoliageWind>(&app).0 as f32,
                    "lavaLightGain" => video.lava_light_gain,
                    "fireLightGain" => video.fire_light_gain,
                    "moonShadowStrength" => video.moon_shadow_strength,
                    "fireFlicker" => video.fire_flicker,
                    "nightGain" => video.night_gain,
                    "interiorGain" => video.interior_gain,
                    _ => panic!("add the observer readback for {k}"),
                };
                assert_eq!(applied, value.parse::<f32>().unwrap(), "{name}/{k}");
            }
        }
    }

    #[test]
    fn preset_then_member_edit_survives_sync_observers_and_later_flushes() {
        for seeded in [false, true] {
            for flush_preset_first in [false, true] {
                let mut app = cvar_app();
                app.world_mut()
                    .non_send_resource_mut::<UiScript>()
                    .register_cvars(registered_pairs());
                // Run the production Update schedule without Startup's on-disk config load.
                if seeded {
                    app.world_mut().run_schedule(Update);
                }
                app.world_mut()
                    .non_send_resource_mut::<UiScript>()
                    .run("SetCVar('lightingQuality', 'Low')")
                    .unwrap();
                if flush_preset_first {
                    app.world_mut().run_schedule(Update);
                    assert_eq!(res::<VideoConfig>(&app).shadow_map_size, 1024);
                }
                app.world_mut()
                    .non_send_resource_mut::<UiScript>()
                    .run("SetCVar('shadowMapSize', '4096')")
                    .unwrap();
                for _ in 0..3 {
                    app.world_mut().run_schedule(Update);
                    let cvars = res::<Cvars>(&app);
                    assert_eq!(cvars.get("shadowMapSize"), Some("4096"));
                    assert_eq!(cvars.get("lightingQuality"), Some(LIGHTING_CUSTOM));
                    assert_eq!(res::<VideoConfig>(&app).shadow_map_size, 4096);
                    let script = app.world().non_send_resource::<UiScript>();
                    assert_eq!(script.cvar("shadowMapSize").as_deref(), Some("4096"));
                    assert_eq!(
                        script.cvar("lightingQuality").as_deref(),
                        Some(LIGHTING_CUSTOM)
                    );
                }
            }
        }
    }

    #[test]
    fn fire_spell_and_lava_gains_above_two_reach_the_video_observer() {
        let mut app = cvar_app();
        for value in ["3.25", "4"] {
            apply(&mut app, "fireLightGain", value);
            apply(&mut app, "spellLightGain", value);
            apply(&mut app, "lavaLightGain", value);
            let expected = value.parse::<f32>().unwrap();
            assert_eq!(res::<VideoConfig>(&app).fire_light_gain, expected);
            assert_eq!(res::<VideoConfig>(&app).spell_light_gain, expected);
            assert_eq!(res::<VideoConfig>(&app).lava_light_gain, expected);
        }
    }

    #[test]
    fn water_and_lava_observers_clamp_and_water_high_is_opt_in() {
        let mut app = cvar_app();
        for (value, water, lava) in [
            ("-1", 0, 0.0),
            ("1.75", 1, 1.75),
            ("2", 2, 2.0),
            ("9", 2, 4.0),
        ] {
            apply(&mut app, "waterQuality", value);
            apply(&mut app, "lavaLightGain", value);
            assert_eq!(res::<VideoConfig>(&app).water_quality, water);
            assert_eq!(res::<VideoConfig>(&app).lava_light_gain, lava);
        }
        let mut cvars = fresh_registry();
        cvars.set("waterQuality", "2");
        assert_eq!(derive_lighting_quality(&cvars), LIGHTING_CUSTOM);
        apply_lighting_preset(&mut cvars, "High");
        assert_eq!(cvars.get("waterQuality"), Some("1"));
    }

    #[test]
    fn the_observers_apply_every_arm() {
        let mut app = cvar_app();
        apply(&mut app, "MusicVolume", "0.7");
        assert_eq!(res::<SoundConfig>(&app).music, 0.7);
        // A string row reaches its knob; a value that is not an address leaves the knob alone.
        apply(&mut app, "realmList", "logon.example.org:3724");
        assert_eq!(
            res::<crate::realmlist::Realmlist>(&app).address(),
            "logon.example.org:3724"
        );
        apply(
            &mut app,
            "realmlist",
            r#"SET realmlist "elsewhere.example.org""#,
        );
        assert_eq!(
            res::<crate::realmlist::Realmlist>(&app).address(),
            "elsewhere.example.org"
        );
        apply(&mut app, "realmList", "not an address");
        assert_eq!(
            res::<crate::realmlist::Realmlist>(&app).address(),
            "elsewhere.example.org",
            "a known key with a bad value is consumed, and the resource keeps its truth",
        );
        // Clamps are the knob's own: volume to [0,1], farclip to FARCLIP_RANGE.
        apply(&mut app, "mastervolume", "7");
        assert_eq!(res::<SoundConfig>(&app).master, 1.0);
        apply(&mut app, "farclip", "50");
        assert_eq!(res::<ViewDistance>(&app).farclip, *FARCLIP_RANGE.start());
        // `nearclip` clamps to the callback's `[0.01, 0.33]` (`0x688d90`).
        apply(&mut app, "nearclip", "0.001");
        assert_eq!(
            res::<ViewDistance>(&app).nearclip,
            0.01,
            "[0x8029d0], the callback's low bound"
        );
        apply(&mut app, "nearclip", "9");
        assert_eq!(
            res::<ViewDistance>(&app).nearclip,
            0.33,
            "[0x808300], its high bound"
        );
        apply(&mut app, "nearclip", "0.3");
        assert_eq!(res::<ViewDistance>(&app).nearclip, 0.3);
        // A sample count, clamped to the reference's [1, 16]; 1 is none.
        apply(&mut app, "gxMultisample", "4");
        assert_eq!(res::<MsaaSetting>(&app).samples, 4);
        apply(&mut app, "anisotropic", "99");
        assert_eq!(
            res::<benilla_assets::TexFilterSetting>(&app).aniso,
            *benilla_assets::ANISO_RANGE.end()
        );
        apply(&mut app, "anisotropic", "0");
        assert_eq!(
            res::<benilla_assets::TexFilterSetting>(&app).aniso,
            *benilla_assets::ANISO_RANGE.start()
        );
        // The knob ships on, so only the flip to 0 proves the arm.
        apply(&mut app, "trilinear", "0");
        assert!(!res::<benilla_assets::TexFilterSetting>(&app).trilinear);
        apply(&mut app, "trilinear", "1");
        assert!(res::<benilla_assets::TexFilterSetting>(&app).trilinear);
        // Clamped to the device's ceiling too: this GPU offers 4, and wgpu refuses a count it
        // lacks.
        apply(&mut app, "gxmultisample", "99");
        assert_eq!(res::<MsaaSetting>(&app).samples, 4);
        // The realistic route in: a config written where 8x exists, opened where it does not.
        apply(&mut app, "gxMultisample", "8");
        assert_eq!(
            res::<MsaaSetting>(&app).samples,
            4,
            "a device that stops at 4x must never be handed an 8"
        );
        apply(&mut app, "gxmultisample", "2");
        assert_eq!(res::<MsaaSetting>(&app).samples, 2);
        apply(&mut app, "gxmultisample", "0");
        assert_eq!(res::<MsaaSetting>(&app).samples, *MSAA_RANGE.start());
        apply(&mut app, "renderScale", "0.75");
        assert_eq!(res::<RenderScale>(&app).0, 0.75);
        apply(&mut app, "renderscale", "9");
        assert_eq!(res::<RenderScale>(&app).0, *RENDER_SCALE_RANGE.end());
        apply(&mut app, "renderscale", "0");
        assert_eq!(res::<RenderScale>(&app).0, *RENDER_SCALE_RANGE.start());
        assert!(!res::<crate::perf::FpsJournalSetting>(&app).0);
        apply(&mut app, "fpsJournal", "1");
        assert!(res::<crate::perf::FpsJournalSetting>(&app).0);
        apply(&mut app, "fpsjournal", "0");
        assert!(!res::<crate::perf::FpsJournalSetting>(&app).0);
        // Enable flags: any nonzero is on, zero is off (the client's int-parse + != 0).
        apply(&mut app, "EnableMusic", "0");
        assert!(!res::<SoundConfig>(&app).music_enabled);
        apply(&mut app, "mastersoundeffects", "1");
        assert!(res::<SoundConfig>(&app).enabled);
        apply(&mut app, "deselectonclick", "0");
        assert!(!res::<ClickConfig>(&app).deselect_on_click);
        apply(&mut app, "MouseInvertPitch", "1");
        assert!(res::<LookConfig>(&app).invert_pitch);
        apply(&mut app, "mousespeed", "1.4");
        assert_eq!(res::<LookConfig>(&app).sensitivity, 1.4);
        apply(&mut app, "mousespeed", "9");
        assert_eq!(res::<LookConfig>(&app).sensitivity, 1.5);
        // The engine's enum (0 Never, 1 Smart, 2 Always); 3, the validator's upper bound, is Never.
        apply(&mut app, "cameraSmoothStyle", "0");
        assert_eq!(res::<FollowConfig>(&app).style, FollowStyle::Never);
        apply(&mut app, "camerasmoothstyle", "2");
        assert_eq!(res::<FollowConfig>(&app).style, FollowStyle::Always);
        apply(&mut app, "cameraSmoothStyle", "3");
        assert_eq!(res::<FollowConfig>(&app).style, FollowStyle::Never);
        apply(&mut app, "cameraSmoothStyle", "1");
        assert_eq!(res::<FollowConfig>(&app).style, FollowStyle::Smart);
        apply(&mut app, "cameraSmoothTrackingStyle", "2");
        assert_eq!(
            res::<FollowConfig>(&app).tracking_style,
            FollowStyle::Always
        );
        assert_eq!(
            res::<FollowConfig>(&app).style,
            FollowStyle::Smart,
            "and only that one"
        );
        apply(&mut app, "cameraYawSmoothSpeed", "270");
        assert_eq!(res::<FollowConfig>(&app).yaw_speed, 270.0);
        apply(&mut app, "cameraYawSmoothSpeed", "9000");
        assert_eq!(
            res::<FollowConfig>(&app).yaw_speed,
            *FOLLOW_SPEED_RANGE.end()
        );
        // The max orbit lands as yards on the knob: `cameraDistanceMax` x the factor, held to
        // [0, 50]; the factor is not held to its slider's 1 to 2.
        apply(&mut app, "cameraDistanceMaxFactor", "1");
        assert_eq!(res::<ZoomLimit>(&app).max, 15.0);
        apply(&mut app, "cameradistancemaxfactor", "3");
        assert_eq!(res::<ZoomLimit>(&app).max, 45.0);
        apply(&mut app, "cameraDistanceMaxFactor", "1");
        apply(&mut app, "cameraDistanceMax", "25");
        assert_eq!(res::<ZoomLimit>(&app).max, 25.0);
        apply(&mut app, "cameraDistanceMaxFactor", "3");
        assert_eq!(res::<ZoomLimit>(&app).max, 50.0);
        apply(&mut app, "cameraDistanceMaxFactor", "-1");
        assert_eq!(res::<ZoomLimit>(&app).max, 0.0);
        apply(&mut app, "cameraDistanceMaxFactor", "nan");
        assert_eq!(res::<ZoomLimit>(&app).max, 50.0);
        // Out of `cameraDistanceMax`'s validator range: refused at the knob, 25 stands.
        apply(&mut app, "cameraDistanceMaxFactor", "1");
        apply(&mut app, "cameraDistanceMax", "60");
        assert_eq!(res::<ZoomLimit>(&app).max, 25.0);
        apply(&mut app, "autoLootDefault", "1");
        assert!(res::<LootConfig>(&app).auto_loot);
        apply(&mut app, "showLootSpam", "0");
        assert!(!res::<LootConfig>(&app).show_loot_spam);
        apply(&mut app, "guildMemberNotify", "1");
        assert!(res::<crate::ui_guild::GuildMemberNotify>(&app).0);
        apply(&mut app, "BlockTrades", "1");
        assert!(res::<crate::ui_trade::BlockTrades>(&app).0);
        apply(&mut app, "UnitNameNPC", "0");
        assert!(!res::<NameConfig>(&app).npc);
        apply(&mut app, "unitnameown", "1");
        assert!(res::<NameConfig>(&app).own);
        apply(&mut app, "ChatBubbles", "0");
        assert!(!res::<BubbleConfig>(&app).all);
        apply(&mut app, "chatbubblesparty", "0");
        assert!(!res::<BubbleConfig>(&app).party);
        // WorldDetail: stop 0/1/2 is density ×1/×2/×3, clamped to the slider's range.
        apply(&mut app, "WorldDetail", "0");
        assert_eq!(res::<ClutterConfig>(&app).density, 1.0);
        apply(&mut app, "worlddetail", "7");
        assert_eq!(res::<ClutterConfig>(&app).density, 3.0);
        // `frillDensity` is the same field in cells per chunk, with its own clamp `[1, 256]`
        // (`0x688de0`); the stops round-trip through both spellings.
        apply(&mut app, "frillDensity", "48");
        assert_eq!(res::<ClutterConfig>(&app).density, 3.0);
        apply(&mut app, "frilldensity", "16");
        assert_eq!(res::<ClutterConfig>(&app).density, 1.0);
        // Past the top stop is honoured; pfUI's `hdgraphic` writes up to 256.
        apply(&mut app, "frillDensity", "256");
        assert_eq!(res::<ClutterConfig>(&app).density, 16.0);
        // `0` is not clutter-off: the callback pins it to 1.
        apply(&mut app, "frillDensity", "9000");
        assert_eq!(res::<ClutterConfig>(&app).density, 16.0);
        apply(&mut app, "frillDensity", "0");
        assert_eq!(res::<ClutterConfig>(&app).density, 1.0 / 16.0);
        apply(&mut app, "WorldDetail", "1");
        assert_eq!(res::<ClutterConfig>(&app).density, 2.0);
        for (wrote, want) in [
            ("0", 0u8),
            ("1", 1),
            ("2", 2),
            ("3", 3),
            ("2.9", 2),
            ("9", 3),
            ("-4", 0),
        ] {
            apply(&mut app, "weatherDensity", wrote);
            assert_eq!(
                res::<benilla_world::weather::WeatherState>(&app).weather_density,
                want,
                "weatherDensity {wrote}"
            );
        }
        // `gamma` is the ramp exponent: `SetGamma` applies `1 - v` before it arrives here. The
        // consumer clamps what the reference accepts unclamped (`SetGamma(5)` writes -4 there).
        for (wrote, want) in [("1.000000", 1.0), ("0.500000", 0.5), ("1.500000", 1.5)] {
            apply(&mut app, "gamma", wrote);
            assert_eq!(
                res::<crate::ui_gamma::DisplayGamma>(&app).0,
                want,
                "gamma {wrote}"
            );
        }
        apply(&mut app, "gamma", "-4.000000");
        assert_eq!(
            res::<crate::ui_gamma::DisplayGamma>(&app).0,
            *crate::ui_gamma::GAMMA_RANGE.start(),
            "a negative exponent clamps at the consumer, where it cannot blank the screen"
        );
        apply(&mut app, "gamma", "99");
        assert_eq!(
            res::<crate::ui_gamma::DisplayGamma>(&app).0,
            *crate::ui_gamma::GAMMA_RANGE.end()
        );
        assert_eq!(res::<ClutterConfig>(&app).frill_density(), 32.0);
        // Both spellings reach the field from a state neither holds; `$WOW_CLUTTER_DENSITY` takes
        // both for the session. Two values, since writing the sibling's mirrored value is a no-op.
        for (key, value) in [
            (benilla_ui::script::CVAR_WORLD_DETAIL, "2"),
            (benilla_ui::script::CVAR_FRILL_DENSITY, "16"),
        ] {
            app.world_mut().resource_mut::<ClutterConfig>().density = 0.5;
            assert_eq!(apply(&mut app, key, value), SetOutcome::Changed);
            assert_ne!(
                res::<ClutterConfig>(&app).density,
                0.5,
                "{key}: reached no knob"
            );
        }
        // `benilla-ui`'s `SetWorldDetail`/`GetWorldDetail` name these CVars by const and cannot see
        // this table, so both must stay registered and the stops must stay `frillDensity`'s unit
        // times n.
        assert!(REGISTERED
            .iter()
            .any(|r| r.name == benilla_ui::script::CVAR_WORLD_DETAIL));
        assert!(REGISTERED
            .iter()
            .any(|r| r.name == benilla_ui::script::CVAR_FRILL_DENSITY));
        // The same for `GetGamma`/`SetGamma`, which read and write `1 - gamma`.
        assert!(REGISTERED
            .iter()
            .any(|r| r.name == benilla_ui::script::CVAR_GAMMA));
        // `RestoreVideoDefaults`' row list lives in `benilla-ui` too; an unregistered row would be
        // skipped silently.
        for key in benilla_ui::script::VIDEO_DEFAULT_CVARS {
            assert!(
                REGISTERED.iter().any(|r| r.name.eq_ignore_ascii_case(key)),
                "{key}: RestoreVideoDefaults would restore it, and nothing registers it"
            );
        }
        // The video pairs and `SetWorldDetail`'s `smallCull` write by name; a missing row would
        // warn and store nothing.
        for key in benilla_ui::script::VIDEO_PAIR_CVARS
            .iter()
            .chain([&benilla_ui::script::CVAR_SMALL_CULL])
        {
            assert!(
                REGISTERED.iter().any(|r| r.name.eq_ignore_ascii_case(key)),
                "{key}: a video verb writes it, and nothing registers it"
            );
        }
        for (n, frill) in benilla_ui::script::WORLD_DETAIL_STOPS.iter().enumerate() {
            assert_eq!(
                *frill,
                benilla_formats::FRILL_DENSITY * (n as u32 + 1),
                "stop {n}: the reference's own 0x804518 entry must be this knob's unit times the stop"
            );
            apply(&mut app, "WorldDetail", &n.to_string());
            let by_stop = res::<ClutterConfig>(&app).density;
            // Off the stop first (the mirror already wrote this spelling), then the stop's cells.
            apply(&mut app, "frillDensity", "1");
            app.world_mut().resource_mut::<ClutterConfig>().density = 0.5;
            apply(&mut app, "frillDensity", &frill.to_string());
            assert_eq!(
                res::<ClutterConfig>(&app).density,
                by_stop,
                "stop {n}: the two spellings disagree"
            );
            assert_eq!(res::<ClutterConfig>(&app).frill_density(), *frill as f32);
        }
        apply(&mut app, "WorldDetail", "1");
        assert_eq!(res::<ClutterConfig>(&app).density, 2.0);
        // A stop set as a CVar keeps `SmallCull` in step, as `SetWorldDetail` writes it.
        apply(&mut app, "WorldDetail", "0");
        assert_eq!(
            res::<Cvars>(&app).get(benilla_ui::script::CVAR_SMALL_CULL),
            Some("0.070000")
        );
        apply(&mut app, "WorldDetail", "1");
        assert_eq!(
            res::<Cvars>(&app).get(benilla_ui::script::CVAR_SMALL_CULL),
            Some("0.040000")
        );
        // Spell Detail: `SStrToInt`, clamp, then the shared emission scalar (`0x689510`).
        for (level, scale) in [
            ("0", 0.33),
            ("1", 0.66),
            ("2", 1.0),
            ("-3", 0.33),
            ("7", 1.0),
        ] {
            apply(&mut app, "spellEffectLevel", level);
            assert_eq!(
                res::<benilla_world::particles::ParticleTuning>(&app).density(),
                scale,
                "spellEffectLevel {level}"
            );
            assert_eq!(
                res::<crate::video::SpellEffectLevel>(&app).0,
                level.parse::<i32>().unwrap(),
                "the record's integer, unclamped, for the shard emitter"
            );
        }
        apply(&mut app, "minimapZoom", "5");
        assert_eq!(res::<MinimapZoom>(&app).outdoor, 5);
        assert_eq!(
            res::<MinimapZoom>(&app).inside,
            3,
            "the two indices are independent"
        );
        apply(&mut app, "minimapinsidezoom", "9");
        assert_eq!(res::<MinimapZoom>(&app).inside, MINIMAP_ZOOM_LEVELS - 1);
        apply(&mut app, "minimapZoom", "-2");
        assert_eq!(res::<MinimapZoom>(&app).outdoor, 0);
        assert_eq!(apply(&mut app, "uiScale", "banana"), SetOutcome::Refused);
        assert_eq!(res::<UiScaleCvar>(&app).0, 0.9);
        assert_eq!(apply(&mut app, "bogus", "1"), SetOutcome::Unknown);
    }

    /// `/console spellEffectLevel 0` prints the handler's own line (`0x689537`-`0x689554`, format
    /// `0x869f94`) with the level it clamped to; a Lua write logs it instead.
    #[test]
    fn the_spell_effect_level_handler_echoes_to_the_console() {
        let mut app = cvar_app();
        app.init_resource::<crate::console::ConsoleEcho>();
        assert_eq!(
            crate::console::execute(app.world_mut(), "spellEffectLevel 0"),
            ["Spell effect level set to 0."]
        );
        assert_eq!(
            crate::console::execute(app.world_mut(), "spellEffectLevel 9"),
            ["Spell effect level set to 2."]
        );
        assert_eq!(res::<Cvars>(&app).get("spellEffectLevel"), Some("9"));
        assert_eq!(
            crate::console::execute(app.world_mut(), "farclip 400"),
            Vec::<String>::new(),
            "a callback that prints nothing adds nothing"
        );
    }

    #[test]
    fn compose_writes_the_diff_and_preserves_what_it_does_not_own() {
        let mut cvars = Cvars::default();
        cvars.own_for_session("uiScale", Some("1.2")); // env-overridden this session
        cvars.load_file(
            [
                ("FutureKnob".to_string(), "3".to_string()), // a newer build's key: preserved
                ("uiScale".to_string(), "0.8".to_string()),  // the file's own, kept as found
                ("farclip".to_string(), "400".to_string()),  // will return to default
            ]
            .into(),
        );
        cvars.set("MusicVolume", "0.7"); // moved: written
        cvars.set("farclip", "350"); // back to default: removed
        let out = cvars.compose();
        assert_eq!(out.get("MusicVolume").map(String::as_str), Some("0.7"));
        assert!(!out.contains_key("MasterVolume"), "at default: absent");
        assert_eq!(out.get("uiScale").map(String::as_str), Some("0.8"));
        assert!(!out.contains_key("farclip"));
        assert_eq!(out.get("FutureKnob").map(String::as_str), Some("3"));
    }

    #[test]
    fn a_change_saves_after_one_quiet_second_with_no_exit() {
        use crate::local_state::test_env::{EnvGuard, ENV_LOCK};
        let _l = ENV_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let tmp = std::env::temp_dir().join(format!("benilla-cvar-quiet-{}", std::process::id()));
        std::fs::remove_dir_all(&tmp).ok();
        let _c = EnvGuard::unset("WOW_CAPTURE");
        let _u = EnvGuard::unset("WOW_UI_SCALE");
        let _f = EnvGuard::unset("WOW_FARCLIP");
        let _d = EnvGuard::unset("WOW_CLUTTER_DENSITY");
        let _h = EnvGuard::set("BENILLA_HOME", tmp.to_str().unwrap());

        let mut app = cvar_app();
        app.update();
        app.world_mut()
            .non_send_resource_mut::<UiScript>()
            .run(r#"SetCVar("MusicVolume", 0.75)"#)
            .unwrap();
        app.update();
        assert!(
            !tmp.join("config.toml").exists(),
            "inside the quiet second nothing is written"
        );

        // The change ages past the quiet second; no `AppExit` anywhere.
        {
            let mut cvars = app.world_mut().resource_mut::<Cvars>();
            assert!(cvars.dirty);
            cvars.last_change = Instant::now().checked_sub(SAVE_QUIET * 2);
        }
        app.update();
        let text = std::fs::read_to_string(tmp.join("config.toml")).unwrap();
        assert!(text.contains("MusicVolume = \"0.75\""), "{text}");
        assert!(!app.world().resource::<Cvars>().dirty);
        assert!(app.should_exit().is_none());
        std::fs::remove_dir_all(&tmp).ok();
    }

    #[test]
    fn a_lua_setcvar_lands_in_config_toml_end_to_end() {
        use crate::local_state::test_env::{EnvGuard, ENV_LOCK};
        let _l = ENV_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let tmp = std::env::temp_dir().join(format!("benilla-cvar-e2e-{}", std::process::id()));
        std::fs::remove_dir_all(&tmp).ok();
        let _c = EnvGuard::unset("WOW_CAPTURE");
        let _u = EnvGuard::unset("WOW_UI_SCALE");
        let _f = EnvGuard::unset("WOW_FARCLIP");
        let _d = EnvGuard::unset("WOW_CLUTTER_DENSITY");
        let _h = EnvGuard::set("BENILLA_HOME", tmp.to_str().unwrap());
        crate::local_state::write_atomic(
            &tmp.join("config.toml"),
            "[cvars]\nMusicVolume = \"0.1\"\n",
        )
        .unwrap();

        let mut app = cvar_app();

        // Startup: the file's MusicVolume reaches the knob; Update: the VM table seeds from it.
        app.update();
        assert_eq!(app.world().resource::<SoundConfig>().music, 0.1);
        assert_eq!(
            app.world_mut()
                .non_send_resource_mut::<UiScript>()
                .cvar("MusicVolume")
                .as_deref(),
            Some("0.1")
        );

        // The Lua write (what a settings slider will do) reaches the knob on the next frame…
        app.world_mut()
            .non_send_resource_mut::<UiScript>()
            .run(r#"SetCVar("MusicVolume", 0.75)"#)
            .unwrap();
        app.update();
        assert_eq!(app.world().resource::<SoundConfig>().music, 0.75);

        // …and the exit flush writes the diff: the moved value, nothing at its default.
        app.world_mut().write_message(AppExit::Success);
        app.update();
        let text = std::fs::read_to_string(tmp.join("config.toml")).unwrap();
        assert!(text.contains("MusicVolume = \"0.75\""), "{text}");
        assert!(!text.contains("MasterVolume"), "defaults stay out:\n{text}");
        let back: LocalConfig = toml::from_str(&text).unwrap();
        // MONKEY (presets): the file named no `graphicsQuality`, so the first boot seeded the
        // High rungs over the governed rows; those are the only other entries, and the label
        // itself, at its default, stays out.
        let player: Vec<&String> = back
            .cvars
            .keys()
            .filter(|k| {
                !GRAPHICS_PRESETS
                    .iter()
                    .any(|(g, _)| g.eq_ignore_ascii_case(k))
            })
            .collect();
        assert_eq!(player.len(), 1, "a diff, not a dump: {text}");
        assert!(!text.contains("graphicsQuality"), "{text}");
        std::fs::remove_dir_all(&tmp).ok();
    }

    /// A client with a real CVar host: every knob resource an observer writes, every observer,
    /// [`CvarPlugin`] and a VM for the mirror.
    fn cvar_app() -> App {
        let mut app = App::new();
        app.add_plugins(bevy::MinimalPlugins)
            .insert_resource(SoundConfig::default())
            .insert_resource(UiScaleCvar(DEFAULT_UI_SCALE))
            .insert_resource(ViewDistance {
                farclip: 350.0,
                nearclip: benilla_world::view::NEARCLIP_DEFAULT,
            })
            .insert_resource(MsaaSetting { samples: 1 })
            // Literal: `TexFilterSetting::default()` reads `$WOW_TRILINEAR`/`$WOW_ANISO`.
            .insert_resource(benilla_assets::TexFilterSetting {
                trilinear: true,
                aniso: 1,
            })
            // A real-shaped device menu: `GetCurrentMultisampleFormat` needs rows to find.
            .insert_resource(benilla_world::view::MsaaFormats {
                formats: vec![(32, 32, 1), (32, 32, 2), (32, 32, 4)],
            })
            .init_resource::<LookConfig>()
            .init_resource::<crate::player::camera_dynamics::CameraOptions>()
            .init_resource::<crate::ui_chat::combat::CombatLogRanges>()
            .init_resource::<crate::combat_text::DamageTextGates>()
            .init_resource::<crate::ui_chat::combat::LogPeriodicSpells>()
            .init_resource::<benilla_world::weather::WeatherState>()
            .init_resource::<benilla_world::particles::ParticleTuning>()
            .init_resource::<crate::video::SpellEffectLevel>()
            .init_resource::<crate::ui_gamma::DisplayGamma>()
            .init_resource::<ClickConfig>()
            .init_resource::<crate::target::AssistAttack>()
            .init_resource::<LootConfig>()
            .init_resource::<NameConfig>()
            .init_resource::<ClutterConfig>()
            .init_resource::<MinimapZoom>()
            .init_resource::<BubbleConfig>()
            .init_resource::<ZoomLimit>()
            .init_resource::<FollowConfig>()
            .init_resource::<VideoConfig>()
            .init_resource::<benilla_world::ffx_glow::FfxSwitches>()
            // Literal, not Default: RenderScale::default() reads $WOW_RENDER_SCALE.
            .insert_resource(RenderScale(1.0))
            // Literal: `Realmlist::default()` reads `$WOW_HOST`.
            .insert_resource(crate::realmlist::Realmlist::unpinned(
                crate::realmlist::DEFAULT_REALMLIST,
            ))
            .init_resource::<PaneRate>()
            .init_resource::<crate::ui_guild::GuildMemberNotify>()
            .init_resource::<crate::ui_trade::BlockTrades>()
            .init_resource::<crate::spell::AutoSelfCast>()
            .init_resource::<crate::perf::FpsJournalSetting>()
            .init_resource::<crate::text_filter::TextFilterSwitches>()
            .init_resource::<crate::game_tip::GameTipSetting>()
            .add_plugins(CvarPlugin);
        for observer in ALL_OBSERVERS {
            observer(&mut app);
        }
        app.insert_non_send_resource(UiScript::new().unwrap());
        app
    }

    /// Every CVar observer in the crate; a new one is a line here and an `add_observer` in its
    /// plugin.
    const ALL_OBSERVERS: &[fn(&mut App)] = &[
        |app| {
            app.add_observer(crate::sound::on_cvar);
        },
        |app| {
            app.add_observer(crate::ui_script::on_cvar);
        },
        |app| {
            app.add_observer(crate::video::on_cvar);
        },
        // MONKEY (p0 graphics programme)
        |app| {
            app.init_resource::<benilla_world::ffx_glow::SkyDither>();
            // MONKEY (fog)
            app.init_resource::<benilla_world::lighting::FogModelSetting>();
            // MONKEY (wind)
            app.init_resource::<benilla_world::wind::FoliageWind>();
            app.init_resource::<benilla_world::wind::FoliageWindStrength>(); // MONKEY (fix-wind)
            app.init_resource::<benilla_world::weather::RainSurfaces>(); // MONKEY (wet)
            app.add_observer(crate::monkey_gfx::on_cvar);
        },
        // MONKEY (reviewfix-a): the zone-skybox bridge, so every governed row has its observer.
        |app| {
            app.insert_resource(crate::zone_skybox::ZoneSkyboxOverride(None));
            app.init_resource::<benilla_world::skybox::ZoneSkyboxes>();
            app.add_observer(crate::zone_skybox::on_cvar);
        },
        |app| {
            app.add_observer(crate::player::camera::on_cvar);
        },
        |app| {
            app.add_observer(crate::player::camera_dynamics::on_cvar);
        },
        |app| {
            app.add_observer(crate::target::on_cvar);
        },
        |app| {
            app.add_observer(crate::spell::cast_target::on_cvar);
        },
        |app| {
            app.add_observer(crate::combat_text::on_cvar);
        },
        |app| {
            app.add_observer(crate::ui_chat::combat::on_cvar);
        },
        |app| {
            app.add_observer(crate::ui_loot::on_cvar);
        },
        |app| {
            app.add_observer(crate::nameplates::on_cvar);
        },
        |app| {
            app.add_observer(crate::game_tip::on_cvar);
        },
        |app| {
            app.add_observer(crate::text_filter::on_cvar);
        },
        |app| {
            app.add_observer(crate::chat_bubble::on_cvar);
        },
        |app| {
            app.add_observer(crate::ui_guild::on_cvar);
        },
        |app| {
            app.add_observer(crate::ui_trade::on_cvar);
        },
        |app| {
            app.add_observer(crate::minimap::on_cvar);
        },
        |app| {
            app.add_observer(crate::portrait::on_cvar);
        },
        |app| {
            app.add_observer(crate::perf::on_cvar);
        },
        |app| {
            app.add_observer(crate::realmlist::on_cvar);
        },
        |app| {
            app.add_observer(crate::ui_gamma::on_cvar);
        },
        |app| {
            app.add_observer(crate::world_backdrop::on_cvar);
        },
    ];

    /// A host write, its latch committed at once, and its observers run before this returns.
    fn apply(app: &mut App, name: &str, value: &str) -> SetOutcome {
        let world = app.world_mut();
        let (outcome, events) = {
            let mut cvars = world.resource_mut::<Cvars>();
            let outcome = cvars.set(name, value);
            cvars.commit_latched();
            (outcome, cvars.take_events())
        };
        for event in events {
            world.trigger(event);
        }
        outcome
    }

    fn res<T: Resource>(app: &App) -> &T {
        app.world().resource::<T>()
    }

    /// The last character entered survives a quit: two launches over one `benilla-config/` with the
    /// real [`CvarPlugin`] and [`crate::char_select`] systems, so the screen's host write must
    /// reach the file through the registry's dirty/compose path.
    #[test]
    fn entering_the_world_survives_the_quit_and_comes_back_selected() {
        use crate::char_select::{ClientState, Roster};
        use crate::local_state::test_env::{EnvGuard, ENV_LOCK};
        let _l = ENV_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let tmp = std::env::temp_dir().join(format!("benilla-lastchar-{}", std::process::id()));
        std::fs::remove_dir_all(&tmp).ok();
        let _c = EnvGuard::unset("WOW_CAPTURE");
        let _u = EnvGuard::unset("WOW_UI_SCALE");
        let _f = EnvGuard::unset("WOW_FARCLIP");
        let _d = EnvGuard::unset("WOW_CLUTTER_DENSITY");
        let _w = EnvGuard::unset("WOW_CHAR");
        let _s = EnvGuard::unset("WOW_CHARSELECT_PICK");
        let _h = EnvGuard::set("BENILLA_HOME", tmp.to_str().unwrap());
        let roster = || {
            (1..=4)
                .map(|g| crate::char_select::test_character(g, &format!("Char{g}")))
                .collect::<Vec<_>>()
        };

        // ── Launch 1: the roster lands, and the player enters the world as the third row. ────
        let (tx, _rx) = crossbeam_channel::unbounded();
        let mut app = cvar_app();
        app.add_plugins(bevy::state::app::StatesPlugin);
        crate::char_select::add_test_systems(&mut app, tx);
        app.update(); // Startup loads the (absent) file; the first Update seeds the VM table
        app.world_mut().write_message(crate::net::CharListMessage {
            characters: roster(),
            realm: None,
        });
        app.update();
        assert_eq!(
            app.world().resource::<Roster>().selected(),
            Some(0),
            "nothing remembered yet, so the first row — the behaviour that was already right",
        );
        app.world_mut().resource_mut::<Roster>().pending_pick = Some(3); // guid 3 = row 2
        app.update();
        app.world_mut().write_message(AppExit::Success);
        app.update();

        let text = std::fs::read_to_string(tmp.join("config.toml")).unwrap();
        assert!(
            text.contains("lastCharacterIndex = \"2\""),
            "entering the world must reach the file, 0-based like Config.wtf:\n{text}"
        );

        // ── Launch 2: a fresh client over the same folder, and the roster arrives. ───────────
        let (tx, _rx2) = crossbeam_channel::unbounded();
        let mut app = cvar_app();
        app.add_plugins(bevy::state::app::StatesPlugin);
        crate::char_select::add_test_systems(&mut app, tx);
        app.update();
        app.world_mut().write_message(crate::net::CharListMessage {
            characters: roster(),
            realm: None,
        });
        app.update();

        assert_eq!(
            app.world().resource::<Roster>().selected(),
            Some(2),
            "the second launch must stand the SAME character on the stage — the whole report",
        );
        assert_eq!(
            *app.world().resource::<State<ClientState>>().get(),
            ClientState::CharSelect,
        );
        std::fs::remove_dir_all(&tmp).ok();
    }

    #[test]
    fn a_minimap_setzoom_reaches_the_knob_and_the_file() {
        use crate::local_state::test_env::{EnvGuard, ENV_LOCK};
        let _l = ENV_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let tmp = std::env::temp_dir().join(format!("benilla-mmzoom-{}", std::process::id()));
        std::fs::remove_dir_all(&tmp).ok();
        let _c = EnvGuard::unset("WOW_CAPTURE");
        let _u = EnvGuard::unset("WOW_UI_SCALE");
        let _f = EnvGuard::unset("WOW_FARCLIP");
        let _d = EnvGuard::unset("WOW_CLUTTER_DENSITY");
        let _h = EnvGuard::set("BENILLA_HOME", tmp.to_str().unwrap());
        // The previous session left the outdoor map zoomed right in.
        crate::local_state::write_atomic(
            &tmp.join("config.toml"),
            "[cvars]\nminimapZoom = \"5\"\n",
        )
        .unwrap();

        let mut app = cvar_app();
        app.update();

        assert_eq!(app.world().resource::<MinimapZoom>().outdoor, 5);
        assert_eq!(app.world().resource::<MinimapZoom>().inside, 3);
        let seed = {
            let z = app.world().resource::<MinimapZoom>();
            (z.outdoor, z.inside)
        };
        {
            let mut script = app.world_mut().non_send_resource_mut::<UiScript>();
            assert_eq!(script.cvar("minimapZoom").as_deref(), Some("5"));
            // The widget is born first, then the persisted level is pushed into it: seeding a
            // widget that does not exist is a no-op.
            script.run(r#"m = CreateFrame("Minimap", "Mini")"#).unwrap();
            script.set_minimap_zoom(seed.0, seed.1);
            assert_eq!(script.eval::<u8>("return m:GetZoom()").unwrap(), 5);
            script.run("m:SetZoom(m:GetZoom() - 2)").unwrap();
        }
        app.update();
        assert_eq!(app.world().resource::<MinimapZoom>().outdoor, 3);

        app.world_mut().write_message(AppExit::Success);
        app.update();
        let text = std::fs::read_to_string(tmp.join("config.toml")).unwrap();
        assert!(
            !text.contains("minimapZoom"),
            "back at the registered default 3, so it leaves the diff entirely:\n{text}"
        );

        app.world_mut()
            .non_send_resource_mut::<UiScript>()
            .run("m:SetZoom(1)")
            .unwrap();
        app.update();
        app.world_mut().write_message(AppExit::Success);
        app.update();
        let text = std::fs::read_to_string(tmp.join("config.toml")).unwrap();
        assert!(text.contains("minimapZoom = \"1\""), "{text}");
        assert_eq!(app.world().resource::<MinimapZoom>().outdoor, 1);
        std::fs::remove_dir_all(&tmp).ok();
    }

    #[test]
    fn the_toml_round_trips() {
        let cfg = LocalConfig {
            cvars: [("MusicVolume".to_string(), "0.7".to_string())].into(),
        };
        let text = format!("{HEADER}{}", toml::to_string(&cfg).unwrap());
        let back: LocalConfig = toml::from_str(&text).unwrap();
        assert_eq!(back.cvars, cfg.cvars);
        let hand = "# my note\n[cvars]\nFarclip = \"500\"\n";
        let parsed: LocalConfig = toml::from_str(hand).unwrap();
        assert_eq!(parsed.cvars.get("Farclip").map(String::as_str), Some("500"));
    }

    /// The string-valued rows, a closed list, each default checked on its own terms: `realmName`
    /// (the reference's registered `""`, `0x882748`) and `gxApi` are empty until the session writes
    /// them; `gxResolution` and `realmList` pass their observers' parsers
    /// ([`crate::video::parse_resolution`], [`crate::realmlist::normalize`]).
    #[test]
    fn the_string_valued_cvars_are_the_realm_and_the_windowed_size() {
        let mut strings: Vec<&str> = REGISTERED
            .iter()
            .filter(|r| r.default.parse::<f32>().is_err())
            .map(|r| r.name)
            .collect();
        strings.sort_unstable(); // the list is the claim, not where the rows sit in the table
        assert_eq!(
            strings,
            vec![
                "graphicsQuality",
                "gxApi",
                "gxResolution",
                "lightingQuality",
                "realmList",
                "realmName"
            ]
        );
        let default_of = |name: &str| {
            REGISTERED
                .iter()
                .find(|r| r.name == name)
                .map(|r| r.default)
                .expect("registered")
        };
        assert_eq!(default_of("realmName"), "");
        assert_eq!(default_of("gxApi"), "");
        assert_eq!(
            crate::video::parse_resolution(default_of("gxResolution")),
            Some(crate::video::DEFAULT_WINDOWED)
        );
        // A default `realmlist::normalize` rejects would ship a client that cannot dial.
        assert_eq!(
            crate::realmlist::normalize(default_of(crate::realmlist::CVAR_REALMLIST)).as_deref(),
            Some(crate::realmlist::DEFAULT_REALMLIST),
        );
    }

    /// A registry loaded from `file`, with no disk.
    fn registry(file: &[(&str, &str)]) -> Cvars {
        let mut cvars = Cvars::default();
        cvars.load_file(
            file.iter()
                .map(|(k, v)| (k.to_string(), v.to_string()))
                .collect(),
        );
        cvars
    }

    /// A latched row's write is staged: the applied value stands and nothing fires or dirties (the
    /// reference's `Set 0x63df50` stores `latchedValue` and skips `InternalSet`) until the commit
    /// (`0x63e060`) applies, fires, persists and mirrors it.
    #[test]
    fn a_latched_row_stages_the_write_until_the_boundary_commits_it() {
        let mut cvars = Cvars::default();
        assert!(
            cvars.row("gxVSync").unwrap().latched,
            "the reference's flags=3 row"
        );
        assert_eq!(cvars.set("gxVSync", "0"), SetOutcome::Staged);
        assert_eq!(cvars.get("gxVSync"), Some("1"), "applied value stands");
        assert_eq!(cvars.row("gxVSync").unwrap().pending.as_deref(), Some("0"));
        assert!(!cvars.has_events(), "nothing fires before the boundary");
        assert!(!cvars.dirty, "nothing to save before the boundary");
        assert_eq!(
            cvars.set("gxVSync", "0"),
            SetOutcome::Unchanged,
            "same stage"
        );
        assert!(
            !cvars.compose().contains_key("gxVSync"),
            "a stage that is never committed never reaches the file"
        );
        assert_eq!(cvars.commit_latched(), 1);
        assert_eq!(cvars.get("gxVSync"), Some("0"));
        assert_eq!(cvars.row("gxVSync").unwrap().pending, None);
        assert_eq!(
            cvars.take_events(),
            vec![CvarChanged {
                name: "gxVSync".into(),
                old: "1".into(),
                new: "0".into()
            }]
        );
        assert!(cvars.dirty);
        assert_eq!(
            cvars.compose().get("gxVSync").map(String::as_str),
            Some("0")
        );
        assert_eq!(
            cvars.take_outbox(),
            vec![("gxVSync".to_string(), "0".to_string())]
        );
        cvars.set("gxVSync", "1");
        assert_eq!(cvars.set("gxVSync", "0"), SetOutcome::Unchanged);
        assert_eq!(cvars.row("gxVSync").unwrap().pending, None);
        assert_eq!(cvars.commit_latched(), 0);
        // A latched row outside the gx set is not the video restart's to commit.
        assert_eq!(cvars.set("SoundBufferSize", "200"), SetOutcome::Staged);
        assert_eq!(cvars.commit_latched(), 0);
        assert_eq!(cvars.get("SoundBufferSize"), Some("100"));
        assert_eq!(
            cvars.row("SoundBufferSize").unwrap().pending.as_deref(),
            Some("200")
        );
        assert!(!cvars.compose().contains_key("SoundBufferSize"));
        assert_eq!(cvars.set("MusicVolume", "0.7"), SetOutcome::Changed);
        assert_eq!(cvars.get("MusicVolume"), Some("0.7"));
        assert_eq!(cvars.take_events().len(), 1);
    }

    #[test]
    fn a_numeric_row_refuses_what_does_not_parse_and_corrects_the_mirror() {
        let mut cvars = Cvars::default();
        assert_eq!(cvars.set_from_vm("uiScale", "banana"), SetOutcome::Refused);
        assert_eq!(cvars.get("uiScale"), Some("0.9"));
        assert!(!cvars.has_events());
        assert!(!cvars.dirty);
        assert_eq!(
            cvars.take_outbox(),
            vec![("uiScale".to_string(), "0.9".to_string())],
            "the VM stored 'banana' synchronously; the host writes the truth back"
        );
        assert_eq!(cvars.set("uiScale", "banana"), SetOutcome::Refused);
        assert!(
            cvars.take_outbox().is_empty(),
            "a host write has no mirror to correct"
        );
        assert_eq!(
            cvars.set("realmList", "not an address"),
            SetOutcome::Changed
        );
        assert_eq!(cvars.set("nosuchrow", "1"), SetOutcome::Unknown);
    }

    #[test]
    fn an_addon_row_persists_like_the_clients_own() {
        let mut cvars = registry(&[("myAddonKnob", "3")]);
        assert_eq!(
            cvars.orphans(),
            vec![("myAddonKnob".to_string(), "3".to_string())],
            "unclaimed until the addon declares it — the VM's saved base"
        );
        cvars.learn_addon_row("myAddonKnob", "1");
        assert_eq!(
            cvars.get("myAddonKnob"),
            Some("3"),
            "starts at the saved value"
        );
        assert_eq!(cvars.default_of("myAddonKnob"), Some("1"));
        assert!(cvars.orphans().is_empty());
        cvars.learn_addon_row("myAddonKnob", "9");
        assert_eq!(
            cvars.default_of("myAddonKnob"),
            Some("1"),
            "a re-declaration is a no-op"
        );
        assert!(!cvars.dirty);
        assert_eq!(cvars.set_from_vm("myAddonKnob", "5"), SetOutcome::Changed);
        assert!(cvars.dirty, "an addon-only change is a change");
        assert_eq!(
            cvars.compose().get("myAddonKnob").map(String::as_str),
            Some("5")
        );
        cvars.set_from_vm("myAddonKnob", "1");
        assert!(
            !cvars.compose().contains_key("myAddonKnob"),
            "back at the addon's default, it leaves the diff"
        );
        let cvars = registry(&[("FutureKnob", "3")]);
        assert_eq!(
            cvars.compose().get("FutureKnob").map(String::as_str),
            Some("3")
        );
    }

    #[test]
    fn a_session_owned_row_answers_the_env_and_never_reaches_the_file() {
        let mut cvars = Cvars::default();
        cvars.own_for_session("uiScale", Some("1.2"));
        cvars.load_file(BTreeMap::from([("uiScale".to_string(), "0.8".to_string())]));
        assert_eq!(
            cvars.get("uiScale"),
            Some("1.2"),
            "the env's, not the file's"
        );
        assert!(
            !cvars.has_events(),
            "the knob already read the env — nothing to apply"
        );
        assert_eq!(cvars.set("uiScale", "1.4"), SetOutcome::Changed);
        assert_eq!(
            cvars.compose().get("uiScale").map(String::as_str),
            Some("0.8"),
            "the file keeps what it said"
        );
        // A lever with no resource still marks the row, so the file cannot apply over the env.
        cvars.own_for_session("farclip", None);
        cvars.load_file(BTreeMap::from([("farclip".to_string(), "500".to_string())]));
        assert_eq!(cvars.get("farclip"), Some("350"));
        assert!(cvars.is_session_owned("FARCLIP"));
    }

    /// The camera reads `gxMultisample` once at spawn, so the file must be applied inside
    /// [`CvarLoad`].
    #[test]
    fn the_loaded_file_reaches_a_knob_before_anything_ordered_after_cvar_load() {
        use crate::local_state::test_env::{EnvGuard, ENV_LOCK};
        let _l = ENV_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let tmp = std::env::temp_dir().join(format!("benilla-cvar-load-{}", std::process::id()));
        std::fs::remove_dir_all(&tmp).ok();
        let _c = EnvGuard::unset("WOW_CAPTURE");
        let _u = EnvGuard::unset("WOW_UI_SCALE");
        let _f = EnvGuard::unset("WOW_FARCLIP");
        let _d = EnvGuard::unset("WOW_CLUTTER_DENSITY");
        let _m = EnvGuard::unset("WOW_MSAA");
        let _h = EnvGuard::set("BENILLA_HOME", tmp.to_str().unwrap());
        crate::local_state::write_atomic(
            &tmp.join("config.toml"),
            "[cvars]\nMusicVolume = \"0.1\"\ngxMultisample = \"4\"\n",
        )
        .unwrap();
        #[derive(Resource, Default)]
        struct SeenAtStartup(Option<(f32, u32)>);
        fn after_load(
            sound: Res<SoundConfig>,
            msaa: Res<MsaaSetting>,
            mut seen: ResMut<SeenAtStartup>,
        ) {
            seen.0 = Some((sound.music, msaa.samples));
        }
        let mut app = cvar_app();
        app.init_resource::<SeenAtStartup>()
            .add_systems(Startup, after_load.after(CvarLoad));
        app.update();
        assert_eq!(
            app.world().resource::<SeenAtStartup>().0,
            Some((0.1, 4)),
            "both the plain row and the latched one are applied by the time CvarLoad is over"
        );
        assert_eq!(
            app.world().resource::<Cvars>().get("gxMultisample"),
            Some("4"),
            "a file value is applied, not staged: the reference's LoadFile runs before Register"
        );
        std::fs::remove_dir_all(&tmp).ok();
    }

    /// MONKEY (integration): deterministic graphics A/B captures may read one explicit fixture,
    /// but the local-state law still exposes no writable capture path.
    #[test]
    fn a_capture_can_read_an_explicit_cvar_fixture_without_enabling_writes() {
        use crate::local_state::test_env::{EnvGuard, ENV_LOCK};
        let _l = ENV_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let tmp = std::env::temp_dir().join(format!(
            "benilla-capture-cvar-fixture-{}",
            std::process::id()
        ));
        std::fs::remove_dir_all(&tmp).ok();
        crate::local_state::write_atomic(
            &tmp.join("config.toml"),
            "[cvars]\nbloom = \"0\"\nsunShafts = \"0\"\ncolorGrading = \"0\"\n",
        )
        .unwrap();
        let _h = EnvGuard::set("BENILLA_HOME", tmp.to_str().unwrap());
        // MONKEY (reviewfix-a): the fixture is named by its own variable, not BENILLA_HOME.
        let _x = EnvGuard::set(
            "WOW_CAPTURE_CVARS",
            tmp.join("config.toml").to_str().unwrap(),
        );
        let _c = EnvGuard::set("WOW_CAPTURE", "post-lava-searing");

        assert_eq!(boot_cvar("bloom").as_deref(), Some("0"));
        assert_eq!(boot_cvar("sunShafts").as_deref(), Some("0"));
        assert_eq!(boot_cvar("colorGrading").as_deref(), Some("0"));
        assert_eq!(
            crate::local_state::config_path(),
            None,
            "capture fixtures are read-only; persistence remains hermetic"
        );
        std::fs::remove_dir_all(&tmp).ok();
    }

    /// Every registered row has a reader: a string literal naming it in this crate's or
    /// `benilla-ui`'s code, or a `LUA_ONLY` entry naming its stock reader. Test files do not count.
    #[test]
    fn every_registered_row_has_a_reader_in_the_source() {
        /// Rows whose only reader is the stock interface.
        const LUA_ONLY: &[(&str, &str)] = &[
            (
                "statusBarText",
                "TextStatusBar.lua reads it on each CVAR_UPDATE",
            ),
            (
                "UberTooltips",
                "GameTooltip's binding-line gate, stock and pfUI",
            ),
            ("gxApi", "pfUI's system tooltip names the backend"),
            (
                "graphicsQuality",
                "MONKEY (presets): the Advanced Graphics page's Graphics Preset row; the host \
                 writes and derives it here in cvars.rs",
            ),
            (
                "useUiScale",
                "UIOptionsFrame.lua and OptionsFrame.lua branch on it to gate the uiScale slider",
            ),
            (
                "DesktopGamma",
                "OptionsFrame.lua's Use Desktop Gamma box, recorded at open and put back on close",
            ),
            (
                "pixelShaders",
                "OptionsFrame.lua's Enable All Shaders box, which its Okay copies into `ffx`",
            ),
            ("specular", "OptionsFrame.lua's Terrain Highlights box"),
            (
                "M2UseShaders",
                "OptionsFrame.lua's Vertex Animation Shaders box",
            ),
            ("M2UsePixelShaders", "OptionsFrame.lua's Phong Shading box"),
            ("lod", "OptionsFrame.lua's World LOD box"),
            (
                "movieSubtitle",
                "OptionsFrame.lua's Cinematic Subtitles box",
            ),
            (
                "useWeatherShaders",
                "OptionsFrame.lua's Weather Shaders box",
            ),
            ("gxTripleBuffer", "OptionsFrame.lua's Triple Buffering box"),
            ("gxCursor", "OptionsFrame.lua's Hardware Cursor box"),
            ("gxFixLag", "OptionsFrame.lua's Fix Input Lag box"),
            ("gxRefresh", "OptionsFrame.lua's refresh-rate dropdown"),
        ];
        let app_src = crate::test_support::src_dir();
        let ui_src = app_src
            .parent()
            .and_then(std::path::Path::parent)
            .expect("crates/")
            .join("benilla-ui")
            .join("src");
        let mut code = String::new();
        for file in crate::test_support::rust_files(&app_src)
            .into_iter()
            .chain(crate::test_support::rust_files(&ui_src))
        {
            let rel = file.to_string_lossy().replace('\\', "/");
            if rel.contains("benilla-app/src/cvars/") || rel.contains("tests") {
                continue;
            }
            let text = std::fs::read_to_string(&file).expect("source is readable");
            for line in text.lines() {
                if !line.trim_start().starts_with("//") {
                    code.push_str(line);
                    code.push('\n');
                }
            }
        }
        let mut orphans = Vec::new();
        for row in REGISTERED {
            if LUA_ONLY.iter().any(|(n, _)| *n == row.name) {
                continue;
            }
            let exact = format!("\"{}\"", row.name);
            let lower = format!("\"{}\"", row.name.to_ascii_lowercase());
            if !(code.contains(&exact) || code.contains(&lower)) {
                orphans.push(row.name);
            }
        }
        assert!(
            orphans.is_empty(),
            "registered rows nothing in the source reads (an observer arm, a `cvars.get`, or a \
             `LUA_ONLY` entry with its stock reader): {orphans:?}"
        );
        for (name, _) in LUA_ONLY {
            assert!(
                REGISTERED.iter().any(|r| r.name == *name),
                "{name}: named Lua-only but not registered"
            );
        }
    }

    /// The rows the reference latches (`flags` 2 or 3 at the register site): the sound-init row and
    /// the `gx*` block `RestartGx` commits. `trilinear`, `anisotropic` and `farclip` register with
    /// `flags = 1` and apply live.
    #[test]
    fn the_latched_rows_are_the_references_own() {
        let mut latched: Vec<&str> = REGISTERED
            .iter()
            .filter(|r| r.latched)
            .map(|r| r.name)
            .collect();
        latched.sort_unstable();
        assert_eq!(
            latched,
            vec![
                "SoundBufferSize",
                "gxApi",
                "gxColorBits",
                "gxCursor",
                "gxDepthBits",
                "gxFixLag",
                "gxMaximize",
                "gxMultisample",
                "gxRefresh",
                "gxResolution",
                "gxTripleBuffer",
                "gxVSync",
                "gxWindow",
            ]
        );
    }

    // ─── MONKEY (reviewfix-a) ────────────────────────────────────────────────────────────────

    /// The value the reference boots a row at, as its [`Reference`] column records it; `None`
    /// for benilla's own rows (nothing to match).
    fn reference_boot_value(row: &Registered) -> Option<&'static str> {
        match &row.reference {
            Reference::Same(v) => Some(v),
            // The reference's own boot code lands where our default does.
            Reference::Overridden { .. } => Some(row.default),
            Reference::Deviates { value, .. } => Some(value),
            Reference::Ours(_) => None,
        }
    }

    /// **The seeded High column against every reference row** (review #6): a governed row a new
    /// player boots off the reference's value must be declared in [`SEEDED_DEVIATIONS`] with
    /// that reference value and a reason, and a declared row must really deviate.
    #[test]
    fn seeded_column_deviates_only_where_declared() {
        let col = graphics_column(GRAPHICS_DEFAULT).unwrap();
        for (k, values) in GRAPHICS_PRESETS {
            if k.eq_ignore_ascii_case("lightingQuality") {
                continue; // High on that ladder IS the registered defaults (its own test)
            }
            let row = REGISTERED
                .iter()
                .find(|r| r.name == *k)
                .expect("governed row registered");
            let declared = SEEDED_DEVIATIONS.iter().find(|(n, _, _)| n == k);
            match reference_boot_value(row) {
                Some(reference) if !same_value(values[col], reference) => {
                    let (_, value, why) = declared.unwrap_or_else(|| {
                        panic!(
                            "{k}: the seeded {GRAPHICS_DEFAULT} value {} leaves the reference's \
                             {reference}; declare it in SEEDED_DEVIATIONS",
                            values[col]
                        )
                    });
                    assert!(
                        same_value(value, reference),
                        "{k}: declared reference is stale"
                    );
                    assert!(!why.trim().is_empty(), "{k}: a deviation owes a reason");
                }
                _ => assert!(
                    declared.is_none(),
                    "{k}: declared in SEEDED_DEVIATIONS but the seed does not deviate"
                ),
            }
        }
        for (n, _, _) in SEEDED_DEVIATIONS {
            assert!(
                GRAPHICS_PRESETS.iter().any(|(k, _)| k == n),
                "{n}: declared but not governed"
            );
        }
    }

    /// Review #14: an env lever owns a governed row for the session; the label must not derive
    /// (and so persist) `Custom` because of it.
    #[test]
    fn a_session_owned_row_does_not_derive_custom() {
        let mut cvars = fresh_registry();
        apply_graphics_preset(&mut cvars, "High");
        lighting_quality(&mut cvars);
        assert_eq!(derive_graphics_quality(&cvars), "High");
        cvars.own_for_session("farclip", Some("500"));
        assert_eq!(cvars.get("farclip"), Some("500"));
        assert_eq!(derive_graphics_quality(&cvars), "High");
        // A lighting member owned by the session is skipped the same way.
        cvars.own_for_session("waterQuality", Some("0"));
        assert_eq!(derive_lighting_quality(&cvars), "High");
    }

    /// Review #15: help strings are one line — a lost `\` continuation leaves a literal `\n` or a
    /// run of indentation spaces in the `/console` text.
    #[test]
    fn help_strings_have_no_broken_continuations() {
        for row in REGISTERED {
            let why = match &row.reference {
                Reference::Ours(why)
                | Reference::Deviates { why, .. }
                | Reference::Overridden { why, .. } => *why,
                Reference::Same(_) => continue,
            };
            assert!(
                !why.contains('\n') && !why.contains("\\n") && !why.contains("   "),
                "{}: broken line continuation in {why:?}",
                row.name
            );
        }
    }

    /// Review #13: a capture reads CVars only from the dedicated `WOW_CAPTURE_CVARS` fixture,
    /// never from a `BENILLA_HOME` the shell happens to carry.
    #[test]
    fn a_capture_reads_cvars_only_from_its_fixture() {
        use crate::local_state::test_env::{EnvGuard, ENV_LOCK};
        let _l = ENV_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let _c = EnvGuard::set("WOW_CAPTURE", "overlook-noon");
        let _h = EnvGuard::set("BENILLA_HOME", "player-home");
        let _x = EnvGuard::unset("WOW_CAPTURE_CVARS");
        assert_eq!(config_read_path(), None);
        let _x = EnvGuard::set("WOW_CAPTURE_CVARS", "fixture/config.toml");
        assert_eq!(
            config_read_path(),
            Some(std::path::PathBuf::from("fixture/config.toml"))
        );
    }
}

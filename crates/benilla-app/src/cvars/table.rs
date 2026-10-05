//! The registry's table: every host-backed CVar, its default, and that default's standing against
//! the reference's registered one ([`Reference`]). The runtime that holds the live values is the
//! parent module.

/// One host-backed CVar: its name, benilla's default, and where that default stands against the
/// reference's ([`Reference`]).
pub(crate) struct Registered {
    /// The registered name, in the reference's own spelling.
    pub(crate) name: &'static str,
    /// What a fresh `benilla-config` runs at, and what `GetCVar` answers until the player moves it.
    pub(crate) default: &'static str,
    /// Where `default` stands against the reference's. Read only by the tests: it records the
    /// reference, never a value this client acts on.
    #[allow(dead_code)]
    pub(crate) reference: Reference,
    /// Registered with flag bit1 (`rec+0x1c & 0x2`, `flags` 2 or 3 at the register site): a write
    /// is staged in [`super::Row::pending`] until [`super::Cvars::commit_latched`].
    pub(crate) latched: bool,
}

impl Registered {
    /// Mark the row latched.
    pub(crate) const fn latched(self) -> Self {
        Self {
            latched: true,
            ..self
        }
    }
}

/// benilla's default against the reference's: the string the reference's `CVar::Register`
/// (`0x63db90`) passes for the name, or, for a setting 1.12 keeps in FrameXML, the value
/// `UIOptionsFrame.lua` boots it at. Not a `Config.wtf` line (`SaveConfig 0x63d980` writes only
/// values off their default), and not always what a fresh install runs at: `hwDetect` rewrites
/// sixteen video CVars from `VideoHardware.dbc` before the first frame ([`Reference::Overridden`]).
/// Each row's register site or FrameXML line is cited above it.
#[allow(dead_code)] // read only by the tests
pub(crate) enum Reference {
    /// The reference registers this default and benilla ships it; the test compares the two.
    Same(&'static str),
    /// The reference registers `registered`, but its own boot code overwrites it before the first
    /// frame; `default` is where that lands, and `why` is the override.
    Overridden {
        registered: &'static str,
        why: &'static str,
    },
    /// The reference ships `value` and benilla ships another; `why` is the reason.
    Deviates {
        value: &'static str,
        why: &'static str,
    },
    /// No reference setting to match: benilla's own knob, or a later-era name for something 1.12
    /// never made settable. `why` says which, and what the reference does instead.
    Ours(&'static str),
}

/// A row whose default is the reference's own registered string.
const fn same(name: &'static str, default: &'static str) -> Registered {
    Registered {
        name,
        default,
        reference: Reference::Same(default),
        latched: false,
    }
}

/// A row whose default follows the reference's boot-time override of its registered string.
const fn overridden(
    name: &'static str,
    default: &'static str,
    registered: &'static str,
    why: &'static str,
) -> Registered {
    Registered {
        name,
        default,
        reference: Reference::Overridden { registered, why },
        latched: false,
    }
}

/// A row that ships something other than the reference's `value`, for `why`.
const fn deviates(
    name: &'static str,
    default: &'static str,
    value: &'static str,
    why: &'static str,
) -> Registered {
    Registered {
        name,
        default,
        reference: Reference::Deviates { value, why },
        latched: false,
    }
}

/// A row the reference has no counterpart for.
const fn ours(name: &'static str, default: &'static str, why: &'static str) -> Registered {
    Registered {
        name,
        default,
        reference: Reference::Ours(why),
        latched: false,
    }
}

/// The table as the script VM's registrar wants it: `(name, default)` pairs in table order.
pub(crate) fn registered_pairs() -> impl Iterator<Item = (&'static str, &'static str)> {
    REGISTERED.iter().map(|r| (r.name, r.default))
}

/// The host-backed CVars, one row per knob that has a reader: a host knob, or a Lua consumer
/// such as the stock Video Options window's verbs.
///
/// Not registered for want of a reader, though pfUI's `hdgraphic` writes them: `lodDist`
/// (`0x688524`, "100.0", read at `0x6afb1d` for the doodad LOD swap), `footstepBias` (`0x6888b4`,
/// "0.125", read at `0x68fcb6`), `mapObjLightLOD` (`0x6886ec`, "0") and `SkyCloudLOD` (`0x6d1d33`,
/// "0"). `DistCull` (`0x688570`) has no reader in the reference either. `maxLOD` is no 1.12 CVar.
pub(crate) const REGISTERED: &[Registered] = &[
    // `realmName` (`0x83f2d0`): registered `""` (`0x882748`), help "Last realm connected to"
    // (`0x85d684`); the client builds its SavedVariables path from it (`0x5ab7d0`). Written from
    // the session's realm when addons load. `Ace/AceState.lua:27` trims it at
    // PLAYER_ENTERING_WORLD, so a nil breaks every Ace addon.
    same("realmName", ""),
    // The logon server address (register site `0x5ab6a6`), a string row judged by
    // `realmlist::on_cvar`.
    deviates(
        crate::realmlist::CVAR_REALMLIST,
        crate::realmlist::DEFAULT_REALMLIST,
        "us.logon.worldofwarcraft.com:3724",
        "that host has not resolved since 2019, so shipping it makes every first launch a \
         DNS failure; benilla dials the machine it is running on",
    ),
    // `autoClearAFK` (`0x5e24d4`, "1" `0x82e748`, handle `[0xc4d68c]` set at `0x5e24ef`, read at
    // `0x5eb84b`) gates five implicit AFK clears: any chat send but type `0x14`, Jump,
    // forward/back, strafe and turn (`0x513d36`/`0x514e23`/`0x514f0b`/`0x514fca`). Off, the clear
    // does nothing at all: no echo, no mirror write, no packet.
    same("autoClearAFK", "1"),
    same("MasterVolume", "1"),
    same("SoundVolume", "1"),
    same("MusicVolume", "0.4"),
    same("AmbienceVolume", "0.6"),
    // Registered "1" at `0x45737a`/`0x45739b`/`0x460a9d`. `MasterSoundEffects` is the Enable All
    // Sound box (`SoundOptionsFrame.lua:6`), which pauses the whole sound engine, not an SFX
    // toggle.
    same("MasterSoundEffects", "1"),
    same("EnableMusic", "1"),
    same("EnableAmbience", "1"),
    // The race/sex refusal voice lines (`0x457877`), `SoundOptionsFrame.lua:3`; the master enable
    // greys it.
    same("EnableErrorSpeech", "1"),
    // Not a 1.12 CVar or checkbox; the later-era name. The reference mutes on losing focus, music
    // included (`WM_ACTIVATE` to `0x7a4860`'s `FSOUND_SetMute(-3, active ? 0 : 1)`), so "0" is its
    // behaviour. The knob is `SoundConfig::background_sound`.
    same("Sound_EnableSoundWhenGameIsInBG", "0"),
    // Registered "1" (`0x4573be`); `SoundConfig::reverb` carries the evidence for shipping "0".
    deviates(
        "SoundReverb",
        "0",
        "1",
        "the reference's reverb is EAX-over-hardware, and that hardware has not existed since \
         Vista, so \"1\" would ship audio the real client has never actually produced on any \
         machine a player runs today",
    ),
    // FMOD 3's mix-ahead buffer in ms (`0x4571ca`, flags 2: read once at sound init); here it sizes
    // the render thread's ring ahead of the IO callback (`sound::output`). `0x457520` registers
    // "50" or "100" by a host probe (`0x835e10`/`0x835e0c`), which one on a current machine
    // untraced; ours is the larger, since the depth has to hide a whole stalled IO cycle.
    same("SoundBufferSize", "100").latched(),
    // benilla's own, not a 1.12 CVar. Deviation: the reference clips at full scale and has no
    // headroom mechanism (its SFX-bus duck `0x457960` is a sidechain armed only by server-pushed
    // voice lines); benilla limits because every SFX is mastered to full scale and overlapping
    // kits clip.
    ours(
        "SoundOutputLimiter",
        "1",
        "benilla's own — the reference sums at full scale and clips; every WoW SFX is mastered \
         to full scale, so overlapping kits need a limiter to keep from distorting",
    ),
    overridden(
        "uiScale",
        "0.9",
        "1.0",
        "a fresh reference client never consults this CVar: `useUiScale` registers \"0\" \
         (`0x48fce4`), and the OFF leg `0x492f70` computes clamp(768/height, 0.9, 1.0) instead — \
         0.9 at 854 px tall and up, which is every window we ship against. It is 1.0 at 768 and \
         below, where our flat 0.9 does diverge; `ui_script::DEFAULT_UI_SCALE` carries that. \
         See `useUiScale` below, whose row this one used to say did not exist",
    ),
    // `useUiScale` (`0x8430c0`), the switch the `uiScale` override gates on:
    // `ContainerFrame.lua:483` and `UIDropDownMenu.lua:525` branch on it and `OptionsFrame.lua:13`
    // gives it a checkbox.
    same("useUiScale", "0"),
    same("farclip", "350"),
    // `nearclip` (`0x68867a`: name `0x84ffb0`, default `0x84fb48` "0.1", flags 1, callback
    // `0x688d90`, record `[0xc7f348]`). The camera re-reads the record every frame: `0x511bc0`
    // (sole caller `0x483094`) stamps `[cam+0x38]` from it through the handle `[0xbe1078]` that
    // `0x50b728` caches, overwriting the ctor's 1/9 (`0x3de38e39`);
    // `benilla_world::view::stamp_near_clip` is that.
    // The callback's derived global `[0xc7b480]` has no reader. pfUI's `hdgraphic` writes
    // 0.06..0.30, inside `[0.01, 0.33]`.
    same("nearclip", "0.1"),
    // `deselectOnClick` and `mouseInvertPitch` are 1.12's own (`UIOptionsFrame.lua:8,4`);
    // `autoLootDefault` is the later-era name, 1.12 having only the shift gesture.
    same("deselectOnClick", "1"),
    // `AutoInteract` (`0x603390`, record `[0xc4d9a4]`), Click-to-Move (`UIOptionsFrame.lua:7`):
    // "1" on koKR alone (`0x603368`). The knob is [`crate::player::Approach`].
    same("AutoInteract", "0"),
    // `BlockTrades` (`0x842fbc`), `UIOptionsFrame.lua:11`: the refusal leg `0x4bf7bc` fires only
    // when it is set. The knob is [`crate::ui_trade::BlockTrades`].
    same("BlockTrades", "0"),
    // `autoSelfCast` (register site `0x6e731d`, record `[0xceac34]`, read at `0x6e53d7`; `0x870dc0`
    // is its name string): a friendly cast that binds nothing falls back to the caster.
    // `TOGGLEAUTOSELFCAST` toggles it.
    same("autoSelfCast", "0"),
    // The spell queue, benilla's own: the knob is [`crate::spell::inflight::PendingCast`]. The reference
    // holds the next cast until the server resolves the last (`0xceca88`, cleared at `0x6e7408`).
    ours(
        "spellQueue",
        "0",
        "benilla's own, off by default — the reference waits one round trip for the server's \
         resolution before the next cast; on, a timed cast's guard opens on the local timer",
    ),
    ours(
        "spellQueueBufferMs",
        "55",
        "benilla's own — the spell queue's margin past the local cast end for the server's tick, \
         nampower's 55 ms default",
    ),
    // The five saved camera views and the live index, at the reference's names and default strings;
    // owned by [`crate::player::camera_view`]. Registered so a `SaveView` persists.
    same(
        crate::player::camera_view::CVAR_ACTIVE_VIEW,
        crate::player::camera_view::ACTIVE_VIEW_DEFAULT,
    ),
    same(
        crate::player::camera_view::VIEW_CVARS[0][0],
        crate::player::camera_view::VIEW_DEFAULTS[0][0],
    ),
    same(
        crate::player::camera_view::VIEW_CVARS[0][1],
        crate::player::camera_view::VIEW_DEFAULTS[0][1],
    ),
    same(
        crate::player::camera_view::VIEW_CVARS[0][2],
        crate::player::camera_view::VIEW_DEFAULTS[0][2],
    ),
    same(
        crate::player::camera_view::VIEW_CVARS[1][0],
        crate::player::camera_view::VIEW_DEFAULTS[1][0],
    ),
    same(
        crate::player::camera_view::VIEW_CVARS[1][1],
        crate::player::camera_view::VIEW_DEFAULTS[1][1],
    ),
    same(
        crate::player::camera_view::VIEW_CVARS[1][2],
        crate::player::camera_view::VIEW_DEFAULTS[1][2],
    ),
    same(
        crate::player::camera_view::VIEW_CVARS[2][0],
        crate::player::camera_view::VIEW_DEFAULTS[2][0],
    ),
    same(
        crate::player::camera_view::VIEW_CVARS[2][1],
        crate::player::camera_view::VIEW_DEFAULTS[2][1],
    ),
    same(
        crate::player::camera_view::VIEW_CVARS[2][2],
        crate::player::camera_view::VIEW_DEFAULTS[2][2],
    ),
    same(
        crate::player::camera_view::VIEW_CVARS[3][0],
        crate::player::camera_view::VIEW_DEFAULTS[3][0],
    ),
    same(
        crate::player::camera_view::VIEW_CVARS[3][1],
        crate::player::camera_view::VIEW_DEFAULTS[3][1],
    ),
    same(
        crate::player::camera_view::VIEW_CVARS[3][2],
        crate::player::camera_view::VIEW_DEFAULTS[3][2],
    ),
    same(
        crate::player::camera_view::VIEW_CVARS[4][0],
        crate::player::camera_view::VIEW_DEFAULTS[4][0],
    ),
    same(
        crate::player::camera_view::VIEW_CVARS[4][1],
        crate::player::camera_view::VIEW_DEFAULTS[4][1],
    ),
    same(
        crate::player::camera_view::VIEW_CVARS[4][2],
        crate::player::camera_view::VIEW_DEFAULTS[4][2],
    ),
    same("mouseInvertPitch", "0"),
    ours(
        "autoLootDefault",
        "0",
        "1.12 has no auto-loot CVar at all — vanilla offers only the shift gesture, so OFF \
         IS the reference's own behaviour; the spelling is era's",
    ),
    // The overhead-name gates, registered at `0x6c7470` into mask `0xce8720`: `UnitNamePlayer`
    // (`0x86c694`) "1" (`0x82e748`), `UnitNameNPC` (`0x86c6a4`) and `UnitNameOwn` (`0x86c6b0`) "0"
    // (`0x82e570`).
    same("UnitNamePlayer", "1"),
    same("UnitNameNPC", "0"),
    same("UnitNameOwn", "0"),
    // `UnitNamePlayerGuild` (`0x86c680`, "1", mask bit `0x10`) is not a show gate: `ShouldShowName`
    // (`0x6070a0`) reads bits `0x1/0x2/0x4`, and this gates the `"\n<%s>"` guild line at
    // `0x609085`. `UnitNamePlayerPVPTitle` (bit `0x20`, "1") has no row: nothing here draws the
    // rank prefix.
    same("UnitNamePlayerGuild", "1"),
    // `WorldDetail` is no 1.12 CVar but the `GetWorldDetail`/`SetWorldDetail` verb name
    // (`OptionsFrame.lua:27`); stops 0/1/2 are `frillDensity` 16/32/48. `SetWorldDetail 0x488dd0`
    // also writes `SmallCull` {0.07, 0.04, 0.01}, and `GetWorldDetail` reads only `SmallCull`,
    // whose registered 0.04 is stop 1: the reference's slider boots at Medium.
    same("WorldDetail", "1"),
    // `SmallCull` (`0x68854a`: name `0x8696f8`, default `0x869718` "0.04", flags 1, callback
    // `0x688b10`, record `[0xc7f330]`). The callback refuses outside [0.001, 2.0] and stores into
    // `[0x868620]`, which nothing reads: no mechanism here either. Its readers are Lua ones:
    // `SetWorldDetail` writes it per stop, and `OptionsFrame_SetDefaults` reads its default
    // (`OptionsFrame.lua:431`). `hwDetect` sets 0.04 on the fallback row a modern GPU reaches.
    same(benilla_ui::script::CVAR_SMALL_CULL, "0.04"),
    // `frillDensity` (`0x68862e`: name `0x8423d8`, default `0x864644` "16", flags 1, callback
    // `0x688de0`, record `[0xc7f2f4]`): detail-doodad cells visited per chunk, clamped to [1, 256]
    // and handed through `0x6725a0` to `[0xc7b494]`, the bound of the scatter loop at
    // `0x6bfcfb`/`0x6bff1c`. One knob with `WorldDetail`, each keeping its own clamp. `hwDetect`
    // (`0x639a60`) sets it from `VideoHardware.dbc` field `+0x18`, 24 on videoID 170. pfUI's
    // `hdgraphic` reads `GetCVar("frillDensity") > 48` and writes up to 256.
    deviates(
        "frillDensity",
        "32",
        "16",
        "the reference's registered 16 is stop 0 and its post-`hwDetect` 24 is on no \
         stop at all, so every stop diverges; Medium (32) is the nearest one no sparser than a \
         fresh install, and erring sparse is the worse failure for ground cover",
    ),
    // ── Combat log display ranges, in yards ──────────────────────────────────────────────────────
    //
    // Registered by `0x626d00` from the `{name, default}` pairs at `0x8629e0`, read as the record's
    // float (`+0x24`). Classes 0 and 1, you and your pet, have no CVar there, only the `100000.0`
    // sentinel.
    same("CombatLogRangeParty", "50"),
    same("CombatLogRangePartyPet", "50"),
    same("CombatLogRangeFriendlyPlayers", "50"),
    same("CombatLogRangeFriendlyPlayersPets", "50"),
    same("CombatLogRangeHostilePlayers", "50"),
    same("CombatLogRangeHostilePlayersPets", "50"),
    same("CombatLogRangeCreature", "30"),
    // Outside that table (`0x626d5f`, "60" at `0x862e14`): `0x62c160` reads it first and falls back
    // to the per-class range only when the lookup fails.
    same(crate::ui_chat::combat::DEATH_LOG_RANGE_CVAR, "60"),
    // ── Floating combat text ─────────────────────────────────────────────────────────────────────
    //
    // `CombatDamage` (`0x6032df`, record `[0xc4d944]`) is the master: its two readers, the word
    // emitter `0x607140` and the number emitter `0x6128b0`, both return at "0", so nothing floats.
    // The `Pet*` rows gate only the owned-by-you branch; the stock Pet Melee Damage box writes both
    // (`UIOptionsFrame.lua:335-336`).
    same("CombatDamage", "1"),
    same("PetMeleeDamage", "1"),
    same("PetSpellDamage", "1"),
    // `CombatLogPeriodicSpells` (`0x6033b3`): the handle is discarded and every use looks it up by
    // name; read as the record's int.
    same(crate::ui_chat::combat::LOG_PERIODIC_CVAR, "1"),
    // ── Sound-panel check buttons ────────────────────────────────────────────────────────────────
    //
    // Category-7 registrations that keep no handle: the reference looks each up by name at use.
    //
    // `SoundListenerAtCharacter` (`0x457890`): the listener at the character, else at the camera
    // (`update_audio_listener`).
    same("SoundListenerAtCharacter", "1"),
    // `EmoteSounds` (`0x4573b9`): the received text-emote voice kit, and only that.
    same("EmoteSounds", "1"),
    // `SoundZoneMusicNoDelay` (`0x4578b3`): `next_track_time`'s immediate path.
    same("SoundZoneMusicNoDelay", "0"),
    // `assistAttack` (`0x48fc50`, record `[0xb4d8f8]`): `/assist` also starts the swing. The "3"
    // beside it is the next registration's, `minimapZoom`: stock `/assist` selects without
    // swinging.
    same("assistAttack", "0"),
    // ── Mouse-look speed, per axis ───────────────────────────────────────────────────────────────
    //
    // `cameraYawMoveSpeed` is the MOUSE_LOOK_SPEED slider (`UIOptionsFrame.lua:89`); a nil there
    // raises in `slider:SetValue` (`0x790980`) and stops `UIOptionsFrame_Load`. The stock Save
    // writes `cameraPitchMoveSpeed` as half of it (`:355-356`). The reference integrates
    // OS-accelerated pixels where we take raw device deltas, so the unit factor lives in
    // `camera::LOOK_YAW_PER_SPEED` and these defaults stay the reference's. The validator
    // `0x50c000` → `0x50b330` rejects values outside [0.1, 360] rather than clamping, and
    // `player::camera::on_cvar` does the same.
    same("cameraYawMoveSpeed", "180"),
    same("cameraPitchMoveSpeed", "90"),
    // MOUSE_SENSITIVITY (`UIOptionsFrame.lua:87`): FrameXML's spelling of the binary's `mouseSpeed`
    // (`0x402c7b`); lookups are case-insensitive (`CVar::Lookup 0x63de30`), here as there. The
    // reference's default is `SPI_GETMOUSESPEED × 0.1`, "1.0" on stock Windows, and its record
    // `[0x882704]` has no reader: the slider sets the OS pointer speed (`0x402ec0`, [0.1, 2.0]).
    // The default matches; Deviation: here the dial multiplies the camera's own rate, because
    // benilla does not change the OS pointer speed, a system-wide setting.
    same("mousespeed", "1"),
    // The zoom-out limit is `cameraDistanceMax × cameraDistanceMaxFactor`, held to [0, 50]
    // (`0x5112d6`). `cameraDistanceMax` (`0x50beb2`, default `0x84fbd0`) has no panel row; its
    // validator `0x50b310` refuses a value outside [0, 50], and `player::camera::on_cvar` does the
    // same. The factor is MAX_FOLLOW_DIST (`UIOptionsFrame.lua:90`), registered "1.0" (`0x82e92c`)
    // with no validator: the slider's 1 to 2 is the slider's, not the CVar's.
    same("cameraDistanceMax", "15.0"),
    same("cameraDistanceMaxFactor", "1"),
    // `cameraSmoothStyle` (`0x50ba92`, default `[0x84f4f4]` "1"), the auto-return behind the
    // character. The engine's enum is 0 Never, 1 Smart, 2 Always, as the stock dropdown writes it
    // (`UIOptionsFrame.lua:525,536,547`); 3, the Never entry's position, is the validator's upper
    // bound (`0x50b330(v, 0, 3)`). See `FollowStyle`.
    same("cameraSmoothStyle", "1"),
    // Read instead of `cameraSmoothStyle` while the state mask holds Track or Fear; no panel row.
    same("cameraSmoothTrackingStyle", "1"),
    // AUTO_FOLLOW_SPEED (`UIOptionsFrame.lua:88`), deg/s, registered "180.0" (`[0xbe1070]`): it
    // sets the transition's duration (`|dyaw| / rate * factor`), an average rate, not a slew. The
    // knob clamps it to `FOLLOW_SPEED_RANGE`. The stock Save also writes `cameraPitchSmoothSpeed`
    // at a quarter (`:353`), unregistered here: `FollowRig` has one rate.
    same("cameraYawSmoothSpeed", "180"),
    // `cameraPivot` `[0xbe10a4]` "1" (`0x50bda3`), smart pivot: gate `0x510690`, routing
    // `0x50fee0`, release `0x5107f0`; ours is `player::camera_dynamics::SmartPivot`.
    same("cameraPivot", "1"),
    // Read by the routing (`0x50fff5`/`0x510004`) in radians of camera rotation, so they carry to
    // benilla's raw-device units unchanged.
    same("cameraPivotDXMax", "0.05"),
    same("cameraPivotDYMin", "0"),
    // The pitch bias's ease-back once the pivot lets go, deg/s (`[0xbe0fc8]`; `0x512a50` divides
    // |Δ| by rate · π/180 for the duration); no panel row.
    same("cameraTargetSmoothSpeed", "90"),
    // `cameraWaterCollision` `[0xbe1088]` "1" (`0x50bd63`, `0x82e748`): one register, two consumers
    // that must ship together. `0x50e5ec` builds it; its `0xf0000` nibble joins the trace mask of
    // all three `0x50e570` queries (reaching `0x69cc13`), and `0x50e629` tests it to lift the sweep
    // origin to `surface + 2/9`. Ours: `benilla_world::collision::camera_filter` and
    // `player::camera_water`.
    same("cameraWaterCollision", "1"),
    // `cameraTerrainTilt` `[0xbe0fd4]` "0" (`0x50bcfd`), Follow Terrain: probe and staircase
    // `0x50d900`, arm `0x50dbc0`; ours is `player::camera_dynamics::TerrainTilt`.
    same("cameraTerrainTilt", "0"),
    // Rate in deg/s (`[0xbe0fc0]`), duration bounds in seconds (`[0xbe1050]`/`[0xbe1054]`). The 3 s
    // floor always binds (20° at 7.5°/s is 2.67 s), so the camera leans rather than tracks.
    same("cameraGroundSmoothSpeed", "7.5"),
    same("cameraTerrainTiltTimeMin", "3"),
    same("cameraTerrainTiltTimeMax", "10"),
    // `cameraBobbing` `[0xbe10c0]` "0" (`0x50b76d`), head bob: kernel `0x511920`, gate `0x5105e0`;
    // ours is `player::camera_dynamics::HeadBob`.
    same("cameraBobbing", "0"),
    // Amplitudes in the CVar's units, scaled by 1/36 (`[0x7ff9d0]`) to yards.
    // `cameraBobbingSmoothSpeed` is the decay rate, read only in the disarm `0x51113a`, which
    // divides the largest component by it for the ramp's duration (~0.069 s at these defaults).
    same("cameraBobbingLRAmplitude", "2"),
    same("cameraBobbingUDAmplitude", "2"),
    same("cameraBobbingFrequency", "0.8"),
    same("cameraBobbingSmoothSpeed", "0.8"),
    // `statusBarText` (`0x48fc34`, record `[0xb4d904]`, no engine reader), read by
    // `TextStatusBar.lua:47,97`: "0" shows the numbers on hover only.
    same("statusBarText", "0"),
    // Enhanced Tooltips (`UIOptionsFrame.lua:15`), registered "1" at `0x48fddd` (`0x82e748`); read
    // only by the stock interface.
    same("UberTooltips", "1"),
    // Registered at `0x603280`: `ChatBubbles` "1", `ChatBubblesParty` "0".
    same("ChatBubbles", "1"),
    same("ChatBubblesParty", "0"),
    // Registered "1" (`0x82e748`), category 4. `profanityFilter` (`0x402e68`, name `0x82e7f4`,
    // callback `0x403570`) masks `ChatProfanity.dbc` spans inside the shared masker `0x4a1a66`,
    // covering all thirteen call sites; `spamFilter` (`0x402e8e`, name `0x82e7d4`, callback
    // `0x4035b0`) silently drops a line matching `SpamMessages.dbc`. The knob is
    // [`crate::text_filter::TextFilterSwitches`].
    same("profanityFilter", "1"),
    same("spamFilter", "1"),
    // Registered by `CGlueMgr::EnterWorld` (`0x46b633` `gameTip` "0", `0x46b658` `showGameTips`
    // "1"), category 5. `gameTip` is the cursor and holds the next tip, not the one on screen;
    // `crate::game_tip` advances it.
    same("gameTip", "0"),
    same("showGameTips", "1"),
    // `showLootSpam` (`0x48fd1c`, name `0x8430a0`, "1" `0x82e748`, record `0xb4e2bc`): off, group
    // loot-roll lines are hidden and only the winner shows. Its three readers are the roll-line
    // composers; the knob is [`crate::ui_loot::LootConfig::show_loot_spam`].
    same("showLootSpam", "1"),
    // `guildMemberNotify` (`0x5e24c7`, "0" `0x82e570`, record `0xc4d3c4`): guildmate log on/off
    // lines, read only in `SMSG_GUILD_EVENT`'s handler. The knob is
    // [`crate::ui_guild::GuildMemberNotify`].
    same("guildMemberNotify", "0"),
    // Both registered "3" (`0x48fc6c`, `0x48fc88`); the minimap's +/- buttons write them through
    // `Minimap:SetZoom`. The knob is [`crate::minimap::MinimapZoom`].
    same("minimapZoom", "3"),
    same("minimapInsideZoom", "3"),
    // Load out of date AddOns, inverted (`0x402c3b`): "1" enforces the check. Read by the load walk
    // ([`Cvars::addon_version_check`]) and live by the VM's gate.
    same("checkAddonVersion", "1"),
    // `gxApi` (`0x63a833`: name `0x842a64`, default `0x864f7c` "direct3d", flags 3, callback
    // `0x63b030`, record `[0xc4ea94]`): the reference builds D3D9 unless it reads "OpenGl"
    // (`0x63a3c4`, `0x842a5c`). Here it reports the wgpu backend (`wgpu::Backend::to_str`), pushed
    // from `RenderAdapterInfo` and owned by the session, so it is never persisted. pfUI's
    // `panel.lua:185` concatenates it.
    deviates(
        "gxApi",
        "",
        "direct3d",
        "descriptive, not a selector — benilla renders through wgpu, which has no D3D9 \
         backend and no chooser; the value is the live adapter's own and is never persisted",
    )
    .latched(),
    // `gxVSync` (`0x63a859`, "1", flags 3), `OptionsFrame.lua:9`. The knob is
    // [`crate::video::VideoConfig::vsync`], which the window's present mode follows at the
    // `RestartGx` commit. `$WOW_NOVSYNC=1` overrides it for the session.
    same("gxVSync", "1").latched(),
    // MONKEY (advanced graphics): the one row on the Advanced Graphics page that is not a knob —
    // it is a NAME for a combination of the rows under it ([`LIGHTING_PRESETS`]). Writing a preset
    // name writes that preset's members; writing anything else is corrected on the next frame by
    // [`lighting_quality`], which re-DERIVES the name from the members those rows actually hold.
    // So the stored value is never stale: it is either a preset every member agrees with, or
    // `Custom`.
    //
    // A STRING row, not an int, so `config.toml` and `/console set lightingQuality Medium` both
    // read as what they mean — and so the ladder can gain a rung without renumbering a player's
    // saved value. `Row::numeric` is false for it (the default does not parse as a number), which
    // is what lets the registry accept a non-numeric write at all; `realmList` is the precedent.
    ours(
        "lightingQuality",
        "High",
        "benilla's own: the preset ladder over the dynamic light + shadow rows — Off / Low / \
         Medium / High / Ultra, or Custom when the members match none of them; the reference has no \
         realtime light or shadow system to preset",
    ),
    // MONKEY (presets): the Graphics Preset — `lightingQuality`'s posture one level up, a NAME
    // for every graphics row at once ([`GRAPHICS_PRESETS`]), re-derived every frame. "High" is
    // what a player with no saved preset boots into: [`Cvars::seed_graphics_preset`] writes the
    // High column over every row their `config.toml` does not carry, so the rows themselves keep
    // their per-lane defaults (Classic-leaning, and `farclip` the reference's 350) and a capture
    // or a test, which never seeds, keeps them too.
    ours(
        "graphicsQuality",
        "High",
        "benilla's own: the preset ladder over every graphics row — Classic / Low / Medium / High \
         / Ultra, or Custom when the rows match none of them; the reference's options have no \
         such preset",
    ),
    // MONKEY (volumetric fog): saved live tier; capture override stays session-only.
    ours("volumetricFog", "1", "benilla's own: near-field volumetric fog, 0 Off / 1 Low / 2 High"),
    // MONKEY (lampfog): opt-in point-light halos; High graphics preset value is 2.
    ours(
        "lampFog",
        "0",
        "benilla's own: lamps scattering through night fog, 0 Off / 1 Low (16 lamps) / 2 High (32 lamps)",
    ),
    // GFX (volumetric light): shadow-mapped light shafts for the sun and the moon.
    ours(
        "volumetricLight",
        "0",
        "benilla's own: sun and moon light shafts marched through the shadow map, 0 Off / 1 Medium / 2 High",
    ),
    ours(
        "volumetricLightStrength",
        "1",
        "benilla's own: brightness of the volumetric light shafts, 0..2 (1 = default)",
    ),
    // GFX (moonlight): the moon as a light on top of the stock night.
    ours(
        "moonLight",
        "0",
        "benilla's own: moonlight on top of the stock night, 0..2 (0 = the reference night)",
    ),
    // MONKEY (p0 skyDither): the FFXGlow combine's deband dither (was env WOW_DITHER only).
    // Default 0 = the reference look; the Graphics preset's High sets 1.
    ours(
        "skyDither",
        "0",
        "benilla's own: faint screen dither against sky and fog banding, 0 Off / 1 On",
    ),
    // MONKEY (post): tier 0 leaves every emissive site and the frame byte-identical.
    ours("bloom", "2", "benilla's own: HDR emissive bloom, 0 Off / 1 Low / 2 High"),
    ours("sunShafts", "1", "benilla's own: depth-occluded screen-space sun shafts"),
    ours("colorGrading", "1", "benilla's own: zone and day/night 32-cube colour grade"),
    // MONKEY (sky): the sky tier; the High graphics preset sets 2.
    ours(
        "skyQuality",
        "0",
        "benilla's own: sky quality, 0 Classic / 1 Enhanced (smooth gradient, sun glow, stars) / \
         2 High (+ detailed sun-lit clouds)",
    ),
    // MONKEY (wind): one tier controls the grass-only and grass-plus-tree receivers.
    ours(
        "foliageWind",
        "2",
        "benilla's own: foliage wind, 0 Off / 1 Grass / 2 Grass + trees",
    ),
    // MONKEY (fix-wind): the sway strength slider under Foliage Wind; not on the preset ladder.
    ours(
        "foliageWindStrength",
        "1",
        "benilla's own: foliage wind strength, a gain on the sway, 0.25..3 (1 = the shipped tuning)",
    ),
    // MONKEY (fog): the distance-fog model. 0 = the 1.12 linear fog (byte-identical), 1 = Modern
    // (exponential, sun/horizon colour, end-fog shift, fog end decoupled from farclip past 777).
    // Default 0; the Graphics preset's High sets 1.
    ours(
        "fogModel",
        "0",
        "benilla's own: distance fog, 0 Classic (1.12 linear) / 1 Modern (soft, sun-coloured horizon)",
    ),
    // MONKEY (wet): rain darkens and glosses sky-exposed surfaces and rings the water. Default 1
    // (only visible while it rains or the ground dries); the Graphics preset's High sets 1.
    ours(
        "rainSurfaces",
        "1",
        "benilla's own: rain wets sky-exposed surfaces and rings the water, 0 Off / 1 On",
    ),
    // MONKEY (ao): opt-in contact shadows; the High graphics preset value is 2.
    ours("ambientOcclusion", "0", "benilla's own: screen-space ambient occlusion, 0 Off / 1 Low / 2 High"),
    // MONKEY (skybox): the living player's zone skybox from `LightParams`; High sets 1.
    ours(
        "zoneSkyboxes",
        "0",
        "benilla's own: draw the zone skybox LightParams names for the living, 0 Off / 1 On; the \
         reference draws a DBC skybox only for the ghost",
    ),
    ours(
        "waterQuality",
        "1",
        "benilla's own: water quality, 0 Classic / 1 Enhanced / 2 High; mirror reflections are opt-in",
    ),
    ours(
        "lavaLightGain",
        "1",
        "benilla's own: brightness of lava lighting its surroundings, 0..4 (0 = no glow)",
    ),
    // Benilla's opt-in realtime shadow-map path, split into two INDEPENDENT lanes over one shared
    // shadow rig (one sun / one map). `worldShadows` = the static world (trees, buildings, foliage)
    // casts realtime shadows and baked MCSH terrain shadows switch off; `characterShadows` =
    // players/NPCs/creatures/mounts cast realtime silhouettes instead of the legacy oval blob.
    //
    // `ours(...)`, not `same(...)`: the reference has no realtime shadow at all — it bakes MCSH
    // into the terrain and draws an oval under every unit — so there is no registered default for
    // these two to agree with, and claiming `Same` would put a false entry in the one column
    // [`Reference`] exists to keep honest. (They read `same("…", "1")` until the 2303 registry
    // port; the default string is unchanged.)
    ours(
        "worldShadows",
        "1",
        "benilla's own: the static world (trees, buildings, alpha-tested foliage) casts a realtime \
         shadow and the baked MCSH terrain shadows stand aside; the reference bakes and has no \
         such switch",
    ),
    ours(
        "characterShadows",
        "1",
        "benilla's own: units cast a realtime silhouette instead of the reference's oval blob \
         decal, which is all 1.12 has and is therefore not a setting there",
    ),
    // Realtime-shadow render distance in yards (the shadow-map cascade range + caster reach).
    // benilla's own — the reference has no realtime shadow to size. Clamped to SHADOW_DISTANCE_RANGE.
    ours(
        "shadowDistance",
        "80",
        "1900: benilla's own realtime-shadow render-distance slider; the reference bakes MCSH and \
         has no cascade to size",
    ),
    // MONKEY (sun shadow perf): the five live dials over the sun lanes' ~5 ms/frame (RTX 3070,
    // 1080p, both lanes on: 45-47 fps, and 68-73 with both off). All benilla's own — the reference
    // bakes MCSH and has no realtime shadow to tune. `shadow_core`'s constants block holds the cost
    // split each one takes; every row is LIVE, so the whole set A/Bs from one chat line.
    ours(
        "shadowMapSize",
        "2048",
        "benilla's own: directional shadow-map edge in texels, 1024/2048/4096 (cost is quadratic)",
    ),
    ours(
        "shadowFilter",
        "1",
        "benilla's own: shadow PCF kernel: 0 hardware-2x2 (1 sample), 1 gaussian (9, the look)",
    ),
    ours(
        "characterShadowRate",
        "30",
        "benilla's own: no effect (character shadows skin on the GPU every frame); kept until retired",
    ),
    ours(
        "worldShadowRate",
        "30",
        "benilla's own: Hz cap on the world lane's environment caster, 0..120 (0 = per frame)",
    ),
    ours(
        "shadowCasterReach",
        "1",
        "benilla's own: multiplier on the shadow caster-collection reach, 0.25..2 (1 = unchanged)",
    ),
    // MONKEY (moon shadows): how dark a MOON-shadowed fragment is allowed to get at night — the
    // dial over the night lane's own shadow term (`benilla_world::lighting::MoonShadowStrength`,
    // bridged by `dynamic_interior` beside the spell gain). `0` is the FAITHFUL null: every
    // consumer guards on it, so a night at 0 renders as the build before the feature did — which
    // is also what the Off and Low presets ask for.
    ours(
        "moonShadowStrength",
        "0.35",
        "benilla's own: how dark a moon-shadowed fragment gets at night, 0..1 (0 = no moon \
         shadow, the reference's own night)",
    ),
    // MONKEY (dynamic interiors): WMO interiors + their props light from the room's LIVE fixtures
    // instead of the MOCV bake / the baked prop probe (`static_gx.wgsl` `interior_room_light`;
    // bridged by `dynamic_interior`). The three numeric knobs are live-tunable from chat —
    // `/script SetCVar("interiorExposure", 2)` — which is how their defaults were found.
    ours(
        "interiorLight",
        "1",
        "benilla's own: fixture-lit WMO interiors (0 = the reference's baked interior path)",
    ),
    ours(
        "interiorAmbient",
        "0.015",
        "benilla's own: interior base ambient, 0..1",
    ),
    ours(
        "interiorFill",
        "0.08",
        "benilla's own: interior per-fixture bounce gain, 0..2",
    ),
    ours(
        "interiorExposure",
        "2.5",
        "benilla's own: interior light-budget multiplier before the soft rolloff, 0.25..8",
    ),
    // MONKEY (soft falloff): the live scale on every interior fixture's AUTHORED attenuation
    // window (WMO MOLT `+0x28/+0x2c`; M2 sources bucket by intensity instead — their authored pair
    // is a template default, not a reach). A fixture's EFFECTIVE RADIUS is `authored end × this`.
    // The artists' own ends — Goldshire inn 6.97-9.53 yd over 10 fixtures, its blacksmith 6.0,
    // NSabbey 4.17-5.56, Stormwind's 606 median 6.94 — are where FULL brightness ends, not where
    // light stops, so `1` drew a hard-edged disc at exactly that radius with black beyond it (the
    // abbey candelabra ring). **2.5** is the default: the pool now tails smoothly to 2.5× the
    // authored end, reading ~⅓ of its 1 yd brightness AT the authored end and ~8 % at twice it.
    // `>1.6` widens further, `<1.6` tightens, `0` switches the window off (the 48 yd lane), so the
    // whole shape A/Bs from chat.
    ours(
        "interiorAttenScale",
        "1.6",
        "benilla's own: scale on interior fixtures' authored attenuation window = their effective \
         radius, 0..8 (0 = no window, the old flat lane)",
    ),
    // MONKEY (room gate): whether an interior fixture may light only the rooms it CLAIMS — its
    // authored MOLR groups unioned with the interior groups whose MOGI bounding box it stands
    // inside (`LightLitRooms` carries the corpus evidence for why MOLR alone is far too sparse:
    // the Goldshire inn authors one on 2 of its 12 groups). Off restores the pre-gate behaviour:
    // every interior fixture in range lights every interior surface in range, so an inn's
    // ground-floor candles light its basement through the floor. Kept as a dial because a room the
    // gate leaves on ambient alone looks the same as a bug, and this tells the two apart in one
    // keystroke.
    ours(
        "interiorRoomGate",
        "1",
        "benilla's own: an interior fixture lights only the WMO groups it claims (0 = the old \
         leak-through-walls behaviour)",
    ),
    ours(
        "interiorShadows",
        "1",
        "benilla's own: interior fixtures cast real shadows (Stage B, the nearest few); needs \
         interiorLight",
    ),
    // MONKEY (outdoor torch shadows): the outdoor half of the same cube-map lane. Its own row
    // because it is its own audience (a night camp, a lit village) and its own cost profile — and
    // because "turn the outdoor shadows off" must not also turn the inn's candles' shadows off.
    ours(
        "exteriorShadows",
        "1",
        "benilla's own: outdoor fire lights (campfires, braziers, lampposts) cast real shadows at \
         night; no effect by day",
    ),
    // MONKEY (daylight: terrain torch casters): the ground as a torch caster.
    ours(
        "torchTerrainShadows",
        "0",
        "benilla's own: the ground casts into outdoor fire shadows (hills and banks block a fire's \
         light); needs exteriorShadows",
    ),
    // MONKEY (static torch cache): residency and per-frame work have separate live budgets.
    ours(
        "interiorShadowCasters",
        "12",
        "benilla's own: resident interior fixture shadow maps, 1..16",
    ),
    ours(
        "interiorShadowDynamic",
        "4",
        "benilla's own: nearest promoted fixtures with moving entity shadows, 0..16",
    ),
    // MONKEY (torch lane perf): the moving-caster REGATHER cadence. Its own row (and not folded
    // into `interiorShadowDynamic`) because it trades a different currency: `Dynamic` buys how
    // MANY fixtures overlay moving casters, this buys how OFTEN the one shared overlay mesh is
    // rebuilt. Neither of the count dials moved the frame time at all, so the cost was never per
    // map -- it was this gather + mesh mutation, paid once a frame no matter what the counts said.
    // `0` is the pre-feature every-frame behaviour, kept as the live A/B.
    ours(
        "interiorShadowEntityRate",
        "30",
        "benilla's own: how often (Hz) moving torch-shadow casters are regathered; 0 = every frame",
    ),
    // MONKEY (torch caster selection): the PCF tap radius on the torch maps. Candle clusters read
    // very hard-edged at 1 (a half-texel box on a 512² face); 2 is a visible softening for four
    // extra texel-neighbourhood taps' worth of cache pressure, no extra samples.
    ours(
        "interiorShadowSoft",
        "1.5",
        "benilla's own: torch-shadow edge softness — the PCF tap radius scale at a CONTACT, 0.5..3",
    ),
    // MONKEY (shadow floor): how BLACK a torch shadow is allowed to get. The lane's shadows were
    // the only occlusion in the direct term and took all of it, which is what made them read as
    // scars rather than as shadows; 0.7 leaves 30 % standing in place of the bounce light this
    // renderer does not have. `1` is the shipped look, `0` is off.
    ours(
        "torchShadowStrength",
        "0.7",
        "benilla's own: torch-shadow darkness — how much of the direct term a shadow removes, 0..1",
    ),
    ours(
        "interiorDebug",
        "0",
        "benilla's own: interior diagnostic overlay — 1 classification, 2 shadow, 3 caster count, 4 WMO lane map",
    ),
    // MONKEY (darkness gains): the two live dim dials. `nightGain` scales the EXTERIOR day/night
    // law (the packed ambient/diffuse/specular rows) by `mix(1, gain, night_w)`, so it is exactly
    // inert by day and full strength after dark; `interiorGain` scales the room lane's inputs (base
    // ambient, per-fixture fill, and every interior fixture's colour). Both fold in at PACK time in
    // `build_light_data`, so `SetCVar` moves the whole world on the very next frame — which is how
    // "20 % / 30 % darker" gets judged at all, and `1` on either is the restore.
    //
    // Night gain leaves fire lights alone; interior gain also scales interior fixtures,
    // including their candlelight. Outdoor fires retain their brightness under both dials.
    ours(
        "nightGain",
        "0.45",
        "benilla's own: exterior night brightness, 0.2..1.5 (1 = the reference's own night)",
    ),
    // MONKEY (lighting debug panel): weaker fresh interiors; persisted gains still win at boot.
    ours(
        "interiorGain",
        "0.5",
        "benilla's own: WMO interior brightness, 0.2..1.5 (1 = the pre-dial fixture-lit room)",
    ),
    // MONKEY (enclosed day floor): the daylight a room INSIDE A BUILDING gets by day, for the
    // doorways this renderer cannot locate in the data (the Goldshire inn's entry group authors no
    // portal, no EXT-class batch, no stitched vertex and no bake hot spot — there is nowhere to
    // stand a fixture). An additive ambient in `interiorAmbient`'s own units, scaled by the sun's
    // day envelope, so it is exactly 0 at night and the night look never moves. `0` is the restore.
    ours(
        "interiorDaylight",
        "0.0",
        "benilla's own: daylight floor for rooms inside a building, 0..1 (0 = none, the old look)",
    ),
    // MONKEY (fix-daylight): the district window split, gateable (default on = merged behaviour).
    ours(
        "daylightWindowSplit",
        "1",
        "benilla's own: split city window batches into window-sized daylight apertures (applies to newly loaded buildings)",
    ),
    // MONKEY (bake floor): the share of an interior batch's own MOCV bake that survives the live-
    // fixture lane. The lane throws the bake away and lets the fixtures decide, which leaves a room
    // no fixture reaches (the Lion's Pride Inn's east vestibule: MOLR 0, no claims, one faded
    // portal hop) rendering black between a sky-lit porch and a candle-lit hall — something the
    // reference client cannot do, because it draws every interior batch at its bake regardless of
    // lights. A fraction of the bake, inside the room law's rolloff, so a lit surface barely moves.
    ours(
        "interiorBakeFloor",
        "0.12",
        "benilla's own: share of an interior batch's baked light kept where no fixture reaches, 0..1 (0 = the old look)",
    ),
    // MONKEY (fire GO lights): the gain on lights SYNTHESISED from a model's flame emitter for the
    // ~410 fire props the artists never gave a light block (campfires, wall torches, magic
    // braziers, forges, candles). Live, like the interior knobs — and `0` is the kill switch for
    // the whole invented-light lane, which matters because unlike everything beside it this one is
    // a heuristic over content rather than a byte-verified mechanism.
    ours(
        "fireLightGain",
        "1",
        "benilla's own: brightness of lights synthesised from fire props' flame emitters (0 = off)",
    ),
    // MONKEY (spellLightGain): the same dial for the SPELL lane — a kit's aura glow, a missile's
    // core, an impact flash, a firework's burst. A separate knob from the one above it because the
    // two are separate judgements: that one tunes SCENERY (how bright is the invented campfire),
    // this one tunes COMBAT (how hard does a fight flash the room), and a spell light carries both
    // markers, so one dial over both would mean dimming the world's hearths to calm a fireball.
    // `0` is this lane's kill switch and reaches nothing else.
    ours(
        "spellLightGain",
        "1",
        "benilla's own: brightness of spell, missile and impact lights (0 = off)",
    ),
    // MONKEY (flame flicker): how hard every FLAME breathes — candles fast and shallow, bonfires
    // slow and shallower still (`benilla_world::lighting::FlameKind`). Live like the gain beside
    // it, and `0` restores the steady constants every fire had before the feature, which is the
    // escape hatch this needs precisely because "subtle" is a judgement call and a flicker that
    // reads as a strobe is worse than none.
    ours(
        "fireFlicker",
        "1",
        "benilla's own: how strongly fire lights flicker — 0 steady, 1 default, 2 pronounced",
    ),
    // `gxWindow` (`0x63a889`): "0" on enUS; zhCN registers "1", as it does `gxMaximize`
    // (`0x63a8e0`), and koKR `AutoInteract` (`0x603390`). The knob is
    // [`crate::video::VideoConfig::display`]. Deviation: "0" raises a borderless fullscreen window
    // instead of mode-setting the display, because Wayland and macOS offer no mode-set and X11's
    // leaves the desktop changed after a crash ([`crate::video`]).
    same("gxWindow", "0").latched(),
    // `gxResolution`, a string row parsed by `video::on_cvar`. Here it is only the windowed size:
    // fullscreen is the monitor's own and no mode list is offered.
    deviates(
        "gxResolution",
        "1600x900",
        "640x480",
        "narrowed to the WINDOWED size only — fullscreen is the monitor's own and we expose \
         no mode list, and 640x480 is not a window anyone would ship a client at",
    )
    .latched(),
    // benilla's own: a body pane's doll renders at half the frame rate while the pane is open; the
    // reference draws it in the main pass. The knob is [`crate::portrait::PaneRate`].
    ours(
        "boothHalfRate",
        "1",
        "benilla's own — the reference draws its doll inside the main pass and has no \
         second view to rate-limit",
    ),
    // `gxMultisample` (`0x63a950`, flags 3). The reference formats its default from field 21 of the
    // `VideoHardware.dbc` row `DetectHardware` (`0x641260`) matches, which is 1, no multisampling,
    // on every fallback row a modern GPU reaches. The knob is [`benilla_world::view::MsaaSetting`],
    // read once at the camera's spawn; `$WOW_MSAA` overrides it for the session.
    same("gxMultisample", "1").latched(),
    // `GetCurrentMultisampleFormat 0x48c580` looks up all three of the Video dropdown's values by
    // name, so these must exist. They describe the swapchain's own pair and steer nothing;
    // `SetMultisampleFormat` writes them as `0x48c640` does.
    deviates(
        "gxColorBits",
        "32",
        "16",
        "these describe, they do not steer — the pair is our swapchain's own, and every \
         format `MsaaFormats` publishes carries it",
    )
    .latched(),
    deviates(
        "gxDepthBits",
        "32",
        "16",
        "as `gxColorBits` — the depth half of the same descriptive pair",
    )
    .latched(),
    // `trilinear` and `anisotropic`, over `benilla_assets::TexFilterSetting` (`tex_filter.rs`
    // carries the derivation). A change applies at the next launch, since a sampler is baked into
    // each texture at load; the reference's UI says "enabled upon restart".
    // `$WOW_TRILINEAR`/`$WOW_ANISO` override for the session.
    //
    // `trilinear` registers "0", but `hwDetect` runs `DetectHardware 0x641260` and sets it from the
    // matched `VideoHardware.dbc` row before the first frame; the fallback rows an unlisted GPU
    // reaches are 168/169/170, and 169 and 170 give 1. The reference install's `gx.log` resolves to
    // videoID 170.
    overridden(
        "trilinear",
        "1",
        "0",
        "`hwDetect` sets it from `VideoHardware.dbc` field 9 before the first frame, and \
         that field is 1 on both fallback rows an unlisted modern GPU can reach — measured on the \
         reference's own `Logs/gx.log` (`videoID: 170`)",
    ),
    // Not one of `hwDetect`'s sixteen (`[0x639a60, 0x639b80)` never reads `0xc7f2e4`), so the
    // registered "1", off, stands.
    same("anisotropic", "1"),
    // The four video CVars the stock window's verbs read and write (`benilla_ui` `video_pairs`);
    // benilla has no mechanism behind any of them. `shadowLevel` (`0x6885bc`, "1", flags 1,
    // callback `0x688c10`, record `[0xc7f350]`) is behind `Get/SetTerrainMip` as `1 − level`: the
    // callback refuses above 1 and prints "Shadow mip level changed upon restart.", and the
    // terrain's shadow-map mip (`0x66f877`) reads it at load. `hwDetect` leaves it at 1.
    same("shadowLevel", "1"),
    // `doodadAnim` (`0x6884d8`, "1", flags 1, callback `0x688a10`, record `[0xc7f354]`), behind
    // `Get/SetDoodadAnim`: the callback toggles bit `0x8000` of `[0xc7b2a4]` and calls `0x6953d0`,
    // a bare `ret`, so no animation reads it. `hwDetect` sets 0 or 1 by CPU tier.
    same("doodadAnim", "1"),
    // `texLodBias` (`0x6885e2`, "0.0" at `0x84fad4`, flags 1, callback `0x688c90`, record
    // `[0xc7f2f8]`), behind `Get/SetTexLodBias`: the callback refuses outside [-1.0, 1.0] and its
    // sink `0x672640` is `ret 4`.
    same("texLodBias", "0.0"),
    // `baseMip` (`0x6887aa`, "0", flags 1, callback `0x689090`, record `[0xc7f2f0]`), behind
    // `Get/SetBaseMip` as `1 − level`: the callback refuses outside [0, 1] ("BaseMip must be 0 or
    // 1") and sets the device's first uploaded mip level (`0x589bf0`, read at `0x59f6ce`). benilla
    // uploads every level; the write is kept and has no effect. `hwDetect` leaves it at 0.
    same("baseMip", "0"),
    // `spellEffectLevel` (`0x688900`: name `0x869324`, default `0x843068` "2", flags 1, callback
    // `0x689510`, record `[0xc7f2a4]`), the Spell Detail slider (`OptionsFrame.lua:32`). The
    // callback's two effects here: the particle emission scalar ([`crate::video::on_cvar`]) and
    // the dynamic-object shard rate, read at each emitter's spawn (`crate::entities`). Its third,
    // skipping one M2-scene ground decal below level 2 (`0x672b3a`), has no counterpart: benilla
    // does not draw that projection.
    same("spellEffectLevel", "2"),
    // Weather Intensity (`OptionsFrame.lua:33`), registered "2" (`0x67b806`, flags 0, callback
    // `0x67b870`, name `0x8685ac`). The reader is `benilla_world::weather::WeatherState`, which
    // scales the precipitation spawn rate by `0x67b870`'s table {0.1, 0.33, 0.66, 1.0}; rendering
    // only.
    deviates(
        "weatherDensity",
        "3",
        "2",
        "every precipitation rate in `benilla-world`'s own precipitation module was \
         derived and graded against the reference install's own apitrace captures, and that \
         install runs \
         `SET weatherDensity \"3\"` (K = 1.0) — so 3 is the value a benilla-vs-reference \
         side-by-side is correct at, and the registered 2 would thin every rate to 0.66 against \
         the only client we compare with. The slider is how a player takes it back down",
    ),
    // `gamma` (`0x402d70`: name `0x82e924` "Gamma", default `0x82e92c` "1.0", flags 0, callback
    // `0x4034d0`). The reference uploads `pow(i/255, gamma)` (`0x591680`) through
    // `SetDeviceGammaRamp`, except when windowed (`byte[dev+0x20b]`). Deviation: benilla, which
    // has no exclusive mode, applies the same curve in the composite pass
    // ([`crate::ui_gamma::DisplayGamma`]), because the skipped upload would make a slider that
    // moves no pixel. Spelled "1.000000" because `SetGamma 0x4891f0` formats with `"%f"`, and
    // Restore Defaults must compare equal to the default; [`sync_cvars`] seeds it the same way.
    same("gamma", "1.000000"),
    // `DesktopGamma` (`0x402d4d`: name `0x82e930`, default "0" `0x82e570`, flags 0, callback
    // `0x403500`, record `[0x8826d4]`), the stock window's Use Desktop Gamma box, which it also
    // sets whenever Windowed Mode is ticked (`OptionsFrame.lua:399-408`). On 1 the callback puts
    // the desktop's own ramp back, and `gamma`'s callback (`0x4034d0`) uploads only while this
    // reads 0. No effect here: [`crate::ui_gamma::DisplayGamma`] follows `gamma` alone, because
    // the stock window ticks this box whenever Windowed Mode is (`:399-408`), so honouring it would
    // drop the gamma of every windowed player who opens that window.
    same("DesktopGamma", "0"),
    // ── The stock Video Options window's other boxes (`OptionsFrame.lua:5-22`) ───────────────────
    //
    // `ffx` (`0x6ccfe3`: name `0x86f2ac`, record `[0xce8a04]`), `ffxGlow` (`0x6cc17d`, name
    // `0x86f200`) and `ffxDeath` (`0x6cc6d1`, name `0x86f238`): each "1", flags 1, with the echo stub
    // `0x6cde30` as callback, and each record's integer read every frame. The FFX begin/end pair
    // (`0x6cd890`/`0x6cda70`) runs the active pass only while `ffx` (`0x6cd8a6`), `pixelShaders`
    // (`0x6cd8b6`) and that pass's own enable hold: `CFFXGlow` (`0x6cc5a0`) reads `ffxGlow` at
    // `0x6cc5a8`, `CFFXDeath` (`0x6cded0`) reads `ffxDeath` at `0x6cdf10`. The stock Okay writes
    // `ffx` as Enable All Shaders (`OptionsFrame.lua:202-204`). The knob is
    // [`benilla_world::ffx_glow::FfxSwitches`].
    same("ffx", "1"),
    same("ffxGlow", "1"),
    same("ffxDeath", "1"),
    // `pixelShaders` (`0x688712`: name `0x869560`, "0", flags 1, callback `0x688f40`, record
    // `[0xc7f360]`) and `specular` (`0x6886a0`: name `0x8695d8`, "0", flags 1, callback `0x688e20`,
    // record `[0xc7f2cc]`), Enable All Shaders and Terrain Highlights. Both callbacks act only on a
    // device whose caps field `[0x58a230()+0x98]` is above -1, setting bits 28 and 27 of the render
    // flags `[0xc7b2a4]`, which gate the shader negotiation `0x6941f0` for terrain, water and
    // models; `pixelShaders` also gates the FFX pass (`0x6cd8b6`). No effect here: benilla always
    // draws the shader leg (`liquid.wgsl` records it), and the stock Okay carries this box into
    // `ffx`, which is wired.
    overridden(
        "pixelShaders",
        "1",
        "0",
        "`hwDetect` (`0x639a60`) sets it from `VideoHardware.dbc` field 11 before the first frame, \
         and that field is non-zero on fallback row 170, the one a modern GPU reaches \
         (`Logs/gx.log`: `videoID: 170`)",
    ),
    overridden(
        "specular",
        "1",
        "0",
        "`hwDetect` sets it from the same field 11 as `pixelShaders`, so row 170 gives 1",
    ),
    // `M2UseShaders` (`0x4028e9`, name `0x82e67c`, "1") and `M2UsePixelShaders` (`0x40290c`, name
    // `0x82e644`, "0"), flags 0, no callback: their registering function reads six of its eight
    // `M2*` records at `0x40297f`-`0x4029dd` into the M2 renderer's feature word `[0xceefb0]`
    // (`0x706dd0`), once at startup, hence the stock tooltip's logout requirement. No effect here:
    // benilla's M2 draw has one skinning and lighting path, chosen by no switch.
    same("M2UseShaders", "1"),
    same("M2UsePixelShaders", "0"),
    // `lod` (`0x68848c`: name `0x8697d0`, "1", flags 1, callback `0x688920`, record `[0xc7f314]`),
    // World LOD: bit 2 of `[0xc7b2a4]`, which gates the doodad LOD distance test at `0x6afb11`. No
    // effect here: benilla has no doodad LOD swap, as `lodDist` above has no row.
    same("lod", "1"),
    // `movieSubtitle` (`0x402c18`: name `0x82e9d8`, "0", flags 0, record `[0x882640]`), Cinematic
    // Subtitles: the pre-rendered movies' subtitles, read by the glue's `GetMovieSubtitles`
    // (`0x46dbc0`). No effect here: benilla plays no pre-rendered movie.
    same("movieSubtitle", "0"),
    // `useWeatherShaders` (`0x67b81d`: name `0x868598`, "1", flags 0, no callback, record
    // `[0xc7b588]`), Weather Shaders: read in each precipitation effect's constructor (`0x67480f`,
    // `0x6775a7`, `0x679274`), which clears the effect's shader flag at 0. No effect here: rain's
    // two legs are one draw with other scalars, but snow's fixed-function leg (`0x678960`) is a
    // CPU-built triangle per flake where the shader leg (`0x678610`) is a point sprite, a second
    // draw path `benilla_world::weather` does not have, so the shader leg always runs.
    same("useWeatherShaders", "1"),
    // The latched `gx*` rows the stock window's boxes and refresh dropdown write (flags 3, committed
    // by `RestartGx`), each callback storing one field of the device-format record `0xc4eb98`.
    //
    // `gxMaximize` (`0x63a8e0`: name `0x864edc`, "0", "1" on zhCN like `gxWindow`, callback
    // `0x63b1f0` → `+0x09` `[0xc4eba1]`): the window rebuild `0x58cf10` makes a windowed, maximized
    // window a borderless popup (style `0x90000000`) at the screen's full size (`GetSystemMetrics`
    // 0 and 1). The knob is [`crate::video::VideoConfig::maximize`].
    same("gxMaximize", "0").latched(),
    // `gxTripleBuffer` (`0x63a80d`: name `0x864f88`, "0", callback `0x63afe0` → `+0x18`), whose
    // callback refuses any value but 0 or 1 ("TripleBuffer must be 0 or 1", `0x8650b8`), a range
    // this registry has no way to refuse with. No effect here: wgpu's present modes offer no
    // triple-buffer choice.
    same("gxTripleBuffer", "0").latched(),
    // `gxCursor` (`0x63a906`: name `0x864eb8`, "1", callback `0x63b220` → `+0x05` `[0xc4eb9d]`),
    // Hardware Cursor. No effect here: benilla always draws the OS cursor (`crate::cursor`) and has
    // no software cursor to fall back to.
    same("gxCursor", "1").latched(),
    // `gxFixLag` (`0x63a9c1`: name `0x864e40`, callback `0x63b310` → `+0x06` `[0xc4eb9e]`), Fix
    // Input Lag: its default is `"%d"` of `VideoHardware.dbc` field 20 (`[[0xc4e6e4]+0x50]`) of the
    // row `DetectHardware` matched, 1 on all three fallback rows (168, 169, 170) a modern GPU can
    // reach. No effect here: benilla has no such switch in its frame pacing.
    same("gxFixLag", "1").latched(),
    // `gxRefresh` (`0x63a7e7`: name `0x864fa8`, "75", callback `0x63aed0` → `+0x28`), the refresh
    // dropdown (`OptionsFrame.lua:242-245`, `:300`); the callback accepts only the eleven rates at
    // `0x8645e0` (60 to 200). No effect here: fullscreen is borderless at the monitor's own rate,
    // and `GetRefreshRates` answers the reference's no-rates sentinel, which disables the dropdown.
    same("gxRefresh", "75").latched(),
    // benilla's own: the world renders at `window × this` while the UI stays native. The knob is
    // [`crate::world_backdrop::RenderScale`], clamped to `RENDER_SCALE_RANGE`; at "1" nothing is
    // resampled. `$WOW_RENDER_SCALE` overrides it for the session.
    ours(
        "renderScale",
        "1",
        "benilla's own — the reference has no off-screen buffer to hang a resolution dial \
         on; its nearest equivalent, `gxResolution`, drops the interface with the world",
    ),
    // benilla's own: `/console fpsJournal 1` appends a per-second row of position, frame cost and
    // per-pass GPU time to `benilla-config/Diagnostics/fps-journal.csv`. The knob is
    // [`crate::perf::FpsJournalSetting`].
    ours(
        "fpsJournal",
        "0",
        "benilla's own — 1.12 has no player-side perf log; its nearest thing is the \
         Ctrl+R framerate label, a number with no file behind it",
    ),
    // `lastCharacterIndex` (`0x402d93`, "0" `0x82e570`, category 4, handle `[0x882674]`), help
    // "Last character selected": a 0-based row (the selection cell `[0x83856c]` under `"%d"`), so
    // "0" is the first character. It mirrors [`crate::char_select::Roster::pending_index`].
    same(crate::char_select::CVAR_LAST_CHARACTER, "0"),
];

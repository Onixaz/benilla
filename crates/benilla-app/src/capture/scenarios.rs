//! The capture scenarios: named deterministic viewpoints (camera eye and look in raw WoW coords,
//! pinned game minute, optional UI fixture) and the golden table. Data only.

/// A named deterministic capture viewpoint: where the camera stands and looks, and the game minute.
#[derive(Clone, Copy)]
pub(super) struct Scenario {
    pub(super) name: &'static str,
    /// The `Map.dbc` id the coords belong to, since raw WoW coords repeat on every continent;
    /// `None` leaves the `$WOW_MAP` knob to decide, for the instruments that go anywhere.
    pub(super) map: Option<u32>,
    /// Camera eye, raw WoW coords `(x, y, z)`.
    pub(super) eye: [f32; 3],
    /// Camera look-at target, raw WoW coords.
    pub(super) look: [f32; 3],
    /// Game minute of day (`0..1440`), pinning the time-of-day lighting.
    pub(super) minute: u32,
    /// A UI window opened with canned state before the shot.
    pub(super) ui: Option<UiFixture>,
}

/// A UI window opened with synthetic state that mirrors what the server sends, so the capture
/// runs the real feed, VM, extract and render chain.
#[derive(Clone, Copy, PartialEq)]
pub(super) enum UiFixture {
    /// The player UI with no window opened: the chrome that is there once the UI loads and the
    /// synthetic unit snapshot lands (`ui_script::demo_unit_feed`). A UI capture all the same, so
    /// [`ui_opted_in`] reads `ui.is_some()`.
    Bare,
    Merchant,
    Gossip,
    Quest,
    /// The bank window fed through a synthetic self-player descriptor: `PLAYER_FIELD_BANK_SLOT`
    /// guids, a bank bag, the purchased count in `PLAYER_BYTES_2` byte 2, and coinage.
    Bank,
    /// The multi-quest greeting panel (`QUEST_GREETING`, `QuestGreetingPanel`): a greeting line
    /// over `UI-Quest-BulletPoint` title rows.
    QuestGreeting,
    /// The quest log fed through a synthetic self-player's `PLAYER_QUEST_LOG` slots.
    QuestLog,
    Loot,
    Bag,
    /// The cooldown sweep at sixteen phases in one still: each backpack slot's
    /// `GetContainerItemCooldown` sits at its own fraction of one long cooldown, in reading order
    /// (slot 1 renders top-left, `ContainerFrame_GenerateFrame` numbering backwards). The VM's
    /// `GetTime` clock is parked at a large value, so the settle's seconds are ~5e-4 of a phase.
    Cooldown,
    /// The cooldown filmstrip with the pet bar's autocast shine beside it: `UI-AutoCastButton.m2`
    /// is four additive emitters and no batch, sharing the tile atlas, so this asks whether its
    /// particles stay inside their own cell.
    CooldownShine,
    /// The bag window with the GameTooltip forced open over a known slot.
    Tooltip,
    /// The world-mouseover tooltip over a seeded unit: the default anchor puts it at the screen's
    /// bottom-right (`GameTooltip.lua:73-77`), never on the hovered model.
    TooltipWorld,
    /// The character window fed through a synthetic self player's stat block and equipped item
    /// guids, with the items in [`crate::items::Items`].
    Character,
    /// The world-entry loading screen held up with its `GameTips.dbc` tip: it is otherwise up for a
    /// second on a path nothing can pause.
    LoadingTip,
    /// The shared StaticPopup plate as the group-invite dialog (`PARTY_INVITE`), the one that
    /// floats over the world rather than another window's art.
    PartyInvite,
    /// A V-key nameplate over a synthetic Timber Wolf (entry 69, level 2, faction 32, display 604).
    /// At the forced 1024×768 window one gx unit is 1280 px, so the 0.1 × 0.025 plate lands at
    /// 128×32 logical px, the border texture's native size.
    VPlates,
    /// The stock world map opened at the Elwynn zone map with alternating explore bits, so the
    /// exploration overlays and the parchment both show.
    WorldMap,
    /// The spellbook over a seeded known-spell set resolved through the real chain (`Spell.dbc`,
    /// `SkillLineAbility.dbc`, the book feed).
    SpellBook,
    /// The macro window over a macro set made through the live `CreateMacro` path, second slot
    /// selected.
    Macro,
    /// The macro window's name and icon popup: the icon grid off `SpellIcon.dbc`, the name box.
    MacroPopup,
    /// The chat edit box opened with a typed draft through the live path (`focus_editbox`,
    /// `chat_edit_live`) over say and yell lines.
    ChatEdit,
    /// The chat dock revealed, Combat Log selected, the cursor on the General tab; `$WOW_TABHOVER`
    /// picks an alternate dock state (`fixtures.rs`).
    ChatTabHover,
    /// The social pane (`FriendsFrame`) opened through the live toggle. Its `FriendsDropDown` has
    /// no anchors, as in the reference, and must draw nothing at the screen origin.
    Social,
    /// Our Options window opened through the live panel path, Controls selected by default.
    Options,
    /// The Options window on the Audio page. The options fixtures read the CVar registration
    /// defaults, since a capture loads no config file.
    OptionsAudio,
    /// The Options window on the Graphics page.
    OptionsGraphics,
    // MONKEY (volumetric fog): photograph the new quality row on its actual page.
    OptionsAdvancedGraphics,
    /// The Options window on the Chat page (1.12's `CHAT_LABEL`), whose rows mix a saved variable
    /// and CVars in one column.
    OptionsChat,
    /// The colour picker seeded at a known colour, so the wheel and brightness markers, on art this
    /// client generates, sit somewhere checkable.
    ColorPicker,
    /// The Controls page with the Camera Following Style dropdown open: `DropDownList1` at the
    /// window's effective scale, the stored entry checked.
    OptionsDropdownList,
    /// The Options window mid-search: "volume" lists the four volume sliders under the Audio head.
    OptionsSearch,
    /// The Options window's Keybindings page, Movement expanded, over the stock commands and
    /// default bindings the load read off the install, and GlobalStrings.
    KeyBindings,
    /// An overhead name with the river behind it: a named unit 25 yd out in the Elwynn river (the
    /// `water-noon` camera). Deep water is opaque (`WATER_DEEP_ALPHA` 1.0), so a name sorted before
    /// the liquid is painted out; it must read at full strength.
    NameWater,
    /// One cell of the lighting matrix: a creature or GameObject spawned through the live path at
    /// `at` ([`SubjectKind`], the note above [`SUBJECT_SUN`]).
    Subject {
        kind: SubjectKind,
        /// Where the subject stands, raw WoW coords: its feet, not its body centre.
        at: [f32; 3],
    },
}

/// What the lighting matrix puts in frame: both spawn with a streamed entity's component set,
/// differing only in `EntityKind`.
#[derive(Clone, Copy, PartialEq)]
pub(super) enum SubjectKind {
    /// The Timber Wolf (entry 69, display 604), the `vplates` and `name-water` subject.
    Creature,
    /// `World\SkillActivated\Containers\TreasureChest01.mdx` (`GameObjectDisplayInfo` 259) at its
    /// closed rest pose.
    Chest,
}

/// The on-demand Northshire framings at the Human start (`SPAWN_XY` (-8949.95, -132.49), ground
/// ≈ 83.5): a ground overlook down at terrain and the Abbey, and a sky view up at the dome.
pub(super) const GROUND_EYE: [f32; 3] = [-8980.0, -160.0, 110.0];
pub(super) const GROUND_LOOK: [f32; 3] = [-8949.95, -132.49, 84.0];
pub(super) const SKY_EYE: [f32; 3] = [-8980.0, -160.0, 112.0];
/// MONKEY (sky): 80 yd over the Northshire sky eye, above every canopy.
pub(super) const SKY_HIGH_EYE: [f32; 3] = [-8980.0, -160.0, 190.0];
pub(super) const SKY_LOOK: [f32; 3] = [-8740.0, 80.0, 168.0]; // horizon in the lower third

// Farmhouse compass looks: an ordinary building shows defects the Abbey is immune to.
pub(super) const HOUSE_EYE: [f32; 3] = [-9439.1, 71.2, 68.0];

/// `Map.dbc` ids the golden spots stand on.
pub(super) const MAP_AZEROTH: u32 = 0;
pub(super) const MAP_KALIMDOR: u32 = 1;
pub(super) const MAP_DEEPRUN_TRAM: u32 = 369;

/// A glue-screen capture: a login-side screen with no world, camera or map, sharing only the
/// shutter with [`Scenario`]. Not in the golden sweep; capturable by name.
#[derive(Clone, Copy)]
pub(super) struct GlueScenario {
    pub(super) name: &'static str,
    pub(super) screen: GlueScreen,
    /// The preview's race, sex and class ids, applied through `WOW_CHARCREATE_PICK` unless that is
    /// already set (`WOW_CHARCREATE_PICK=6,0,11 WOW_CAPTURE=glue-charcreate` is the tauren stage).
    pub(super) pick: Option<(u8, u8, u8)>,
}

/// Which glue screen a [`GlueScenario`] photographs. The select screen needs a roster and is not
/// here.
#[derive(Clone, Copy, PartialEq)]
pub(crate) enum GlueScreen {
    /// Character creation: the `UI_*` backdrop scene's rig lighting and per-race fog, the preview
    /// body in its starting outfit, the GlueXML panel.
    CharCreate,
    /// The login screen, `UI_MainMenu` behind the account form. Its backdrop is the narrowest of
    /// the seven scenes, so a wide window (`WOW_WIN=2560x1440`) reaches its edges first.
    Login,
    /// The realm list over the login screen, fed a list in two categories
    /// (`super::realm_list`).
    RealmList,
}

/// The glue scenarios: character creation as a human male warrior, the race the reference's own
/// screenshots use, the login screen, and the realm list's category tabs over it.
pub(super) const GLUE_SCENARIOS: &[GlueScenario] = &[
    GlueScenario {
        name: "glue-charcreate",
        screen: GlueScreen::CharCreate,
        pick: Some((1, 0, 1)),
    },
    GlueScenario {
        name: "glue-login",
        screen: GlueScreen::Login,
        pick: None,
    },
    GlueScenario {
        name: super::realm_list::NAME,
        screen: GlueScreen::RealmList,
        pick: None,
    },
];

/// The golden baseline: two spots at noon and at night, Elwynn water and a Felwood hollow on
/// Kalimdor, their coordinates recorded with `/shot` (`benilla-config/shots.txt`). Held small on
/// purpose: a viewpoint being worked on belongs in [`ON_DEMAND`], capturable by name.
pub(super) const SCENARIOS: &[Scenario] = &[
    Scenario {
        name: "water-noon",
        map: Some(MAP_AZEROTH),
        eye: WATER_EYE,
        look: WATER_LOOK,
        minute: 720,
        ui: None,
    },
    Scenario {
        name: "water-night",
        map: Some(MAP_AZEROTH),
        eye: WATER_EYE,
        look: WATER_LOOK,
        minute: 0,
        ui: None,
    },
    Scenario {
        name: "felwood-noon",
        map: Some(MAP_KALIMDOR),
        eye: FELWOOD_EYE,
        look: FELWOOD_LOOK,
        minute: 720,
        ui: None,
    },
    Scenario {
        name: "felwood-night",
        map: Some(MAP_KALIMDOR),
        eye: FELWOOD_EYE,
        look: FELWOOD_LOOK,
        minute: 0,
        ui: None,
    },
];

/// The Northshire overlook, north-east of the Abbey looking at it: terrain, trees, the Abbey WMO,
/// stained glass, props.
pub(super) const OVERLOOK_EYE: [f32; 3] = [-8955.0, -98.5, 91.1];
pub(super) const OVERLOOK_LOOK: [f32; 3] = [-8912.9, -125.4, 87.7];

/// The Elwynn river south-east of Northshire: open water, the shoreline blend, a murloc camp, fog.
pub(super) const WATER_EYE: [f32; 3] = [-9527.0, -310.6, 70.8];
pub(super) const WATER_LOOK: [f32; 3] = [-9499.4, -351.3, 61.4];

/// The Lion's Pride Inn common room at Goldshire: the hearth's MOCV-alpha self-illumination, a
/// daylight window, the chandelier, props; it exercises portal culling, the interior bake, MOLT
/// point lights and WMO props.
///
/// The camera must stand over a floor face: over a floorless pocket the portal cull's down-ray
/// reads outside and culls the containing group.
pub(super) const INN_EYE: [f32; 3] = [-9471.4, 39.4, 59.9];
pub(super) const INN_LOOK: [f32; 3] = [-9458.8, -7.5, 48.2];

/// An Elwynn rail fence across the sun's shadow boundary at 10:24: one span in sun, one in shade,
/// so both states of the MCSH sun term share one frame and only their difference carries the shot.
/// The fence is a doodad, whose shade is baked per vertex rather than ramped by `entity_shade`.
pub(super) const FENCE_EYE: [f32; 3] = [-9511.9, -4.0, 61.9];
pub(super) const FENCE_LOOK: [f32; 3] = [-9552.0, 18.6, 42.4];

/// A Felwood hollow on Kalimdor: the root mat, emissive `felwoodmushroom` doodads, a sludge pool
/// (a liquid type no other golden shot has), the zone's green fog and light. Its ADT tile
/// (`33_24`) exists in Azeroth too, empty.
pub(super) const FELWOOD_EYE: [f32; 3] = [4060.9, -944.3, 256.8];
pub(super) const FELWOOD_LOOK: [f32; 3] = [4014.0, -954.4, 242.9];

// ---------------------------------------------------------------------------------------------
// The lighting matrix: one subject, three lanes, two sides.
//
// The golden spots hold no creature or GameObject, so these cells cover the object light path.
// The three positions are the three lanes an object's light can take, found with
// `WOW_LIGHT_AT`/`WOW_LIGHT_GRID` (`wmo_portal::audit::light_probe`):
//
//   SUN    (-9500, 56)      terrain z 56.48  MCSH false  exterior-on-terrain, sun term at full
//   SHADE  (-9500, 44)      terrain z 55.95  MCSH true   exterior-on-terrain, sun term dimmed
//   INDOOR (-9469.4, 31.9)  no terrain       zone-text indoor 5, interior bake group 05, no sun
//
// SUN and SHADE differ only in the baked shadow bit `entity_shade` ramps on (2.5 lit, 0.5
// shadowed). The lighting sun sits near azimuth 45° (`sun::follow`): `front` puts the camera on
// that bearing, facing the lit side, `rear` opposite. Every cell frames its subject identically,
// so a diff is about light.

/// Lighting-matrix subject positions, the feet, raw WoW coords.
pub(super) const SUBJECT_SUN: [f32; 3] = [-9500.0, 56.0, 56.48];
pub(super) const SUBJECT_SHADE: [f32; 3] = [-9500.0, 44.0, 55.95];
pub(super) const SUBJECT_INDOOR: [f32; 3] = [-9469.4, 31.9, 57.9];

/// Every other named viewpoint, capturable by name (`WOW_CAPTURE=<name>`) but not in the sweep.
pub(super) const ON_DEMAND: &[Scenario] = &[
    // MONKEY (volumetric fog): UI proof uses the same live page selection as a player.
    Scenario {
        name: "ui-options-advanced",
        map: Some(MAP_AZEROTH),
        eye: GROUND_EYE,
        look: GROUND_LOOK,
        minute: 720,
        ui: Some(UiFixture::OptionsAdvancedGraphics),
    },
    // MONKEY (volumetric fog): reproducible dawn trees, moonlit lamps and the inn's common room.
    Scenario {
        name: "volfog-dawn",
        map: Some(MAP_AZEROTH),
        eye: [-9460.0, 70.0, 58.0],
        look: [-9410.0, 120.0, 67.0],
        minute: 390,
        ui: None,
    },
    Scenario {
        name: "volfog-night",
        map: Some(MAP_AZEROTH),
        eye: [-9460.0, 70.0, 58.0],
        look: [-9450.0, 20.0, 61.0],
        minute: 0,
        ui: None,
    },
    // MONKEY (volumetric fog): celestial_sun_direction(420) has WoW azimuth
    // 45 degrees and elevation 12.20778 degrees: dz = hypot(70,70)*tan(elevation).
    // The Goldshire lake bank has sunlit air behind the gaps between the trees.
    Scenario {
        name: "volfog-sun",
        map: Some(MAP_AZEROTH),
        eye: WATER_EYE,
        look: [-9457.0, -240.6, 92.21753],
        minute: 420,
        ui: None,
    },
    Scenario {
        name: "volfog-inn",
        map: Some(MAP_AZEROTH),
        eye: INN_EYE,
        look: INN_LOOK,
        minute: 390,
        ui: None,
    },
    // WOW_CAPTURE_WATER_T=<seconds>: fixed Enhanced water phase (read once, default 0).
    // Compare 0 and 1.5; Classic and non-water animation stay frozen.
    // Above Lakeshire's broken docks, looking across Lake Everstill and its rocky shores.
    Scenario {
        name: "water-lake",
        map: Some(MAP_AZEROTH),
        eye: [-9350.0, -2340.0, 82.0],
        look: [-9367.0, -2436.0, 57.1],
        minute: 720,
        ui: None,
    },
    // Westfall western coast: sand in the foreground, open sea to the west.
    Scenario {
        name: "water-ocean",
        map: Some(MAP_AZEROTH),
        eye: [-10500.0, 2112.0, 6.0],
        look: [-10420.0, 2192.0, 0.0],
        minute: 720,
        ui: None,
    },
    Scenario {
        name: "water-ocean-dusk",
        map: Some(MAP_AZEROTH),
        eye: [-10500.0, 2112.0, 6.0],
        look: [-10420.0, 2192.0, 0.0],
        minute: 1110,
        ui: None,
    },
    // Owner's Westfall shore, facing 0.86 rad with the white moon above the sea.
    Scenario {
        name: "water-ocean-moon",
        map: Some(MAP_AZEROTH),
        eye: [-9754.7, 1714.9, 3.0],
        look: [-9689.45, 1790.68, 7.0],
        minute: 170,
        ui: None,
    },
    // Elevated oblique beach view: compare phases 0, 2 and 4 seconds for swash run-up.
    Scenario {
        name: "water-beach-top",
        map: Some(MAP_AZEROTH),
        eye: [-9754.7, 1714.9, 30.7],
        look: [-9739.8, 1700.0, 0.6],
        minute: 400,
        ui: None,
    },
    // Elwynn river from flight height; both banks and the river bed at dusk.
    Scenario {
        name: "water-river-dusk",
        map: Some(MAP_AZEROTH),
        eye: [-9500.0, -390.0, 97.7],
        look: [-9500.0, -433.26, 61.4],
        minute: 1140,
        ui: None,
    },
    // The same river from well above: several chunks and a bend in one frame (flow continuity).
    Scenario {
        name: "water-river-high",
        map: Some(MAP_AZEROTH),
        eye: [-9440.0, -380.0, 150.0],
        look: [-9500.0, -433.26, 57.6],
        minute: 1080,
        ui: None,
    },
    // Standing in the shallows at the bank: bed clutter seen through a foot of water.
    Scenario {
        name: "water-river-bank",
        map: Some(MAP_AZEROTH),
        eye: [-9500.0, -418.0, 60.2],
        look: [-9500.0, -428.0, 57.2],
        minute: 720,
        ui: None,
    },
    // The sunken rowing boat off Longshore: an OBJECT in shallow scene depth over a deeper bed -
    // the surf must not be painted on its hull (owner screenshot).
    Scenario {
        name: "water-wreck",
        map: Some(MAP_AZEROTH),
        eye: [-9606.0, 1257.0, 9.0],
        look: [-9588.0, 1256.0, 0.0],
        minute: 480,
        ui: None,
    },
    // Longshore shallows at noon, looking down through 2-4 yd of sea: refraction and the caustic
    // web on the sand (the morning scenes are too low-sun to focus).
    Scenario {
        name: "water-caustics",
        map: Some(MAP_AZEROTH),
        eye: [-9612.0, 1262.0, 7.0],
        look: [-9598.0, 1262.0, -3.5],
        minute: 720,
        ui: None,
    },
    // Canal water, wall, hull and dock-post contacts in one near view.
    Scenario {
        name: "stormwind-canal-near",
        map: Some(MAP_AZEROTH),
        eye: [-8850.0, 760.0, 102.0],
        look: [-8779.29, 830.71, 78.91],
        minute: 720,
        ui: None,
    },
    // Sunset companion to the near canal contact probe, matching the rejected water reference.
    Scenario {
        name: "water-canal-dusk",
        map: Some(MAP_AZEROTH),
        eye: [-8850.0, 760.0, 102.0],
        look: [-8779.29, 830.71, 78.91],
        minute: 1110,
        ui: None,
    },
    // Enhanced-water contact probe: lower, closer view across the Elwynn river bank.
    Scenario {
        name: "water-shore",
        map: Some(MAP_AZEROTH),
        eye: [-9520.0, -321.0, 65.5],
        look: [-9501.0, -350.0, 61.4],
        minute: 720,
        ui: None,
    },
    // ---- The Deeprun Tram's undersea tube (map 369) ----
    // The one shipped map with no `Light.dbc` row, not even the falloff-0 global maps 0 and 1
    // carry, so its atmosphere is the WMO's own MFOG (record 2: RGB(30,53,100), end 236.1 yd,
    // start scalar 0.05); and a global (WDT `MODF`) WMO.
    //
    // Do not baseline it: server-less, no group of the WMO reaches the frame, so the shot is the
    // bare sky dome from any eye, though the live client draws the map. Kept as the reproducer.
    Scenario {
        name: "tram-undersea",
        map: Some(MAP_DEEPRUN_TRAM),
        eye: TRAM_EYE,
        look: TRAM_LOOK,
        minute: 720,
        ui: None,
    },
    // ---- The Tainted Scar, Blasted Lands ----
    // Looking up across the crater, mostly sky. The zone's `LightParams` 36 has cloud density
    // 0.85 (Elwynn 0.50), so the cloud layer's colour is the sky: the cloud-palette reproducer.
    Scenario {
        name: "tainted-scar-noon",
        map: Some(MAP_AZEROTH),
        eye: SCAR_EYE,
        look: SCAR_LOOK,
        minute: 720,
        ui: None,
    },
    // ---- Out of the baseline sweep ----
    // `chest-shade-{front,rear}` are not reproducible (two runs of one build: MAE 2.721 / 2.649,
    // ~7.8 % of pixels).
    Scenario {
        name: "overlook-noon",
        map: Some(MAP_AZEROTH),
        eye: OVERLOOK_EYE,
        look: OVERLOOK_LOOK,
        minute: 720,
        ui: None,
    },
    Scenario {
        name: "overlook-night",
        map: Some(MAP_AZEROTH),
        eye: OVERLOOK_EYE,
        look: OVERLOOK_LOOK,
        minute: 0,
        ui: None,
    },
    // No `inn-night`: hearth and candles light the room, so the clock barely reaches it (MAE
    // 0.198 against `inn-noon`).
    Scenario {
        name: "inn-noon",
        map: Some(MAP_AZEROTH),
        eye: INN_EYE,
        look: INN_LOOK,
        minute: 720,
        ui: None,
    },
    // At 10:24, not noon: the shadow boundary is the subject, and at noon it slides off the rails.
    Scenario {
        name: "fence-shadowline-day",
        map: Some(MAP_AZEROTH),
        eye: FENCE_EYE,
        look: FENCE_LOOK,
        minute: 624,
        ui: None,
    },
    // The shadow bit is baked and present at every hour; the night frame pins how much of the shade
    // term survives once the ambient and moon palette carry the frame.
    Scenario {
        name: "fence-shadowline-night",
        map: Some(MAP_AZEROTH),
        eye: FENCE_EYE,
        look: FENCE_LOOK,
        minute: 0,
        ui: None,
    },
    Scenario {
        name: "creature-sun-front",
        map: Some(MAP_AZEROTH),
        eye: [-9496.46, 59.54, 58.28],
        look: [-9500.00, 56.00, 57.28],
        minute: 720,
        ui: Some(UiFixture::Subject {
            kind: SubjectKind::Creature,
            at: SUBJECT_SUN,
        }),
    },
    Scenario {
        name: "creature-sun-rear",
        map: Some(MAP_AZEROTH),
        eye: [-9503.54, 52.46, 58.28],
        look: [-9500.00, 56.00, 57.28],
        minute: 720,
        ui: Some(UiFixture::Subject {
            kind: SubjectKind::Creature,
            at: SUBJECT_SUN,
        }),
    },
    Scenario {
        name: "creature-shade-front",
        map: Some(MAP_AZEROTH),
        eye: [-9496.46, 47.54, 57.75],
        look: [-9500.00, 44.00, 56.75],
        minute: 720,
        ui: Some(UiFixture::Subject {
            kind: SubjectKind::Creature,
            at: SUBJECT_SHADE,
        }),
    },
    Scenario {
        name: "creature-shade-rear",
        map: Some(MAP_AZEROTH),
        eye: [-9503.54, 40.46, 57.75],
        look: [-9500.00, 44.00, 56.75],
        minute: 720,
        ui: Some(UiFixture::Subject {
            kind: SubjectKind::Creature,
            at: SUBJECT_SHADE,
        }),
    },
    Scenario {
        name: "creature-indoor-front",
        map: Some(MAP_AZEROTH),
        eye: [-9466.57, 34.73, 59.70],
        look: [-9469.40, 31.90, 58.70],
        minute: 720,
        ui: Some(UiFixture::Subject {
            kind: SubjectKind::Creature,
            at: SUBJECT_INDOOR,
        }),
    },
    Scenario {
        name: "creature-indoor-rear",
        map: Some(MAP_AZEROTH),
        eye: [-9472.23, 29.07, 59.70],
        look: [-9469.40, 31.90, 58.70],
        minute: 720,
        ui: Some(UiFixture::Subject {
            kind: SubjectKind::Creature,
            at: SUBJECT_INDOOR,
        }),
    },
    Scenario {
        name: "chest-sun-front",
        map: Some(MAP_AZEROTH),
        eye: [-9496.82, 59.18, 57.98],
        look: [-9500.00, 56.00, 56.93],
        minute: 720,
        ui: Some(UiFixture::Subject {
            kind: SubjectKind::Chest,
            at: SUBJECT_SUN,
        }),
    },
    Scenario {
        name: "chest-sun-rear",
        map: Some(MAP_AZEROTH),
        eye: [-9503.18, 52.82, 57.98],
        look: [-9500.00, 56.00, 56.93],
        minute: 720,
        ui: Some(UiFixture::Subject {
            kind: SubjectKind::Chest,
            at: SUBJECT_SUN,
        }),
    },
    Scenario {
        name: "chest-shade-front",
        map: Some(MAP_AZEROTH),
        eye: [-9496.82, 47.18, 57.45],
        look: [-9500.00, 44.00, 56.40],
        minute: 720,
        ui: Some(UiFixture::Subject {
            kind: SubjectKind::Chest,
            at: SUBJECT_SHADE,
        }),
    },
    Scenario {
        name: "chest-shade-rear",
        map: Some(MAP_AZEROTH),
        eye: [-9503.18, 40.82, 57.45],
        look: [-9500.00, 44.00, 56.40],
        minute: 720,
        ui: Some(UiFixture::Subject {
            kind: SubjectKind::Chest,
            at: SUBJECT_SHADE,
        }),
    },
    // Not reproducible (two runs of one build: front MAE 1.551 / 9.2 % of pixels, rear 0.529 /
    // 7.6 %): the whole body shifts brightness in registration, so the GameObject's interior light
    // lane does not converge by the shutter, while the creature at the same spot is bit-identical.
    Scenario {
        name: "chest-indoor-front",
        map: Some(MAP_AZEROTH),
        eye: [-9466.85, 34.45, 59.40],
        look: [-9469.40, 31.90, 58.35],
        minute: 720,
        ui: Some(UiFixture::Subject {
            kind: SubjectKind::Chest,
            at: SUBJECT_INDOOR,
        }),
    },
    Scenario {
        name: "chest-indoor-rear",
        map: Some(MAP_AZEROTH),
        eye: [-9471.95, 29.35, 59.40],
        look: [-9469.40, 31.90, 58.35],
        minute: 720,
        ui: Some(UiFixture::Subject {
            kind: SubjectKind::Chest,
            at: SUBJECT_INDOOR,
        }),
    },
    Scenario {
        name: "house-north",
        map: Some(MAP_AZEROTH),
        eye: HOUSE_EYE,
        look: [-9389.1, 71.2, 58.0],
        minute: 720,
        ui: None,
    },
    Scenario {
        name: "house-south",
        map: Some(MAP_AZEROTH),
        eye: HOUSE_EYE,
        look: [-9489.1, 71.2, 58.0],
        minute: 720,
        ui: None,
    },
    Scenario {
        name: "house-west",
        map: Some(MAP_AZEROTH),
        eye: HOUSE_EYE,
        look: [-9439.1, 121.2, 58.0],
        minute: 720,
        ui: None,
    },
    Scenario {
        name: "house-east",
        map: Some(MAP_AZEROTH),
        eye: HOUSE_EYE,
        look: [-9439.1, 21.2, 58.0],
        minute: 720,
        ui: None,
    },
    // At midnight the SIDN night fraction (`grade.x`) is 1.0, so the MOMT 0x10 window glow is live;
    // every other WMO scenario sits before the 20:30 ramp.
    Scenario {
        name: "house-north-midnight",
        map: Some(MAP_AZEROTH),
        eye: HOUSE_EYE,
        look: [-9389.1, 71.2, 58.0],
        minute: 0,
        ui: None,
    },
    // The inn kitchen: its hearth carries the building's strongest MOCV-alpha bake, α≈100 at the
    // firebox (group-local (-32.9, 1.5, 2), world ≈ (-9461.7, -8.4, 58) per MODF uid 71414 on tile
    // 31,49, origin (-9464.25, 24.39, 56.53), rot -97°). Over a floor face, as for `INN_EYE`;
    // the reference culls a floorless pocket's group the same way (outside leg `0x6811ca`).
    Scenario {
        name: "inn-interior",
        map: Some(MAP_AZEROTH),
        eye: [-9463.3, 4.4, 58.8],
        look: [-9462.1, -5.6, 58.5],
        minute: 720,
        ui: None,
    },
    Scenario {
        name: "northshire-dusk",
        map: Some(MAP_AZEROTH),
        eye: GROUND_EYE,
        look: GROUND_LOOK,
        minute: 1170, // 19:30, warm dusk light and fog
        ui: None,
    },
    Scenario {
        name: "northshire-sky-noon",
        map: Some(MAP_AZEROTH),
        eye: SKY_EYE,
        look: SKY_LOOK,
        minute: 720, // day sky-dome gradient + fog horizon
        ui: None,
    },
    Scenario {
        name: "northshire-sky-dusk",
        map: Some(MAP_AZEROTH),
        eye: SKY_EYE,
        look: SKY_LOOK,
        minute: 1170, // dusk dome warp + low sun + stars emerging
        ui: None,
    },
    // Straight into the sun at 17:30 (elevation ≈30°, azimuth 45°, clear of the Northshire ridge)
    // with the view lerp at max: the 20-unit sunGlare quad must fade off with no hard edge.
    Scenario {
        name: "northshire-sun-flare",
        map: Some(MAP_AZEROTH),
        eye: SKY_EYE,
        look: [-8797.0, 23.0, 264.0], // eye + 300·(elev 30°, az 45°), the sun at 17:30
        minute: 1050,
        ui: None,
    },
    // The moon rising at 22:44 (azimuth 45°, elevation ≈15°): the disc rises edge-first behind the
    // ridge (per-pixel terrain occlusion), and no glare ring shows, the moon dnCurve being zero
    // until 22:45.
    Scenario {
        name: "northshire-moonrise",
        map: Some(MAP_AZEROTH),
        eye: SKY_EYE,
        look: [-8775.0, 45.0, 190.0], // eye + 300·(elev 15°, az 45°), the moon at 22:44
        minute: 1364,
        ui: None,
    },
    // Midnight, moon overhead (azimuth 45°, elevation 55°), dnCurve 1.0, star curve 1.0: the disc
    // and its glare ring at full strength over the star field.
    Scenario {
        name: "northshire-moon-halo",
        map: Some(MAP_AZEROTH),
        eye: SKY_EYE,
        look: [-8858.0, -38.0, 358.0], // eye + 300·(elev 55°, az 45°), the moon at 00:00
        minute: 0,
        ui: None,
    },
    // The `.tele Stormwind` spot (vmangos `game_tele`: -8833.38, 628.63, 94.01, o=1.065) at head
    // height into the Trade District: the city-scale perf scene, the whole city WMO resident.
    Scenario {
        name: "stormwind",
        map: Some(MAP_AZEROTH),
        eye: [-8833.38, 628.63, 96.0],
        look: [-8809.1, 672.3, 94.0],
        minute: 720,
        ui: None,
    },
    // The UI window fixtures: each a shipped window with canned state over the noon ground view.
    Scenario {
        name: "ui-merchant",
        map: Some(MAP_AZEROTH),
        eye: GROUND_EYE,
        look: GROUND_LOOK,
        minute: 720,
        ui: Some(UiFixture::Merchant),
    },
    Scenario {
        name: "ui-gossip",
        map: Some(MAP_AZEROTH),
        eye: GROUND_EYE,
        look: GROUND_LOOK,
        minute: 720,
        ui: Some(UiFixture::Gossip),
    },
    Scenario {
        name: "ui-bank",
        map: Some(MAP_AZEROTH),
        eye: GROUND_EYE,
        look: GROUND_LOOK,
        minute: 720,
        ui: Some(UiFixture::Bank),
    },
    Scenario {
        name: "ui-quest",
        map: Some(MAP_AZEROTH),
        eye: GROUND_EYE,
        look: GROUND_LOOK,
        minute: 720,
        ui: Some(UiFixture::Quest),
    },
    Scenario {
        name: "ui-questgreeting",
        map: Some(MAP_AZEROTH),
        eye: GROUND_EYE,
        look: GROUND_LOOK,
        minute: 720,
        ui: Some(UiFixture::QuestGreeting),
    },
    Scenario {
        name: "ui-questlog",
        map: Some(MAP_AZEROTH),
        eye: GROUND_EYE,
        look: GROUND_LOOK,
        minute: 720,
        ui: Some(UiFixture::QuestLog),
    },
    Scenario {
        name: "ui-loot",
        map: Some(MAP_AZEROTH),
        eye: GROUND_EYE,
        look: GROUND_LOOK,
        minute: 720,
        ui: Some(UiFixture::Loot),
    },
    Scenario {
        name: "ui-bag",
        map: Some(MAP_AZEROTH),
        eye: GROUND_EYE,
        look: GROUND_LOOK,
        minute: 720,
        ui: Some(UiFixture::Bag),
    },
    // Sixteen bag slots, sixteen cooldown phases, one still.
    Scenario {
        name: "ui-cooldown",
        map: Some(MAP_AZEROTH),
        eye: GROUND_EYE,
        look: GROUND_LOOK,
        minute: 720,
        ui: Some(UiFixture::Cooldown),
    },
    // The filmstrip with the autocast shine sharing the atlas: the cell-bleed instrument.
    Scenario {
        name: "ui-cooldown-shine",
        map: Some(MAP_AZEROTH),
        eye: GROUND_EYE,
        look: GROUND_LOOK,
        minute: 720,
        ui: Some(UiFixture::CooldownShine),
    },
    Scenario {
        name: "ui-tooltip",
        map: Some(MAP_AZEROTH),
        eye: GROUND_EYE,
        look: GROUND_LOOK,
        minute: 720,
        ui: Some(UiFixture::Tooltip),
    },
    // The default anchor, screen bottom-right (−13/+70), over a seeded hostile wolf.
    Scenario {
        name: "ui-tooltip-world",
        map: Some(MAP_AZEROTH),
        eye: GROUND_EYE,
        look: GROUND_LOOK,
        minute: 720,
        ui: Some(UiFixture::TooltipWorld),
    },
    Scenario {
        name: "ui-char",
        map: Some(MAP_AZEROTH),
        eye: GROUND_EYE,
        look: GROUND_LOOK,
        minute: 720,
        ui: Some(UiFixture::Character),
    },
    // Player and target frames from `demo_unit_feed`'s synthetic snapshots.
    Scenario {
        name: "ui-unitframes",
        map: Some(MAP_AZEROTH),
        eye: GROUND_EYE,
        look: GROUND_LOOK,
        minute: 720,
        ui: Some(UiFixture::Bare),
    },
    // `demo_unit_feed` seeds a rogue with four points on the wolf for this scenario only; the demo
    // player is otherwise a warrior, which lights no dot.
    Scenario {
        name: "ui-combopoints",
        map: Some(MAP_AZEROTH),
        eye: GROUND_EYE,
        look: GROUND_LOOK,
        minute: 720,
        ui: Some(UiFixture::Bare),
    },
    Scenario {
        name: "ui-partyinvite",
        map: Some(MAP_AZEROTH),
        eye: GROUND_EYE,
        look: GROUND_LOOK,
        minute: 720,
        ui: Some(UiFixture::PartyInvite),
    },
    // The main action bar, its slots and XP seeded by `demo_unit_feed`. The bar is 1024 wide plus
    // 128 px end caps, so `lib.rs` gives this scenario a wider, shorter window.
    Scenario {
        name: "ui-actionbar",
        map: Some(MAP_AZEROTH),
        eye: GROUND_EYE,
        look: GROUND_LOOK,
        minute: 720,
        ui: Some(UiFixture::Bare),
    },
    // Framed like the reference screenshot: an eye-height look at a wolf ~8 yd off. Plates are
    // widgets hung off `WorldFrame`, so they need this scenario's `ui:` fixture; `lib.rs` sizes the
    // window 1024×768, the 1:1 gx window.
    Scenario {
        name: "vplates",
        map: Some(MAP_AZEROTH),
        eye: [-8956.5, -137.5, 85.6],
        look: [-8949.95, -132.49, 84.8],
        minute: 720,
        ui: Some(UiFixture::VPlates),
    },
    // `lib.rs` gives the map's centred 1024×768 chrome a 1100×800 window.
    Scenario {
        name: "ui-worldmap",
        map: Some(MAP_AZEROTH),
        eye: GROUND_EYE,
        look: GROUND_LOOK,
        minute: 720,
        ui: Some(UiFixture::WorldMap),
    },
    // A human warrior's book seeded with two off-class spells.
    Scenario {
        name: "ui-spellbook",
        map: Some(MAP_AZEROTH),
        eye: GROUND_EYE,
        look: GROUND_LOOK,
        minute: 720,
        ui: Some(UiFixture::SpellBook),
    },
    Scenario {
        name: "ui-macro",
        map: Some(MAP_AZEROTH),
        eye: GROUND_EYE,
        look: GROUND_LOOK,
        minute: 720,
        ui: Some(UiFixture::Macro),
    },
    Scenario {
        name: "ui-macro-popup",
        map: Some(MAP_AZEROTH),
        eye: GROUND_EYE,
        look: GROUND_LOOK,
        minute: 720,
        ui: Some(UiFixture::MacroPopup),
    },
    Scenario {
        name: "ui-chatedit",
        map: Some(MAP_AZEROTH),
        eye: GROUND_EYE,
        look: GROUND_LOOK,
        minute: 720,
        ui: Some(UiFixture::ChatEdit),
    },
    // The tab plate and its additive highlight over the world, which the UI-over-world composite
    // decides.
    Scenario {
        name: "ui-chat-tabhover",
        map: Some(MAP_AZEROTH),
        eye: GROUND_EYE,
        look: GROUND_LOOK,
        minute: 720,
        ui: Some(UiFixture::ChatTabHover),
    },
    Scenario {
        name: "ui-social",
        map: Some(MAP_AZEROTH),
        eye: GROUND_EYE,
        look: GROUND_LOOK,
        minute: 720,
        ui: Some(UiFixture::Social),
    },
    // The backdrop is whatever `Map.dbc` and `LoadingScreens.dbc` give this map, so the shot covers
    // the art chain and the bar as well as the tip.
    Scenario {
        name: "loading-tip",
        map: Some(MAP_AZEROTH),
        eye: GROUND_EYE,
        look: GROUND_LOOK,
        minute: 720,
        ui: Some(UiFixture::LoadingTip),
    },
    Scenario {
        name: "ui-options",
        map: Some(MAP_AZEROTH),
        eye: GROUND_EYE,
        look: GROUND_LOOK,
        minute: 720,
        ui: Some(UiFixture::Options),
    },
    Scenario {
        name: "ui-options-audio",
        map: Some(MAP_AZEROTH),
        eye: GROUND_EYE,
        look: GROUND_LOOK,
        minute: 720,
        ui: Some(UiFixture::OptionsAudio),
    },
    Scenario {
        name: "ui-options-graphics",
        map: Some(MAP_AZEROTH),
        eye: GROUND_EYE,
        look: GROUND_LOOK,
        minute: 720,
        ui: Some(UiFixture::OptionsGraphics),
    },
    Scenario {
        name: "ui-options-chat",
        map: Some(MAP_AZEROTH),
        eye: GROUND_EYE,
        look: GROUND_LOOK,
        minute: 720,
        ui: Some(UiFixture::OptionsChat),
    },
    Scenario {
        name: "ui-color-picker",
        map: Some(MAP_AZEROTH),
        eye: GROUND_EYE,
        look: GROUND_LOOK,
        minute: 720,
        ui: Some(UiFixture::ColorPicker),
    },
    Scenario {
        name: "ui-options-dropdown",
        map: Some(MAP_AZEROTH),
        eye: GROUND_EYE,
        look: GROUND_LOOK,
        minute: 720,
        ui: Some(UiFixture::OptionsDropdownList),
    },
    Scenario {
        name: "ui-keybindings",
        map: Some(MAP_AZEROTH),
        eye: GROUND_EYE,
        look: GROUND_LOOK,
        minute: 720,
        ui: Some(UiFixture::KeyBindings),
    },
    Scenario {
        name: "ui-options-search",
        map: Some(MAP_AZEROTH),
        eye: GROUND_EYE,
        look: GROUND_LOOK,
        minute: 720,
        ui: Some(UiFixture::OptionsSearch),
    },
    // Same camera as `water-noon`, plus a named unit out in the water.
    Scenario {
        name: "name-water",
        map: Some(MAP_AZEROTH),
        eye: WATER_EYE,
        look: WATER_LOOK,
        minute: 720,
        ui: Some(UiFixture::NameWater),
    },
    // ---- Searing Gorge, the MAGMA-LIGHT reproducers (report: "lava not really glowing") ----
    // `lava-searing` is the owner's own spot and clock, server-less: their `.go xyz` pin lifted to
    // eye height, looking down their reported facing of 1.53 rad, at MIDNIGHT — the only time of
    // day at which a warm fixture over lava is separable from the sun. `-noon` is the control (the
    // same frame with the sky on: if the rock reads in one and not the other, the defect is the
    // fixture's and not the terrain's). `-river` stands ON the bank of a lava channel, because the
    // owner's pin may simply be out of reach of any magma — 16 yd is a brazier, not a floodlight.
    Scenario {
        name: "lava-searing",
        map: Some(MAP_AZEROTH),
        eye: LAVA_SEARING_EYE,
        look: LAVA_SEARING_LOOK,
        minute: 0,
        ui: None,
    },
    Scenario {
        name: "lava-searing-noon",
        map: Some(MAP_AZEROTH),
        eye: LAVA_SEARING_EYE,
        look: LAVA_SEARING_LOOK,
        minute: 720,
        ui: None,
    },
    Scenario {
        name: "lava-searing-river",
        map: Some(MAP_AZEROTH),
        eye: LAVA_RIVER_EYE,
        look: LAVA_RIVER_LOOK,
        minute: 0,
        ui: None,
    }, // MONKEY (daylight): city interiors by day and terrain torch shadows at night. World coords
    // from the census (`lighting::daylight::census`, Stormwind uid 10047, Ironforge uid 7706).
    // The Gilded Rose ground floor (group g268, portal + aperture seeds).
    Scenario {
        name: "daylight-sw-inn",
        map: Some(MAP_AZEROTH),
        eye: [-8871.0, 681.0, 99.8],
        look: [-8858.0, 668.0, 98.6],
        minute: 720,
        ui: None,
    },
    // The Cathedral of Light nave (groups g135/g146, lit only by two EXT-class window batches).
    Scenario {
        name: "daylight-sw-cathedral",
        map: Some(MAP_AZEROTH),
        eye: [-8556.0, 826.0, 109.0],
        look: [-8515.0, 862.0, 112.0],
        minute: 720,
        ui: None,
    },
    // A Trade District house (group g68 NEH02, one exterior portal).
    Scenario {
        name: "daylight-sw-shop",
        map: Some(MAP_AZEROTH),
        eye: [-8799.0, 696.0, 104.3],
        look: [-8786.0, 707.0, 103.0],
        minute: 720,
        ui: None,
    },
    // Ironforge: the Great Forge hall (g64, no opening to the sky) and the gate hall (g7, the
    // city's one exterior portal).
    Scenario {
        name: "daylight-if-forge",
        map: Some(MAP_AZEROTH),
        eye: [-4930.0, -945.0, 503.5],
        look: [-4890.0, -985.0, 503.0],
        minute: 720,
        ui: None,
    },
    Scenario {
        name: "daylight-if-gate",
        map: Some(MAP_AZEROTH),
        eye: [-4975.0, -895.0, 503.5],
        look: [-5005.0, -852.0, 506.0],
        minute: 720,
        ui: None,
    }, // Goldshire's lamps at midnight: terrain receiving (and, with `torchTerrainShadows`, casting)
    // torch cube shadows.
    Scenario {
        name: "daylight-torch-goldshire",
        map: Some(MAP_AZEROTH),
        eye: [-9460.0, 70.0, 58.0],
        look: [-9450.0, 20.0, 61.0],
        minute: 0,
        ui: None,
    }, // Two Elwynn road lampposts whose own light the terrain blocks most (the census's
    // `lamp_terrain_occlusion_scan`: 18 % and 16 % of the ground within 25 yd), at midnight, framed
    // across the lamp toward the blocked side.
    Scenario {
        name: "daylight-torch-hill-a",
        map: Some(MAP_AZEROTH),
        eye: [-9314.8, 134.1, 70.0],
        look: [-9320.4, 165.6, 64.0],
        minute: 0,
        ui: None,
    },
    Scenario {
        name: "daylight-torch-hill-b",
        map: Some(MAP_AZEROTH),
        eye: [-9163.1, 181.1, 77.0],
        look: [-9147.1, 153.4, 71.0],
        minute: 0,
        ui: None,
    },
    // MONKEY (post): lane-specific A/B and perf viewpoints. Kept at the end for merge isolation.
    Scenario {
        name: "post-lava-searing",
        map: Some(MAP_AZEROTH),
        eye: LAVA_RIVER_EYE,
        look: LAVA_RIVER_LOOK,
        minute: 0,
        ui: None,
    },
    Scenario {
        name: "post-stormwind-night",
        map: Some(MAP_AZEROTH),
        eye: [-8833.38, 628.63, 96.0],
        look: [-8809.1, 672.3, 94.0],
        minute: 0,
        ui: None,
    },
    Scenario {
        name: "post-sunshafts-noon",
        map: Some(MAP_AZEROTH),
        eye: WATER_EYE,
        // Noon's 85-degree sun above the Goldshire trees (azimuth 45 degrees).
        look: [-9508.5, -292.1, 369.7],
        minute: 720,
        ui: None,
    },
    Scenario {
        name: "post-duskwood-grade",
        map: Some(MAP_AZEROTH),
        eye: [-10580.0, -1200.0, 45.0],
        look: [-10540.0, -1160.0, 30.0],
        minute: 1260,
        ui: None,
    },
    // ---- MONKEY (p0 baseline): the graphics programme's baseline set ----
    // Every lane diffs its work against captures of these (plus the water-*, volfog-* and canal
    // scenes above). Raw WoW coords, map per scenario; noon = 720, dusk = 1170, night = 0.
    // Elwynn from the Goldshire road toward the lake and the forest: sky, fog and trees.
    Scenario {
        name: "elwynn-noon",
        map: Some(MAP_AZEROTH),
        eye: GFX_ELWYNN_EYE,
        look: GFX_ELWYNN_LOOK,
        minute: 720,
        ui: None,
    },
    Scenario {
        name: "elwynn-dusk",
        map: Some(MAP_AZEROTH),
        eye: GFX_ELWYNN_EYE,
        look: GFX_ELWYNN_LOOK,
        minute: 1170,
        ui: None,
    },
    Scenario {
        name: "elwynn-night",
        map: Some(MAP_AZEROTH),
        eye: GFX_ELWYNN_EYE,
        look: GFX_ELWYNN_LOOK,
        minute: 0,
        ui: None,
    },
    // The same Elwynn view for rain. Capture has no weather field: run it with
    // `WOW_WEATHER=1,0.8` (kind 1 rain, grade 0.8; `weather::parse_env_script`).
    Scenario {
        name: "elwynn-rain",
        map: Some(MAP_AZEROTH),
        eye: GFX_ELWYNN_EYE,
        look: GFX_ELWYNN_LOOK,
        minute: 720,
        ui: None,
    },
    // Westfall: a farmstead, fields and the dry palette.
    Scenario {
        name: "westfall-farm",
        map: Some(MAP_AZEROTH),
        eye: [-10080.0, 1000.0, 58.0],
        look: [-10140.0, 1070.0, 38.0],
        minute: 720,
        ui: None,
    },
    // Redridge: Lake Everstill from above Lakeshire, a wider frame than `water-lake`.
    Scenario {
        name: "redridge-lake",
        map: Some(MAP_AZEROTH),
        eye: [-9350.0, -2340.0, 82.0],
        look: [-9500.0, -2650.0, 50.0],
        minute: 720,
        ui: None,
    },
    // Duskwood: the road into Darkshire, the zone's short dark fog.
    Scenario {
        name: "duskwood-road",
        map: Some(MAP_AZEROTH),
        eye: [-10560.0, -1000.0, 50.0],
        look: [-10570.0, -1200.0, 32.0],
        minute: 720,
        ui: None,
    },
    // Burning Steppes: the ash plain and the sky a skybox would replace.
    Scenario {
        name: "burning-steppes",
        map: Some(MAP_AZEROTH),
        eye: [-7700.0, -2100.0, 200.0],
        look: [-7900.0, -1700.0, 150.0],
        minute: 720,
        ui: None,
    },
    // Blasted Lands: the Dark Portal from the north.
    Scenario {
        name: "blasted-lands-portal",
        map: Some(MAP_AZEROTH),
        eye: [-11700.0, -3200.0, 20.0],
        look: [-11900.0, -3208.0, 0.0],
        minute: 720,
        ui: None,
    },
    // Mount Hyjal (Kalimdor), inside Light.dbc sphere 270.
    Scenario {
        name: "hyjal-mount",
        map: Some(MAP_KALIMDOR),
        eye: GFX_HYJAL_EYE,
        look: GFX_HYJAL_LOOK,
        minute: 720,
        ui: None,
    },
    // Stormwind: the Cathedral of Light's nave (interior WMO lighting).
    Scenario {
        name: "stormwind-cathedral-interior",
        map: Some(MAP_AZEROTH),
        eye: GFX_CATHEDRAL_EYE,
        look: GFX_CATHEDRAL_LOOK,
        minute: 720,
        ui: None,
    },
    // Ironforge: the Great Forge (lava, fire light, a huge interior).
    Scenario {
        name: "ironforge-forge",
        map: Some(MAP_AZEROTH),
        eye: GFX_FORGE_EYE,
        look: GFX_FORGE_LOOK,
        minute: 720,
        ui: None,
    },
    // MONKEY (sky): the sky lane's fixtures, from 80 yd above Northshire (clear of the canopy).
    // Dusk faces the low sun (az 45°, elev ≈5° at 20:00); night faces away from the moon at 01:00
    // (full star curve); zenith looks nearly straight up; overcast is meant for
    // `WOW_WEATHER=1,0.5`; noon faces away from the sun.
    Scenario {
        name: "sky-elwynn-dusk",
        map: Some(MAP_AZEROTH),
        eye: SKY_HIGH_EYE,
        look: [-8770.0, 50.0, 232.0],
        minute: 1200,
        ui: None,
    },
    Scenario {
        name: "sky-elwynn-night",
        map: Some(MAP_AZEROTH),
        eye: SKY_HIGH_EYE,
        look: [-9164.0, -344.0, 340.0],
        minute: 60,
        ui: None,
    },
    Scenario {
        name: "sky-zenith-night",
        map: Some(MAP_AZEROTH),
        eye: SKY_HIGH_EYE,
        look: [-9017.0, -197.0, 485.0],
        minute: 60,
        ui: None,
    },
    Scenario {
        name: "sky-overcast",
        map: Some(MAP_AZEROTH),
        eye: SKY_HIGH_EYE,
        look: [-9185.0, 45.0, 268.0],
        minute: 1000,
        ui: None,
    },
    Scenario {
        name: "sky-noon",
        map: Some(MAP_AZEROTH),
        eye: SKY_HIGH_EYE,
        look: [-9185.0, -365.0, 268.0],
        minute: 720,
        ui: None,
    },
    // MONKEY wind: keep these at the END of the table. They are the three visual instruments for
    // W1: authored grass weight/lean, static-tree classification/sway, and the viewer bender.
    Scenario {
        name: "wind-grass-elwynn",
        map: Some(MAP_AZEROTH),
        // MONKEY (integration): restore the lane's earlier world-visible framing. The closer
        // handoff coordinates put the camera inside a Goldshire building and showed no grass.
        eye: [-9505.0, 85.0, 64.0],
        look: [-9460.0, 45.0, 57.0],
        minute: 720,
        ui: None,
    },
    Scenario {
        name: "wind-forest-elwynn",
        map: Some(MAP_AZEROTH),
        // East of Goldshire looking south-west across the lake and Elwynn forest.
        eye: [-9380.0, -30.0, 80.0],
        look: [-9600.0, -200.0, 62.0],
        minute: 720,
        ui: None,
    },
    Scenario {
        name: "wind-player-parting",
        map: Some(MAP_AZEROTH),
        eye: [-9472.0, 57.0, 60.5],
        look: [-9460.0, 45.0, 57.5],
        minute: 720,
        // The capture plugin stands a real player at `look` without opting into the HUD: this is
        // a world-image instrument, and UI script errors must not cover the grass it measures.
        ui: None,
    },
    // MONKEY (fog): Duskwood's short zone fog (Light 15, LightParams 14) from above the canopy west
    // of Darkshire toward the town, for the Classic/Modern fog A/B (`WOW_FOGMODEL=0|1`).
    Scenario {
        name: "fog-duskwood-noon",
        map: Some(MAP_AZEROTH),
        eye: [-10700.0, -900.0, 160.0],
        look: [-10560.0, -1180.0, 60.0],
        minute: 720,
        ui: None,
    },
    // MONKEY (fog): a long Elwynn vista from the Northshire ridge south over Goldshire's forest.
    Scenario {
        name: "fog-vista-noon",
        map: Some(MAP_AZEROTH),
        eye: [-9000.0, -100.0, 200.0],
        look: [-9500.0, -100.0, 90.0],
        minute: 720,
        ui: None,
    },
    Scenario {
        name: "fog-vista-dusk",
        map: Some(MAP_AZEROTH),
        eye: [-9000.0, -100.0, 200.0],
        look: [-9500.0, -100.0, 90.0],
        minute: 1170,
        ui: None,
    },
    Scenario {
        name: "fog-duskwood-dusk",
        map: Some(MAP_AZEROTH),
        eye: [-10700.0, -900.0, 160.0],
        look: [-10560.0, -1180.0, 60.0],
        minute: 1170,
        ui: None,
    },
    // MONKEY (wet): rain on surfaces. Run with rain forced and the ground pre-soaked, e.g.
    // `WOW_WEATHER=1,0.8 WOW_WETNESS=1 WOW_WET_T=3`, and A/B with `WOW_RAIN_SURFACES=0|1`.
    // Goldshire's square: the road, the grass, the inn's roofs and walls.
    Scenario {
        name: "wet-goldshire",
        map: Some(MAP_AZEROTH),
        eye: [-9460.0, 70.0, 59.5],
        look: [-9452.0, 30.0, 55.0],
        minute: 720,
        ui: None,
    },
    // The Elwynn river bank, close enough (< 25 yd) for the rain rings.
    Scenario {
        name: "wet-river",
        map: Some(MAP_AZEROTH),
        eye: [-9514.0, -330.0, 64.0],
        look: [-9500.0, -352.0, 61.4],
        minute: 720,
        ui: None,
    },
    // A Stormwind Trade District street: exterior WMO paving, walls, eaves.
    Scenario {
        name: "wet-stormwind",
        map: Some(MAP_AZEROTH),
        eye: [-8833.38, 628.63, 98.5],
        look: [-8815.0, 662.0, 93.5],
        minute: 720,
        ui: None,
    },
    // MONKEY (fix-wet): the wet-stormwind framing at midnight: lit windows must stay lit in rain.
    Scenario {
        name: "wet-stormwind-night",
        map: Some(MAP_AZEROTH),
        eye: [-8833.38, 628.63, 98.5],
        look: [-8815.0, 662.0, 93.5],
        minute: 0,
        ui: None,
    },
    // MONKEY (ao): contact-shadow subjects. Goldshire's street (props against walls, eaves),
    // an Elwynn forest floor (trunks, bushes, grass cutouts) and a close unit on open ground.
    Scenario {
        name: "ao-goldshire",
        map: Some(MAP_AZEROTH),
        eye: [-9430.0, 50.0, 61.0],
        look: [-9462.0, 30.0, 57.5],
        minute: 720,
        ui: None,
    },
    Scenario {
        name: "ao-forest",
        map: Some(MAP_AZEROTH),
        eye: [-9560.0, 60.0, 62.0],
        look: [-9600.0, 90.0, 58.0],
        minute: 720,
        ui: None,
    },
    Scenario {
        name: "ao-character",
        map: Some(MAP_AZEROTH),
        eye: [-9495.2, 60.8, 59.0],
        look: [-9500.00, 56.00, 56.9],
        minute: 720,
        ui: Some(UiFixture::Subject {
            kind: SubjectKind::Creature,
            at: SUBJECT_SUN,
        }),
    },
    // MONKEY (lampfog): lampFog 0/2 A/B set. Kept at the END so parallel lane tables merge cleanly.
    // Goldshire square: the Lion's Pride fixtures and outdoor lamps share one night frame.
    Scenario {
        name: "lampfog-goldshire-night",
        map: Some(MAP_AZEROTH),
        eye: [-9460.0, 70.0, 58.0],
        look: [-9450.0, 20.0, 61.0],
        minute: 0,
        ui: None,
    },
    // Stormwind Trade District from the established city-scale camera, after dark.
    Scenario {
        name: "lampfog-stormwind-night",
        map: Some(MAP_AZEROTH),
        eye: [-8833.38, 628.63, 96.0],
        look: [-8809.1, 672.3, 94.0],
        minute: 0,
        ui: None,
    },
    // Goldshire's east road, looking through the known exterior lamp at (-9477, 53, 60). This is
    // deliberately not the fence instrument: that camera looks down at an unlit verge at night.
    Scenario {
        name: "lampfog-elwynn-road-night",
        map: Some(MAP_AZEROTH),
        eye: [-9505.0, 36.0, 61.5],
        look: [-9455.0, 75.0, 59.0],
        minute: 0,
        ui: None,
    },
    // Exact Goldshire framing at noon: lampFog 0 and 2 must be pixel-identical.
    Scenario {
        name: "lampfog-goldshire-day",
        map: Some(MAP_AZEROTH),
        eye: [-9460.0, 70.0, 58.0],
        look: [-9450.0, 20.0, 61.0],
        minute: 720,
        ui: None,
    },
    // MONKEY (skybox): zone skyboxes (run with `WOW_ZONE_SKYBOXES=1`) and the two stock skybox
    // lanes. Karazahn40 (Turtle map 814) is the one stock sphere naming a clear-slot skybox
    // (Light 553, HellfireSkyBox); the zone shots need the sky patch.
    // The stock MOSB lane: Stratholme_B's groups flagged 0x40000, weight = the interior crossfade.
    Scenario {
        name: "skybox-stratholme-noon",
        map: Some(329),
        eye: SKYBOX_STRAT_EYE,
        look: SKYBOX_STRAT_LOOK,
        minute: 720,
        ui: None,
    },
    Scenario {
        name: "skybox-karazhan-noon",
        map: Some(814),
        eye: SKYBOX_KZ_EYE,
        look: SKYBOX_KZ_LOOK,
        minute: 720,
        ui: None,
    },
    Scenario {
        name: "skybox-karazhan-night",
        map: Some(814),
        eye: SKYBOX_KZ_EYE,
        look: SKYBOX_KZ_LOOK,
        minute: 0,
        ui: None,
    },
    Scenario {
        name: "skybox-steppes-noon",
        map: Some(MAP_AZEROTH),
        eye: SKYBOX_STEPPES_EYE,
        look: SKYBOX_STEPPES_LOOK,
        minute: 720,
        ui: None,
    },
    Scenario {
        name: "skybox-steppes-night",
        map: Some(MAP_AZEROTH),
        eye: SKYBOX_STEPPES_EYE,
        look: SKYBOX_STEPPES_LOOK,
        minute: 0,
        ui: None,
    },
    Scenario {
        name: "skybox-blasted-noon",
        map: Some(MAP_AZEROTH),
        eye: SKYBOX_BLASTED_EYE,
        look: SKYBOX_BLASTED_LOOK,
        minute: 720,
        ui: None,
    },
    Scenario {
        name: "skybox-blasted-night",
        map: Some(MAP_AZEROTH),
        eye: SKYBOX_BLASTED_EYE,
        look: SKYBOX_BLASTED_LOOK,
        minute: 0,
        ui: None,
    },
    Scenario {
        name: "skybox-hyjal-noon",
        map: Some(MAP_KALIMDOR),
        eye: SKYBOX_HYJAL_EYE,
        look: SKYBOX_HYJAL_LOOK,
        minute: 720,
        ui: None,
    },
    Scenario {
        name: "skybox-hyjal-night",
        map: Some(MAP_KALIMDOR),
        eye: SKYBOX_HYJAL_EYE,
        look: SKYBOX_HYJAL_LOOK,
        minute: 0,
        ui: None,
    },
    // MONKEY (integration): the Westfall sea from 30 yd up, aimed ~72 yd out, so the enhanced
    // water's 64-80 yd fine/coarse LOD ring crosses the frame centre (review #7, WOW_WATER=2).
    Scenario {
        name: "water-ocean-lodring",
        map: Some(MAP_AZEROTH),
        eye: [-10500.0, 2112.0, 30.0],
        look: [-10449.0, 2163.0, 0.0],
        minute: 720,
        ui: None,
    },
    // MONKEY (perf): Northshire at midnight from the ground — where the owner's night FPS drop is
    // reported (torch lanes, lamp fog, bleed fixtures all live). Probe fixtures, not look fixtures.
    Scenario {
        name: "perf-northshire-night",
        map: Some(MAP_AZEROTH),
        eye: GROUND_EYE,
        look: GROUND_LOOK,
        minute: 0,
        ui: None,
    },
    // MONKEY (perf): the abbey front at midnight, close to its lamps and candelabra.
    Scenario {
        name: "perf-abbey-night",
        map: Some(MAP_AZEROTH),
        eye: [-8935.0, -145.0, 86.0],
        look: [-8905.0, -160.0, 85.0],
        minute: 0,
        ui: None,
    },
    // MONKEY (leftovers): the Westfall sea with the visible sun ~5° up on its fixed 45° bearing
    // (minute 1200), framed so the sun sits above the open water: the sun-shaft sky mask test.
    Scenario {
        name: "leftovers-sea-sunset",
        map: Some(MAP_AZEROTH),
        eye: [-10500.0, 2112.0, 6.0],
        look: [-10420.0, 2192.0, 8.0],
        minute: 1200,
        ui: None,
    },
    // GFX (volumetric light) / (moonlight): the Goldshire lake-bank forest, facing the fixed 45°
    // bearing the sun and the moon share. Afternoon shafts through the canopy; the same spot at
    // midnight (moonlight + moon shadows + faint moon shafts) and across the dusk hand-over.
    Scenario {
        name: "vol-shafts-day",
        map: Some(MAP_AZEROTH),
        eye: VOL_FOREST_EYE,
        look: [-9488.5, -258.5, 80.5], // face 45°, pitch 12°
        minute: 1000,
        ui: None,
    },
    Scenario {
        name: "vol-moon-forest",
        map: Some(MAP_AZEROTH),
        eye: VOL_FOREST_EYE,
        look: [-9489.0, -259.0, 83.5], // face 45°, pitch 15°
        minute: 0,
        ui: None,
    },
    Scenario {
        name: "vol-dusk-2000",
        map: Some(MAP_AZEROTH),
        eye: VOL_FOREST_EYE,
        look: [-9488.5, -258.5, 72.0],
        minute: 1200,
        ui: None,
    },
    Scenario {
        name: "vol-dusk-2130",
        map: Some(MAP_AZEROTH),
        eye: VOL_FOREST_EYE,
        look: [-9488.5, -258.5, 72.0],
        minute: 1290,
        ui: None,
    },
    Scenario {
        name: "vol-moonrise-2300",
        map: Some(MAP_AZEROTH),
        eye: VOL_FOREST_EYE,
        look: [-9488.5, -258.5, 72.0],
        minute: 1380,
        ui: None,
    },
];

/// GFX (volumetric light): the Goldshire lake bank among the trees (eye 2 yd over the feet).
pub(super) const VOL_FOREST_EYE: [f32; 3] = [-9530.0, -300.0, 68.0];
/// MONKEY (p0 baseline): the Elwynn programme view, above the canopy east of Goldshire looking
/// south-west over the forest to the river and the hills (framed with the `vista` instrument).
pub(super) const GFX_ELWYNN_EYE: [f32; 3] = [-9400.0, -100.0, 122.0];
pub(super) const GFX_ELWYNN_LOOK: [f32; 3] = [-9477.6, -160.6, 104.6];
/// MONKEY (p0 baseline): Mount Hyjal, over Light.dbc sphere 270's centre (world yards; the
/// sphere's own z 873 is below the terrain), pitched a little up: its params-269 fog (end 278 yd,
/// orange) fills the frame today, which is the subject the skybox and fog lanes change.
pub(super) const GFX_HYJAL_EYE: [f32; 3] = [4636.0, -4461.0, 1152.0];
pub(super) const GFX_HYJAL_LOOK: [f32; 3] = [4724.2, -4416.0, 1165.9];
/// MONKEY (p0 baseline): the Cathedral of Light nave, standing over the floor.
pub(super) const GFX_CATHEDRAL_EYE: [f32; 3] = [-8530.0, 845.0, 112.0];
pub(super) const GFX_CATHEDRAL_LOOK: [f32; 3] = [-8500.0, 880.0, 110.0];
/// MONKEY (p0 baseline): Ironforge's interior by the Great Forge ring (a lit WMO interior; the
/// camera must stand over a floor face or the portal cull hides the city).
pub(super) const GFX_FORGE_EYE: [f32; 3] = [-4880.0, -1000.0, 506.0];
pub(super) const GFX_FORGE_LOOK: [f32; 3] = [-4950.4, -929.6, 497.3];

/// The owner's reported vantage for the lava-glow report: `(-7048.8, -1000.6, 242.0)` facing
/// 1.53 rad, Searing Gorge, map 0. The eye takes the standard [`VISTA_EYE_HEIGHT`]-ish lift off
/// the pin (a `.go` lands at the feet) and the look runs 60 yd down the reported facing,
/// `(cos 1.53, sin 1.53) = (0.041, 0.999)`, pitched a little down so the GROUND — the surface the
/// report says is only faintly tinted — fills the lower frame rather than the sky.
pub(super) const LAVA_SEARING_EYE: [f32; 3] = [-7048.8, -1000.6, 244.0];
pub(super) const LAVA_SEARING_LOOK: [f32; 3] = [-7046.3, -940.7, 236.0];

/// An OPEN lava river in Searing Gorge, found from the run's own magma census (see
/// `_fx/lava_findings.md`) — the shot that answers "does magma light the rock beside it", with the
/// magma unambiguously inside a fixture's reach and the receivers on the EXTERIOR lane.
///
/// Picking this spot was the round's methodological lesson. The obvious choice — the nearest magma
/// to the owner's pin — is a channel under a cave roof, where every visible surface is
/// interior-class WMO. A "does the lava light the rock" instrument planted there measures the
/// interior lane and says nothing about the open world, and it cost one wrong conclusion before
/// the pixels caught it. This vantage is open sky over open terrain.
pub(super) const LAVA_RIVER_EYE: [f32; 3] = [-7469.25, -848.17, 269.83];
pub(super) const LAVA_RIVER_LOOK: [f32; 3] = [-7505.6, -861.4, 259.5];

/// The `name-close` subject: the `name-water` wolf's name, orbited at any distance. World text is
/// the only consumer that draws the glyph sheet at other than 1:1, so a glyph-cell sampling defect
/// shows only in a magnified name.
///
/// Knobs, defaults in parentheses: `WOW_NAME_DIST` yd from the name (4), `WOW_NAME_AZ` the eye's
/// bearing in degrees, 0 = +X (124), `WOW_NAME_EL` its elevation in degrees (8), `WOW_NAME_H` the
/// name's height above the feet in yd (1.4). The camera looks straight at the name.
pub(super) const NAME_CLOSE_AT: [f32; 3] = super::fixtures::NAME_WATER_POS;

/// The Deeprun Tram tube (`tram-undersea`). The Subway WMO is the map's global `MODF` at the
/// origin with identity rotation, so these are also its model-space coords.
pub(super) const TRAM_EYE: [f32; 3] = [-2.44, -1250.0, -120.0];
pub(super) const TRAM_LOOK: [f32; 3] = [-2.44, -1400.0, -118.0];

/// The Tainted Scar (`tainted-scar-noon`): the eye is `.go xyz -11892.70 -2647.08 -4.68`
/// (`game_tele TheTaintedScar`) lifted clear of the crater floor, looking north 25° up, since the
/// rim hides the dome from a level look; the ridge sits low in the frame as the control.
pub(super) const SCAR_EYE: [f32; 3] = [-11892.7, -2647.1, 20.0];
pub(super) const SCAR_LOOK: [f32; 3] = [-11792.7, -2647.1, 66.6];

/// Find a scenario by name across both tables, as `WOW_CAPTURE=` resolution does; every `ui-*`
/// fixture lives in [`ON_DEMAND`].
pub(super) fn by_name(name: &str) -> Option<&'static Scenario> {
    SCENARIOS
        .iter()
        .chain(ON_DEMAND.iter())
        .find(|s| s.name == name)
}

fn scenario_declares_ui() -> bool {
    std::env::var("WOW_CAPTURE")
        .ok()
        .and_then(|name| by_name(&name))
        .is_some_and(|s| s.ui.is_some())
}

/// Whether the player UI is opted into this capture: the scenario declares a `ui:` fixture, or
/// `WOW_CAPTURE_UI=1` asks for it over a world scene, whose baselines are otherwise UI-free.
///
/// Its three consumers must agree, or the capture is plausible and wrong: the UI load
/// (`ui_script::lifecycle::ui_wanted`), the synthetic `"player"`/`"target"` snapshot
/// (`ui_script::capture_ui_active`, `demo_unit_feed`) and the window size (`lib.rs`).
pub(crate) fn ui_opted_in() -> bool {
    std::env::var("WOW_CAPTURE_UI").as_deref() == Ok("1") || scenario_declares_ui()
}

// MONKEY (skybox): the skybox shots, eye just above the ground at a sphere's centre, pitched up so
// the upper frame is sky.
const SKYBOX_STRAT_EYE: [f32; 3] = [3450.0, -3380.0, 150.0];
const SKYBOX_STRAT_LOOK: [f32; 3] = [3600.0, -3300.0, 200.0];
const SKYBOX_KZ_EYE: [f32; 3] = [-6474.0, -2912.0, 40.0];
const SKYBOX_KZ_LOOK: [f32; 3] = [-6300.0, -2800.0, 95.0];
const SKYBOX_STEPPES_EYE: [f32; 3] = [-7979.0, -2571.0, 260.0];
const SKYBOX_STEPPES_LOOK: [f32; 3] = [-7800.0, -2450.0, 320.0];
const SKYBOX_BLASTED_EYE: [f32; 3] = [-11300.0, -3073.0, 30.0];
const SKYBOX_BLASTED_LOOK: [f32; 3] = [-11120.0, -2960.0, 90.0];
const SKYBOX_HYJAL_EYE: [f32; 3] = [4637.0, -4461.0, 1130.0];
const SKYBOX_HYJAL_LOOK: [f32; 3] = [4800.0, -4350.0, 1230.0];

#[cfg(test)]
mod ui_opt_in_tests {
    use super::*;
    use crate::local_state::test_env::{EnvGuard, ENV_LOCK};

    /// A window scenario opts the UI in without `WOW_CAPTURE_UI=1`; a world scenario does not.
    #[test]
    fn a_ui_scenario_opts_the_ui_in_and_a_world_scenario_does_not() {
        let _l = ENV_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let _ui = EnvGuard::unset("WOW_CAPTURE_UI");

        let _c = EnvGuard::set("WOW_CAPTURE", "ui-questlog");
        assert!(
            ui_opted_in(),
            "a `ui:` scenario is a UI capture by construction"
        );

        // A `Bare` fixture opens no window and still opts in.
        let _c = EnvGuard::set("WOW_CAPTURE", "ui-unitframes");
        assert!(ui_opted_in(), "…including the fixtureless `Bare` ones");

        // A world scenario's baseline tests the world render and stays UI-free.
        let world = SCENARIOS
            .iter()
            .chain(ON_DEMAND.iter())
            .find(|s| s.ui.is_none())
            .expect("the tables have world scenarios");
        let _c = EnvGuard::set("WOW_CAPTURE", world.name);
        assert!(
            !ui_opted_in(),
            "world scenario {} must stay pristine",
            world.name
        );

        // The env var opts a world scene in.
        let _on = EnvGuard::set("WOW_CAPTURE_UI", "1");
        assert!(ui_opted_in(), "the env var still opts a world scene in");
    }

    /// A `ui-*` name without a fixture would be a world capture wearing a UI name.
    #[test]
    fn every_ui_named_scenario_declares_a_fixture() {
        for s in SCENARIOS
            .iter()
            .chain(ON_DEMAND.iter())
            .filter(|s| s.name.starts_with("ui-"))
        {
            assert!(
                s.ui.is_some(),
                "scenario {} is named ui-* but declares no `ui:` fixture — it would be treated as \
                 a world capture and photographed with no player UI. If it opens no window, that \
                 is what `UiFixture::Bare` is for.",
                s.name
            );
        }
    }
}

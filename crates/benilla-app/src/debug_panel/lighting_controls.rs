//! MONKEY (lighting debug panel): a native editor for the LIVE video CVars. Reads are typed
//! `VideoConfig` fields, never a second settings store; changed widgets alone call
//! `UiScript::set_cvar_engine`, whose queue is the same one Lua `SetCVar` feeds. The normal host
//! sync owns clamps, renderer bridges and persistence, so a slider and a chat command cannot
//! acquire separate truths. Registry lookup is reserved for an explicit reset, not idle frames.

use std::ops::RangeInclusive;

use benilla_ui::script::UiScript;
use bevy_egui::egui;

use crate::cvars::REGISTERED;
use crate::shadow_core::{
    CASTER_REACH_RANGE, MAX_SHADOW_RATE, SHADOW_DISTANCE_RANGE, SHADOW_MAP_SIZES,
};
use crate::video::VideoConfig;

enum Widget {
    Toggle,
    Slider(RangeInclusive<f32>),
    Integer(RangeInclusive<f32>),
    Choice(&'static [u32]),
}

struct Knob {
    name: &'static str,
    read: fn(&VideoConfig) -> f32,
    widget: Widget,
    help: &'static str,
}

// MONKEY (lighting debug panel): keep names, typed reads and presentation together. The numeric
// bounds mirror `video::on_cvar`'s MONKEY (lighting) arms — the registry's change-callback
// observer since 2303 — including their wider ambient/fill/attenuation ranges;
// shared shadow constants follow the setter directly. Casting the booleans through u32 keeps
// the descriptor read-only without inventing a parallel configuration type.
macro_rules! knob {
    ($name:literal, $field:ident, $widget:expr, $help:literal) => {
        Knob {
            name: $name,
            read: |v| v.$field as f32,
            widget: $widget,
            help: $help,
        }
    };
    (flag $name:literal, $field:ident, $help:literal) => {
        Knob {
            name: $name,
            read: |v| v.$field as u32 as f32,
            widget: Widget::Toggle,
            help: $help,
        }
    };
}

use Widget::{Choice, Integer, Slider};

// This developer-only panel intentionally keeps its labels and help text in English.
const INTERIOR: &[Knob] = &[
    knob!(flag "interiorLight", interior_light, "Fixture-lit interiors; off restores the baked path."),
    knob!(
        "interiorGain",
        interior_gain,
        Slider(0.2..=1.5),
        "Overall room input brightness."
    ),
    knob!(
        "interiorAmbient",
        interior_ambient,
        Slider(0.0..=1.0),
        "Base room ambient."
    ),
    knob!(
        "interiorFill",
        interior_fill,
        Slider(0.0..=2.0),
        "Per-fixture bounce gain."
    ),
    // MONKEY (bake floor): index 4, immediately after fill, because the three presets below take
    // `INTERIOR[1..5]` as one group — this dial belongs with gain/ambient/fill, not with the
    // exposure and reach knobs the presets deliberately leave alone.
    knob!(
        "interiorBakeFloor",
        interior_bake_floor,
        Slider(0.0..=1.0),
        "Share of a room's baked light kept where no fixture reaches; 0 = fixtures only."
    ),
    knob!(
        "interiorExposure",
        interior_exposure,
        Slider(0.25..=8.0),
        "Light budget before soft rolloff."
    ),
    knob!(
        "interiorAttenScale",
        interior_atten_scale,
        Slider(0.0..=8.0),
        "Pool radius scale; 0 disables the authored window."
    ),
    knob!(
        "interiorDaylight",
        interior_daylight,
        Slider(0.0..=1.0),
        "Sun-driven day floor in enclosed rooms."
    ),
    knob!(flag "interiorRoomGate", interior_room_gate, "Restrict fixtures to the rooms they claim."),
];

const GROUPS: &[(&str, &[Knob])] = &[
    (
        "Water",
        &[knob!(
            "waterQuality",
            water_quality,
            Choice(&[0, 1, 2]),
            "0 Classic, 1 Enhanced, 2 High; High adds mirror reflections and costs the most."
        )],
    ),
    ("Interior light", INTERIOR),
    (
        "Fire",
        &[
            knob!(
                "fireLightGain",
                fire_light_gain,
                Slider(0.0..=4.0),
                "Brightness of lights synthesized from flame emitters."
            ),
            // MONKEY (spellLightGain): in the "Fire" group because a spell light IS one of the invented
            // sources this group tunes — it is just the one whose brightness is a combat setting rather
            // than a scenery one, which is exactly why it needs its own dial and not the one above it.
            knob!(
                "spellLightGain",
                spell_light_gain,
                Slider(0.0..=4.0),
                "Brightness of spell, missile and impact lights."
            ),
            knob!(
                "lavaLightGain",
                lava_light_gain,
                Slider(0.0..=4.0),
                "How strongly lava lights its surroundings; 0 turns the glow off."
            ),
            knob!(
                "fireFlicker",
                fire_flicker,
                Slider(0.0..=2.0),
                "0 steady, 1 default, 2 pronounced."
            ),
        ],
    ),
    (
        "Torch shadows",
        &[
            knob!(flag "interiorShadows", interior_shadows, "Interior fixture shadows; requires interiorLight."),
            knob!(flag "exteriorShadows", exterior_shadows, "Outdoor fire shadows at night."),
            knob!(
                "interiorShadowCasters",
                interior_shadow_casters,
                Integer(1.0..=16.0),
                "Resident fixture shadow maps."
            ),
            knob!(
                "interiorShadowDynamic",
                interior_shadow_dynamic,
                Integer(1.0..=crate::torch_shadow::MAX_TORCH_DYNAMIC as f32),
                "Nearest fixtures with moving entity shadows."
            ),
            knob!(
                "interiorShadowSoft",
                interior_shadow_soft,
                Slider(0.5..=3.0),
                "Shadow edge softness at a contact; the penumbra grows from it."
            ),
            knob!(
                "torchShadowStrength",
                torch_shadow_strength,
                Slider(0.0..=1.0),
                "Shadow darkness: how much direct light a shadow removes. 1 = black."
            ),
            knob!(
                "interiorShadowEntityRate",
                interior_shadow_entity_rate,
                Integer(0.0..=240.0),
                "Moving caster regather Hz; 0 = every frame."
            ),
        ],
    ),
    (
        "Sun shadows",
        &[
            knob!(flag "characterShadows", character_shadows, "Realtime character silhouettes instead of blobs."),
            knob!(flag "worldShadows", world_shadows, "Realtime static-world shadows instead of baked terrain shadows."),
            knob!(
                "shadowDistance",
                shadow_distance,
                Slider(SHADOW_DISTANCE_RANGE),
                "Shadow distance in yards."
            ),
            knob!(
                "shadowMapSize",
                shadow_map_size,
                Choice(&SHADOW_MAP_SIZES),
                "Map edge in texels; larger maps cost more GPU time and memory."
            ),
            knob!(
                "shadowFilter",
                shadow_filter,
                Widget::Toggle,
                "0 hardware 2x2, 1 Gaussian PCF."
            ),
            knob!(
                "characterShadowRate",
                character_shadow_rate,
                Integer(0.0..=MAX_SHADOW_RATE as f32),
                "Character shadow update Hz; 0 = every frame."
            ),
            knob!(
                "worldShadowRate",
                world_shadow_rate,
                Integer(0.0..=MAX_SHADOW_RATE as f32),
                "World shadow update Hz; 0 = every frame."
            ),
            knob!(
                "shadowCasterReach",
                shadow_caster_reach,
                Slider(CASTER_REACH_RANGE),
                "Caster collection reach multiplier."
            ),
        ],
    ),
    (
        "Night",
        &[
            knob!(
                "nightGain",
                night_gain,
                Slider(0.2..=1.5),
                "Exterior night brightness; inert by day."
            ),
            // MONKEY (moon shadows): in "Night" rather than beside the sun dials, because that is
            // where both its cost and its taste question live — it is inert by day, and what it buys
            // is "the world is not flat after dark" for a second shadowed directional term.
            knob!(
                "moonShadowStrength",
                moon_shadow_strength,
                Slider(0.0..=1.0),
                "How dark a moon-shadowed fragment gets at night; 0 = no moon shadow."
            ),
        ],
    ),
    (
        "Debug",
        &[knob!(
            "interiorDebug",
            interior_debug,
            Choice(&[0, 1, 2, 3, 4]),
            "0 off, 1 classification, 2 shadow, 3 caster count, 4 WMO lane map."
        )],
    ),
];

fn reset(script: &mut UiScript, knobs: &[Knob]) {
    for knob in knobs {
        let registered = REGISTERED
            .iter()
            .find(|r| r.name == knob.name)
            .expect("lighting panel knob must be registered");
        script.set_cvar_engine(registered.name, registered.default);
    }
}

fn row(ui: &mut egui::Ui, video: &VideoConfig, script: &mut UiScript, knob: &Knob) {
    let mut value = (knob.read)(video);
    // MONKEY (lighting debug panel): stack the label over numeric controls so even the longest
    // CVar fits the existing 280 px panel. Sliders retain their editable current-value readout;
    // integer knobs cannot queue fractional counts/rates, and flags explicitly show their 0/1.
    let changed = ui
        .push_id(knob.name, |ui| match &knob.widget {
            Widget::Toggle => {
                let mut enabled = value != 0.0;
                let response = ui
                    .checkbox(&mut enabled, format!("{}  {}", knob.name, value as u32))
                    .on_hover_text(knob.help);
                value = enabled as u32 as f32;
                response.changed()
            }
            Slider(range) | Integer(range) => {
                ui.label(knob.name).on_hover_text(knob.help);
                let mut slider = egui::Slider::new(&mut value, range.clone());
                if matches!(knob.widget, Integer(_)) {
                    slider = slider.integer();
                }
                ui.add(slider).on_hover_text(knob.help).changed()
            }
            Choice(options) => {
                ui.label(knob.name).on_hover_text(knob.help);
                let mut changed = false;
                egui::ComboBox::from_id_salt("value")
                    .selected_text((value as u32).to_string())
                    .show_ui(ui, |ui| {
                        for &option in *options {
                            changed |= ui
                                .selectable_value(&mut value, option as f32, option.to_string())
                                .changed();
                        }
                    });
                changed
            }
        })
        .inner;
    if changed {
        script.set_cvar_engine(knob.name, &value.to_string());
    }
}

pub(super) fn section(ui: &mut egui::Ui, video: &VideoConfig, script: Option<&mut UiScript>) {
    let Some(script) = script else {
        ui.weak("Lighting controls are available once the UI session is ready.");
        return;
    };
    // MONKEY (advanced graphics): the `lightingQuality` ladder — the same row the player reaches
    // on the Advanced Graphics page, so this panel and that window cannot show different answers.
    // The rungs are `cvars::LIGHTING_PRESETS`, never a second table here; the CURRENT name is read
    // off the VM mirror rather than derived locally, because the registry's own per-frame
    // derivation is the one authority on it and a second derivation would be a second opinion.
    ui.horizontal(|ui| {
        ui.label("Lighting quality").on_hover_text(
            "Writes the member rows below; reads back Custom when they match no rung.",
        );
        let current = script.cvar("lightingQuality").unwrap_or_default();
        for (name, _) in crate::cvars::LIGHTING_PRESETS {
            if ui
                .selectable_label(current.eq_ignore_ascii_case(name), *name)
                .clicked()
            {
                script.set_cvar_engine("lightingQuality", name);
            }
        }
        if current.eq_ignore_ascii_case(crate::cvars::LIGHTING_CUSTOM) {
            ui.weak(crate::cvars::LIGHTING_CUSTOM);
        }
    });
    ui.horizontal(|ui| {
        ui.label("Interior presets");
        for (label, values) in [
            // MONKEY (bake floor): the fourth value is `interiorBakeFloor`, and it rides the
            // presets because it is a room BRIGHTNESS INPUT exactly like the three beside it.
            // Leaving it out would make the presets change the RATIO between a fixture-lit room
            // and a fixture-starved one rather than the level of both: Bright would raise the
            // candles and leave the vestibule at the Default floor. (The `interiorGain` the
            // presets also set does scale it at pack time, so its own value composes with that
            // — which is why Dim asks for 0.08 and not for 0.12 halved twice.)
            ("Dim", Some(["0.5", "0.015", "0.08", "0.08"])),
            ("Default", None),
            ("Bright", Some(["1.0", "0.04", "0.16", "0.2"])),
        ] {
            if ui
                .small_button(label)
                .on_hover_text(
                    "Only gain, ambient, fill and bake floor; other settings stay as they are.",
                )
                .clicked()
            {
                if let Some(values) = values {
                    for (knob, value) in INTERIOR[1..5].iter().zip(values) {
                        script.set_cvar_engine(knob.name, value);
                    }
                } else {
                    reset(script, &INTERIOR[1..5]);
                }
            }
        }
    });
    if ui
        .small_button("Reset all")
        .on_hover_text("Restore registry defaults for every group in Lighting & shadows.")
        .clicked()
    {
        for (_, knobs) in GROUPS {
            reset(script, knobs);
        }
    }
    for (name, knobs) in GROUPS {
        ui.push_id(name, |ui| {
            ui.separator();
            ui.strong(*name);
            for knob in *knobs {
                row(ui, video, script, knob);
            }
            if ui.small_button("Reset group to defaults").clicked() {
                reset(script, knobs);
            }
        });
    }
}

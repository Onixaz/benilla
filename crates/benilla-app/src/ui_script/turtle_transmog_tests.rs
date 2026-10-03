//! Turtle's transmog window (`Turtle_TransmogUI`, on a Turtle patch chain only) opens off the
//! gossip menu whose greeting is `TRANSMOG_TRIGGER`: its `GOSSIP_SHOW` hides `GossipFrame` and
//! shows `TransmogFrame`, and `GOSSIP_CLOSED` restores the gossip frame.

use benilla_ui::script::{GossipMenu, ScriptValue, UiScript};

use crate::local_state::test_env::ENV_LOCK;

fn trigger_menu() -> GossipMenu {
    GossipMenu {
        greeting: "TRANSMOG_TRIGGER".into(),
        quests: Vec::new(),
        options: Vec::new(),
    }
}

/// The gossip open as `crate::ui_gossip::feed_gossip` delivers it, with `fires` `GOSSIP_SHOW`s.
fn open(s: &mut UiScript, fires: usize) {
    s.set_unit(
        "npc",
        Some(benilla_ui::script::UnitState {
            exists: true,
            name: Some("Fashionista".into()),
            ..Default::default()
        }),
    );
    s.set_gossip(Some(trigger_menu()));
    for _ in 0..fires {
        s.fire_event("GOSSIP_SHOW", vec![ScriptValue::Str("Fashionista".into())]);
    }
}

/// The close as the app delivers it: the Lua close intent clears the menu, then `GOSSIP_CLOSED`.
fn close_with_x(s: &mut UiScript) {
    s.run("TransmogFrameCloseButton:Click()").unwrap();
    let _ = s.take_gossip_close();
    s.set_gossip(None);
    s.fire_event("GOSSIP_CLOSED", vec![]);
}

fn shown(s: &UiScript) -> (bool, f32, String) {
    let transmog = s
        .eval::<bool>("return TransmogFrame:IsVisible() == 1")
        .unwrap();
    let alpha = s.eval::<f32>("return GossipFrame:GetAlpha()").unwrap();
    let pushable = s
        .eval::<String>("return tostring(UIPanelWindows.GossipFrame.pushable)")
        .unwrap();
    (transmog, alpha, pushable)
}

#[test]
fn transmog_opens_on_every_visit() {
    benilla_formats::wow_data_or_skip!();
    let _env = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    // The char-enum record the live load seats first: Turtle's file scope reads `UnitRace`.
    let (mut s, failures) = super::layer_tests::production_load_with_record(
        "transmog",
        benilla_ui::script::PlayerRecord {
            name: "Probesix".into(),
            race: Some(("Human".into(), "Human".into())),
            class: Some(("Mage".into(), "MAGE".into())),
            sex: 3,
        },
    );
    if !s.eval::<bool>("return TransmogFrame ~= nil").unwrap() {
        eprintln!("skip: not a Turtle chain");
        return;
    }
    let turtle: Vec<_> = failures.iter().filter(|f| f.contains("Transmog")).collect();
    assert!(
        turtle.is_empty(),
        "Turtle's transmog files load clean: {turtle:#?}"
    );
    // A second `GOSSIP_SHOW` for one open must not break the next visit either.
    for fires in [1usize, 2] {
        for visit in 1..=2 {
            open(&mut s, fires);
            let got = shown(&s);
            assert!(
                got.0,
                "visit {visit} (fires {fires}) shows TransmogFrame: {got:?}"
            );
            assert_eq!(got.1, 0.0, "the gossip frame is hidden behind it");
            close_with_x(&mut s);
        }
    }
}

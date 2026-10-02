//! The dressing room: the `DressUpModel` verbs (table `0x84f190`) and the intent queue behind them
//! and behind `SetUnit`/`RefreshUnit` on a dressing-room pane. The reference edits a clone of the
//! unit's model (`0x5059a0`, `0x504350`); here the app composes the look from the intents in
//! order, which matters: on a closed window `DressUpItem` resets, then tries on
//! (`DressUpFrame.lua:3-7`).

use mlua::{Lua, Table, Value};

use super::object::frame_handle_of;
use super::Model;
use crate::widget::{FrameHandle, FrameKind};

/// Registry key of the `DressUpModel` method table, its own three; the rest come by the chain.
pub(super) const REG_DRESSUPMODEL_METHODS: &str = "__benilla_dressupmodel_methods";

/// One queued dressing-room intent, applied in order.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum DressUpIntent {
    /// `SetUnit`, `RefreshUnit` and `Dress` (the Reset button): the reference's rebuild from the
    /// unit (`0x505b50`), dropping every substitution.
    Dress,
    /// `Undress()`: bodyslots `0..0xb` cleared, worn and tried-on alike; the hands keep what they
    /// hold (`0x504490`).
    Undress,
    /// `TryOn(item)`: the item in its `InventoryType`'s slot, with no class, level or proficiency
    /// check (`DressUpFrame.lua:2-16`).
    TryOn(u32),
    /// The window was hidden, so the booth empties. The reference keeps its state while hidden,
    /// but the next `DressUpItem` re-issues `SetUnit("player")`, so that state is never seen.
    Close,
}

impl super::UiScript {
    /// Drain the dressing rooms' intents with the pane each was made on, oldest first; the app
    /// applies them in order. Each `DressUpModel` widget is its own room: the reference clones a
    /// model per widget (`0x5059a0`).
    pub fn take_dressup_intents(&mut self) -> Vec<(FrameHandle, DressUpIntent)> {
        std::mem::take(&mut self.model_mut().dressup_intents)
    }
}

fn queue(lua: &Lua, pane: FrameHandle, intent: DressUpIntent) {
    lua.app_data_mut::<Model>()
        .expect("model app_data")
        .dressup_intents
        .push((pane, intent));
}

/// Queue the rebuild when `PlayerModel`'s `SetUnit`/`RefreshUnit` lands on a `DressUpModel`.
pub(super) fn redress_if_dressup(lua: &Lua, this: &Table) -> mlua::Result<()> {
    let h = frame_handle_of(lua, this)?;
    let is_dressup = lua
        .app_data_ref::<Model>()
        .expect("model app_data")
        .arena
        .frame(h)
        .is_some_and(|f| f.kind == FrameKind::DressUpModel);
    if is_dressup {
        queue(lua, h, DressUpIntent::Dress);
    }
    Ok(())
}

/// `trunc(tonumber(arg))`, the reference's `__ftol` (`0x40a2b0`) over `lua_tonumber`; anything
/// else is 0.
fn item_arg(v: &Value) -> i64 {
    match v {
        Value::Integer(i) => *i,
        Value::Number(n) => n.trunc() as i64,
        Value::String(s) => s
            .to_str()
            .ok()
            .and_then(|t| t.trim().parse::<f64>().ok())
            .map_or(0, |n| n.trunc() as i64),
        _ => 0,
    }
}

pub(super) fn install(lua: &Lua) -> mlua::Result<()> {
    let m = lua.create_table()?;

    // `0x504c00`.
    m.set(
        "Undress",
        lua.create_function(|lua, this: Table| {
            let h = frame_handle_of(lua, &this)?;
            queue(lua, h, DressUpIntent::Undress);
            Ok(())
        })?,
    )?;

    // `0x504cd0`, the Reset button (`DressUpFrame.xml:182`).
    m.set(
        "Dress",
        lua.create_function(|lua, this: Table| {
            let h = frame_handle_of(lua, &this)?;
            queue(lua, h, DressUpIntent::Dress);
            Ok(())
        })?,
    )?;

    // `0x504d90` calls `0x504540(itemId)`; item 0 (a non-number or a link) previews nothing, and
    // so does a negative id here.
    m.set(
        "TryOn",
        lua.create_function(|lua, (this, item): (Table, Value)| {
            let h = frame_handle_of(lua, &this)?;
            if let Ok(id) = u32::try_from(item_arg(&item)) {
                if id != 0 {
                    queue(lua, h, DressUpIntent::TryOn(id));
                }
            }
            Ok(())
        })?,
    )?;

    lua.set_named_registry_value(REG_DRESSUPMODEL_METHODS, m)
}

#[cfg(test)]
mod tests {
    use super::DressUpIntent;
    use crate::script::UiScript;

    fn room() -> UiScript {
        let mut s = UiScript::new().unwrap();
        s.set_screen_size(800.0, 600.0);
        s.run(r#"dm = CreateFrame("DressUpModel", "DM", UIParent)"#)
            .unwrap();
        s
    }

    #[test]
    fn a_dress_up_model_is_a_player_model_plus_three_and_the_chain_runs_one_way() {
        let s = room();
        s.run(r#"pm = CreateFrame("PlayerModel", "PMOnly", UIParent)"#)
            .unwrap();
        for verb in [
            "TryOn",
            "Dress",
            "Undress",
            "SetUnit",
            "RefreshUnit",
            "SetRotation",
            "SetCamera",
        ] {
            assert_eq!(
                s.eval::<String>(&format!("return type(DM.{verb})"))
                    .unwrap(),
                "function",
                "DressUpModel answers {verb}"
            );
        }
        for verb in ["TryOn", "Dress", "Undress"] {
            assert_eq!(
                s.eval::<String>(&format!("return type(PMOnly.{verb})"))
                    .unwrap(),
                "nil",
                "a PlayerModel must NOT answer {verb}"
            );
        }
        assert_eq!(
            s.eval::<String>("return DM:GetObjectType()").unwrap(),
            "DressUpModel"
        );
        for t in ["DressUpModel", "PlayerModel", "Model", "Frame", "Region"] {
            assert!(
                s.eval::<bool>(&format!("return DM:IsObjectType(\"{t}\") == 1"))
                    .unwrap(),
                "IsObjectType({t})"
            );
        }
    }

    /// The stock `DressUpItemLink` passes the link's digits: a digit string is an id, a link is 0.
    #[test]
    fn the_verbs_queue_intents_in_order_and_try_on_reads_its_argument_like_the_client() {
        let mut s = room();
        s.run(
            r#"
            DM:SetUnit("player")
            DM:TryOn("117")
            DM:TryOn(1234.7)
            DM:Undress()
            DM:RefreshUnit()
            DM:Dress()
            DM:TryOn("|cffffffff|Hitem:117|h[Tough Jerky]|h|r")
            DM:TryOn(0)
            DM:TryOn(nil)
            DM:TryOn(-5)
            "#,
        )
        .unwrap();
        let taken = s.take_dressup_intents();
        let dm = taken.first().expect("an intent").0;
        assert!(taken.iter().all(|(h, _)| *h == dm), "every intent names DM");
        assert_eq!(
            taken.into_iter().map(|(_, i)| i).collect::<Vec<_>>(),
            vec![
                DressUpIntent::Dress,
                DressUpIntent::TryOn(117),
                DressUpIntent::TryOn(1234),
                DressUpIntent::Undress,
                DressUpIntent::Dress,
                DressUpIntent::Dress,
            ]
        );
        // The same two PlayerModel verbs on a paper-doll pane queue nothing.
        s.run(r#"pm = CreateFrame("PlayerModel", "PMOnly", UIParent) pm:SetUnit("player") pm:RefreshUnit()"#)
            .unwrap();
        assert!(s.take_dressup_intents().is_empty());
        // Its yaw is its own facing, read by name like every other pane's.
        s.run("DM:SetRotation(0.61)").unwrap();
        assert!((s.model_pane_facing("DM") - 0.61).abs() < 1e-6);
    }

    /// Two widgets are two rooms: an intent names its own pane, as each widget clones its own
    /// model (`0x5059a0`); Turtle's transmog tiles each try on a different item.
    #[test]
    fn each_dress_up_model_queues_for_itself() {
        let mut s = room();
        s.run(
            r#"
            dm2 = CreateFrame("DressUpModel", "DM2", UIParent)
            DM:TryOn(1) dm2:TryOn(2) DM:Undress()
            "#,
        )
        .unwrap();
        let taken = s.take_dressup_intents();
        assert_eq!(taken.len(), 3);
        assert_eq!(taken[0].0, taken[2].0, "DM's two intents name one pane");
        assert_ne!(taken[0].0, taken[1].0, "DM2's names another");
        assert_eq!(s.frame_name(taken[1].0).as_deref(), Some("DM2"));
    }

    /// The camera freezes at the unit's ready edge through the root as it stands (`0x505890`):
    /// a `SetPosition` before `SetUnit` is in the camera, one after moves only the body.
    #[test]
    fn set_unit_freezes_the_camera_root() {
        let s = room();
        s.run(
            r#"
            DM:SetModelScale(4) DM:SetPosition(1, 2, 3) DM:SetUnit("player")
            DM:SetPosition(5.8, 0.1, -1.2)
            "#,
        )
        .unwrap();
        let m = s.model_pane("DM").unwrap();
        assert_eq!(m.camera_root, Some(((1.0, 2.0, 3.0), 4.0)));
        assert_eq!(m.position, (5.8, 0.1, -1.2));
    }
}

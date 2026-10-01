//! Drives [`benilla_assets::WorldAssets`] from the client: each system needs a client piece (the
//! light buffer, `MapChange`, the art scope), so it lives here rather than in `benilla-assets`.

use bevy::prelude::*;
use bevy::render::renderer::RenderDevice;

use crate::art_scope::{ArtScope, ArtSlot};
use benilla_assets::{AssetSet, RenderConfig, WorldAssets};
use benilla_formats::open_chain;

/// Opens the patch chain at startup and inserts the shared [`WorldAssets`] and [`RenderConfig`].
pub(crate) struct AssetPlugin;

impl Plugin for AssetPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Startup, open_world_assets.in_set(AssetSet::Open));
        app.add_systems(Update, (evict_world_art, scope_world_art));
        // MONKEY (torch shadows Phase 3A): the shared torch depth image + table buffer's extract
        // plugins and per-frame table upload — always on, like the light buffer's, because every
        // model material binds them (`open_world_assets` creates them below).
        crate::static_gx::register_torch_shared(app);
    }
}

/// Clears `textures` and `model_materials` on a map change, or they pin every map's art forever.
/// The UI sprite caches stay: they are global, and their negative entries stop per-frame re-walks.
fn evict_world_art(
    mut changes: MessageReader<crate::world_map::MapChange>,
    assets: Option<ResMut<WorldAssets>>,
) {
    if changes.is_empty() {
        return;
    }
    changes.clear();
    if let Some(mut a) = assets {
        a.textures.clear();
        a.model_materials.clear();
    }
}

/// Expires the world-art caches by distance within a map: a cached material pins the decoded BLP it
/// samples, so `textures` is what frees VRAM. The UI sprite caches stay, as on a map change.
fn scope_world_art(mut scope: ArtScope, assets: Option<ResMut<WorldAssets>>) {
    if let Some(mut a) = assets {
        scope.apply(&mut a.model_materials, ArtSlot::ClutterMats);
        scope.apply(&mut a.textures, ArtSlot::Textures);
    }
}

/// Open the vanilla patch chain from wherever the install is ([`benilla_formats::wow_data`] —
/// `$WOW_DATA`, the project folder on a dev build, else beside the binary; decision 1175) and
/// insert the shared [`WorldAssets`] (chain + dedup caches) + [`RenderConfig`]. If the client data
/// can't be found or opened, `WorldAssets` is simply absent and downstream startup falls back to
/// an empty free-fly scene.
fn open_world_assets(
    mut commands: Commands,
    device: Res<RenderDevice>,
    mut images: ResMut<Assets<Image>>,
) {
    // The one shared global-light buffer, created here (RenderDevice is live by Startup) so it exists
    // before any material is built. Inserted FIRST — ahead of the install lookup, so no early return
    // below can skip it: it is cloned into `WorldAssets` (for model materials) and read as the
    // `SharedLightBuffer` resource by the terrain streamer, the model/particle lanes and the
    // render-world upload. Always present — even with no client data — so the render upload has a
    // target; harmless if unused. It used to be created *after* the lookup, so a client that found no
    // install had no buffer at all and `particles::model::update_model_particles` — a hard
    // `Res<SharedLightBuffer>` — could not validate (decision 1451).
    let shared_light = crate::lighting::new_shared_light_buffer(&device);
    let light_buf = shared_light.0.clone();
    commands.insert_resource(shared_light);
    // MONKEY (torch shadows Phase 3A): the shared torch depth image + table buffer, on the same
    // always-present rule and for the same reason — every model material binds them, and a
    // material built against a missing image never gets a bind group (every model blanks).
    let (torch_image, torch_buffer) = crate::static_gx::new_torch_shared(&device, &mut images);
    let torch = benilla_assets::materials::TorchBinds {
        depth: torch_image.0.clone(),
        table: torch_buffer.0.clone(),
    };
    commands.insert_resource(torch_image);
    commands.insert_resource(torch_buffer);
    let Some(data) = benilla_formats::wow_data() else {
        warn!(
            "no WoW install found — looked in {:?}; starting with no world",
            benilla_formats::candidates()
        );
        return;
    };
    // Stale tiles released per frame: 1 outpaces even boosted free-fly's stale row a second.
    let unload_budget = std::env::var("WOW_TILE_UNLOAD")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1);
    commands.insert_resource(RenderConfig { unload_budget });

    match open_chain(&data) {
        Ok(chain) => commands.insert_resource(WorldAssets::open(chain, light_buf, torch)),
        Err(e) => error!("failed to open client data: {e:#}"),
    }
}

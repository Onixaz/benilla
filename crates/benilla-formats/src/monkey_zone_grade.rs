//! MONKEY (post): `MonkeyZoneGrade.dbc`, the optional area-to-day/night LUT table.

use anyhow::{Context, Result};
use benilla_dbc::{FieldType, Schema, SchemaField};

use crate::dbc::{f32_at, parse, str_at, u32_at};
use crate::Chain;

const TABLE: &str = "DBFilesClient\\MonkeyZoneGrade.dbc";

/// One authored zone grade. Missing files and absent area rows are handled by the caller as the
/// identity cube, so this type only represents complete rows.
#[derive(Clone, Debug, PartialEq)]
pub struct MonkeyZoneGrade {
    pub id: u32,
    pub area_id: u32,
    pub day_lut: String,
    pub night_lut: String,
    pub strength: f32,
}

#[derive(Clone, Debug, Default)]
pub struct MonkeyZoneGrades {
    rows: Vec<MonkeyZoneGrade>,
}

impl MonkeyZoneGrades {
    pub fn for_area(&self, area_id: u32) -> Option<&MonkeyZoneGrade> {
        self.rows.iter().find(|row| row.area_id == area_id)
    }

    pub fn rows(&self) -> impl Iterator<Item = &MonkeyZoneGrade> {
        self.rows.iter()
    }
}

/// Read the five-column extension table through the same typed DBC parser as the stock tables.
pub fn load_monkey_zone_grades(chain: &mut Chain) -> Result<MonkeyZoneGrades> {
    let bytes = chain
        .read_file(TABLE)
        .context("reading MonkeyZoneGrade.dbc")?;
    parse_monkey_zone_grades(&bytes)
}

/// Parse an in-memory extension table through the typed DBC reader. This is used for the loose
/// capture patch; the MPQ-backed path above and this path therefore share the exact schema.
pub fn parse_monkey_zone_grades(bytes: &[u8]) -> Result<MonkeyZoneGrades> {
    let mut schema = Schema::new("MonkeyZoneGrade");
    schema.add_field(SchemaField::new("ID", FieldType::UInt32));
    schema.add_field(SchemaField::new("AreaID", FieldType::UInt32));
    schema.add_field(SchemaField::new("DayLUT", FieldType::String));
    schema.add_field(SchemaField::new("NightLUT", FieldType::String));
    schema.add_field(SchemaField::new("Strength", FieldType::Float32));
    let records = parse(bytes, schema, "MonkeyZoneGrade.dbc")?;
    let rows = records
        .records()
        .iter()
        .filter_map(|record| {
            Some(MonkeyZoneGrade {
                id: u32_at(record, 0)?,
                area_id: u32_at(record, 1)?,
                day_lut: str_at(&records, record, 2)?,
                night_lut: str_at(&records, record, 3)?,
                strength: f32_at(record, 4).unwrap_or(0.0).clamp(0.0, 1.0),
            })
        })
        .collect();
    Ok(MonkeyZoneGrades { rows })
}

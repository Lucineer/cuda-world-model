/*!
# cuda-world-model

Agent's model of the external world.

The world model is where perception meets prediction. An agent that
only reacts to what it sees has no foresight. An agent with a world
model can PREDICT what happens next.

- Spatial layout (grid-based environment representation)
- Object permanence (objects exist even when not perceived)
- State tracking (what's happening in the world)
- Prediction (what will happen next)
- Updating (correcting model when predictions fail)
*/

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// A point in 2D space
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Position { pub x: f64, pub y: f64 }
impl Position { pub fn new(x: f64, y: f64) -> Self { Position { x, y } } pub fn distance_to(&self, other: &Position) -> f64 { ((self.x - other.x).powi(2) + (self.y - other.y).powi(2)).sqrt() } pub fn manhattan(&self, other: &Position) -> f64 { (self.x - other.x).abs() + (self.y - other.y).abs() } }

/// A region in space
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Region { pub id: String, pub center: Position, pub radius: f64, pub kind: RegionKind, pub properties: HashMap<String, f64>, pub confidence: f64 }

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum RegionKind { Open, Obstacle, Hazard, Resource, Goal, Unknown, Transit }

/// A world object with permanence
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct WorldObject {
    pub id: String,
    pub kind: String,
    pub position: Position,
    pub last_seen: u64,
    pub expected_position: Position,
    pub velocity: (f64, f64),
    pub confidence: f64,
    pub permanence: f64,  // belief this object still exists [0, 1]
    pub properties: HashMap<String, f64>,
}

impl WorldObject {
    pub fn update_observation(&mut self, pos: Position, timestamp: u64) {
        let dt = (timestamp.saturating_sub(self.last_seen) as f64 / 1000.0).max(0.001);
        self.velocity = ((pos.x - self.position.x) / dt, (pos.y - self.position.y) / dt);
        self.position = pos;
        self.expected_position = pos;
        self.last_seen = timestamp;
        self.confidence = (self.confidence + 0.2).min(1.0);
        self.permanence = (self.permanence + 0.1).min(1.0);
    }

    pub fn predict_position(&self, now: u64) -> Position {
        let dt = (now.saturating_sub(self.last_seen) as f64 / 1000.0).max(0.0);
        Position { x: self.expected_position.x + self.velocity.0 * dt, y: self.expected_position.y + self.velocity.1 * dt }
    }

    pub fn decay(&mut self, now: u64, half_life_ms: u64) {
        let age = now.saturating_sub(self.last_seen);
        self.confidence *= 0.5_f64.powf(age as f64 / half_life_ms as f64);
        self.permanence *= 0.5_f64.powf(age as f64 / (half_life_ms as f64 * 3.0)); // permanence decays slower
    }
}

/// An event in the world
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct WorldEvent {
    pub id: String,
    pub kind: String,
    pub position: Position,
    pub timestamp: u64,
    pub importance: f64,
    pub description: String,
}

/// The world model
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct WorldModel {
    pub objects: HashMap<String, WorldObject>,
    pub regions: Vec<Region>,
    pub events: Vec<WorldEvent>,
    pub agent_position: Position,
    pub bounds: Option<(f64, f64, f64, f64)>, // x_min, y_min, x_max, y_max
    pub next_event_id: u64,
    pub confidence: f64,
    pub prediction_accuracy: f64,
}

impl WorldModel {
    pub fn new() -> Self { WorldModel { objects: HashMap::new(), regions: vec![], events: vec![], agent_position: Position::new(0.0, 0.0), bounds: None, next_event_id: 1, confidence: 0.5, prediction_accuracy: 0.5 } }

    pub fn add_object(&mut self, obj: WorldObject) { self.objects.insert(obj.id.clone(), obj); }

    pub fn observe_object(&mut self, id: &str, pos: Position, timestamp: u64) {
        if let Some(obj) = self.objects.get_mut(id) { obj.update_observation(pos, timestamp); }
    }

    /// Predict all object positions
    pub fn predict(&self, now: u64) -> HashMap<String, Position> {
        self.objects.iter().map(|(id, obj)| (id.clone(), obj.predict_position(now))).collect()
    }

    /// Check prediction accuracy (compare predicted vs actual)
    pub fn check_prediction(&mut self, obj_id: &str, actual_pos: Position) -> f64 {
        let predicted = match self.objects.get(obj_id) { Some(o) => o.predict_position(now()), None => return 0.0 };
        let error = predicted.distance_to(&actual_pos);
        // Convert error to accuracy (lower error = higher accuracy)
        let accuracy = (1.0 / (1.0 + error)).max(0.0);
        self.prediction_accuracy = self.prediction_accuracy * 0.8 + accuracy * 0.2; // EMA
        accuracy
    }

    /// Find objects near a position
    pub fn objects_near(&self, pos: &Position, radius: f64) -> Vec<&WorldObject> {
        self.objects.values().filter(|o| o.position.distance_to(pos) <= radius).collect()
    }

    /// Find objects by kind
    pub fn objects_by_kind(&self, kind: &str) -> Vec<&WorldObject> { self.objects.values().filter(|o| o.kind == kind).collect() }

    /// Check if position is in any hazardous region
    pub fn is_hazardous(&self, pos: &Position) -> bool {
        self.regions.iter().any(|r| r.kind == RegionKind::Hazard && r.center.distance_to(pos) <= r.radius)
    }

    /// Check if position is accessible (not in obstacle)
    pub fn is_accessible(&self, pos: &Position) -> bool {
        !self.regions.iter().any(|r| r.kind == RegionKind::Obstacle && r.center.distance_to(pos) <= r.radius)
    }

    /// Record a world event
    pub fn record_event(&mut self, kind: &str, pos: Position, importance: f64, desc: &str) {
        let evt = WorldEvent { id: format!("evt_{}", self.next_event_id), kind: kind.to_string(), position: pos, timestamp: now(), importance, description: desc.to_string() };
        self.next_event_id += 1;
        if self.events.len() > 100 { self.events.remove(0); }
        self.events.push(evt);
    }

    /// Decay old object confidences
    pub fn decay(&mut self, now: u64) {
        for obj in self.objects.values_mut() { obj.decay(now, 30_000); }
        self.objects.retain(|_, o| o.permanence > 0.05); // remove objects no longer believed to exist
    }

    /// Update model confidence based on prediction accuracy
    pub fn update_confidence(&mut self) {
        self.confidence = self.confidence * 0.9 + self.prediction_accuracy * 0.1;
    }

    /// Summary
    pub fn summary(&self) -> String {
        format!("WorldModel: {} objects, {} regions, {} events, prediction_acc={:.2}, model_confidence={:.2}",
            self.objects.len(), self.regions.len(), self.events.len(), self.prediction_accuracy, self.confidence)
    }
}

fn now() -> u64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap_or_default().as_millis() as u64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_position_distance() {
        let a = Position::new(0.0, 0.0);
        let b = Position::new(3.0, 4.0);
        assert!((a.distance_to(&b) - 5.0).abs() < 0.01);
    }

    #[test]
    fn test_add_observe_object() {
        let mut wm = WorldModel::new();
        wm.add_object(WorldObject { id: "wall".into(), kind: "obstacle".into(), position: Position::new(5.0, 0.0), last_seen: 0, expected_position: Position::new(5.0, 0.0), velocity: (0.0, 0.0), confidence: 0.8, permanence: 0.9, properties: HashMap::new() });
        wm.observe_object("wall", Position::new(5.0, 0.0), 1000);
        assert_eq!(wm.objects["wall"].confidence, 1.0); // capped
    }

    #[test]
    fn test_predict_position() {
        let mut wm = WorldModel::new();
        let mut obj = WorldObject { id: "ball".into(), kind: "moving".into(), position: Position::new(0.0, 0.0), last_seen: 0, expected_position: Position::new(0.0, 0.0), velocity: (1.0, 0.0), confidence: 0.8, permanence: 0.9, properties: HashMap::new() };
        obj.last_seen = 1000;
        wm.add_object(obj);
        let predicted = wm.objects["ball"].predict_position(3000); // 2 seconds later
        assert!((predicted.x - 2.0).abs() < 0.01);
    }

    #[test]
    fn test_check_prediction() {
        let mut wm = WorldModel::new();
        wm.add_object(WorldObject { id: "x".into(), kind: "y".into(), position: Position::new(0.0, 0.0), last_seen: now(), expected_position: Position::new(0.0, 0.0), velocity: (0.0, 0.0), confidence: 0.8, permanence: 0.9, properties: HashMap::new() });
        let accuracy = wm.check_prediction("x", Position::new(0.1, 0.0)); // very close
        assert!(accuracy > 0.8);
    }

    #[test]
    fn test_objects_near() {
        let mut wm = WorldModel::new();
        wm.add_object(WorldObject { id: "a".into(), kind: "x".into(), position: Position::new(0.0, 0.0), last_seen: 0, expected_position: Position::new(0.0, 0.0), velocity: (0.0, 0.0), confidence: 0.8, permanence: 0.9, properties: HashMap::new() });
        wm.add_object(WorldObject { id: "b".into(), kind: "x".into(), position: Position::new(100.0, 100.0), last_seen: 0, expected_position: Position::new(100.0, 100.0), velocity: (0.0, 0.0), confidence: 0.8, permanence: 0.9, properties: HashMap::new() });
        let near = wm.objects_near(&Position::new(1.0, 1.0), 5.0);
        assert_eq!(near.len(), 1);
    }

    #[test]
    fn test_hazard_detection() {
        let mut wm = WorldModel::new();
        wm.regions.push(Region { id: "hazard1".into(), center: Position::new(5.0, 5.0), radius: 2.0, kind: RegionKind::Hazard, properties: HashMap::new(), confidence: 0.9 });
        assert!(wm.is_hazardous(&Position::new(6.0, 5.0)));
        assert!(!wm.is_hazardous(&Position::new(0.0, 0.0)));
    }

    #[test]
    fn test_obstacle_blocks() {
        let mut wm = WorldModel::new();
        wm.regions.push(Region { id: "wall".into(), center: Position::new(5.0, 5.0), radius: 1.0, kind: RegionKind::Obstacle, properties: HashMap::new(), confidence: 0.9 });
        assert!(!wm.is_accessible(&Position::new(5.0, 5.0)));
        assert!(wm.is_accessible(&Position::new(0.0, 0.0)));
    }

    #[test]
    fn test_decay_removes_objects() {
        let mut wm = WorldModel::new();
        wm.add_object(WorldObject { id: "ghost".into(), kind: "x".into(), position: Position::new(0.0, 0.0), last_seen: 0, expected_position: Position::new(0.0, 0.0), velocity: (0.0, 0.0), confidence: 0.01, permanence: 0.01, properties: HashMap::new() });
        wm.decay(now() + 100_000_000);
        assert!(wm.objects.get("ghost").is_none());
    }

    #[test]
    fn test_record_event() {
        let mut wm = WorldModel::new();
        wm.record_event("collision", Position::new(1.0, 1.0), 0.9, "hit wall");
        assert_eq!(wm.events.len(), 1);
    }

    #[test]
    fn test_objects_by_kind() {
        let mut wm = WorldModel::new();
        wm.add_object(WorldObject { id: "a".into(), kind: "wall".into(), position: Position::new(0.0, 0.0), last_seen: 0, expected_position: Position::new(0.0, 0.0), velocity: (0.0, 0.0), confidence: 0.8, permanence: 0.9, properties: HashMap::new() });
        wm.add_object(WorldObject { id: "b".into(), kind: "food".into(), position: Position::new(1.0, 1.0), last_seen: 0, expected_position: Position::new(1.0, 1.0), velocity: (0.0, 0.0), confidence: 0.8, permanence: 0.9, properties: HashMap::new() });
        assert_eq!(wm.objects_by_kind("wall").len(), 1);
    }

    #[test]
    fn test_summary() {
        let wm = WorldModel::new();
        let s = wm.summary();
        assert!(s.contains("0 objects"));
    }
}

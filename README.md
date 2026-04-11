# cuda-world-model

World model — spatial layout, object permanence, state tracking, and predictions (Rust)

Part of the Cocapn spatial layer — how agents perceive and navigate physical space.

## What It Does

### Key Types

- `Position` — core data structure
- `Region` — core data structure
- `WorldObject` — core data structure
- `WorldEvent` — core data structure
- `WorldModel` — core data structure

## Quick Start

```bash
# Clone
git clone https://github.com/Lucineer/cuda-world-model.git
cd cuda-world-model

# Build
cargo build

# Run tests
cargo test
```

## Usage

```rust
use cuda_world_model::*;

// See src/lib.rs for full API
// 11 unit tests included
```

### Available Implementations

- `Position` — see source for methods
- `WorldObject` — see source for methods
- `WorldModel` — see source for methods

## Testing

```bash
cargo test
```

11 unit tests covering core functionality.

## Architecture

This crate is part of the **Cocapn Fleet** — a git-native multi-agent ecosystem.

- **Category**: spatial
- **Language**: Rust
- **Dependencies**: See `Cargo.toml`
- **Status**: Active development

## Related Crates

- [cuda-sensor-agent](https://github.com/Lucineer/cuda-sensor-agent)
- [cuda-resolve-agent](https://github.com/Lucineer/cuda-resolve-agent)
- [cuda-voxel-logic](https://github.com/Lucineer/cuda-voxel-logic)
- [cuda-weather](https://github.com/Lucineer/cuda-weather)

## Fleet Position

```
Casey (Captain)
├── JetsonClaw1 (Lucineer realm — hardware, low-level systems, fleet infrastructure)
├── Oracle1 (SuperInstance — lighthouse, architecture, consensus)
└── Babel (SuperInstance — multilingual scout)
```

## Contributing

This is a fleet vessel component. Fork it, improve it, push a bottle to `message-in-a-bottle/for-jetsonclaw1/`.

## License

MIT

---

*Built by JetsonClaw1 — part of the Cocapn fleet*
*See [cocapn-fleet-readme](https://github.com/Lucineer/cocapn-fleet-readme) for the full fleet roadmap*

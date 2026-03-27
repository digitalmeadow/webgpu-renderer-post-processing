# @digitalmeadow/webgpu-renderer-post-processing

Post-processing effects for WebGPURenderer

## Installation

```bash
npm install @digitalmeadow/webgpu-renderer-post-processing
```

## Features

- Single dependency on @digitalmeadow/webgpu-renderer

## Usage

```typescript
import { Renderer } from "@digitalmeadow/webgpu-renderer";
import { BloomPass } from "@digitalmeadow/webgpu-renderer-post-processing";

  const renderer = new Renderer();
  await renderer.init();

  const bloomPass = new BloomPass(renderer.getDevice(), {
    threshold: 0.95,
    radius: 20,
    intensity: 10.0,
    iterations: 2,
  });

  renderer.addPostPass(bloomPass);
```

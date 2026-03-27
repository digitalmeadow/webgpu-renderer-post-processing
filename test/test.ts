import {
  Renderer,
  World,
  Scene,
  DirectionalLight,
  Camera,
  FlyControls,
  Time,
  Vec3,
  AnimationController,
  ConvexHull,
} from "@digitalmeadow/webgpu-renderer";

async function main() {
  const canvas = document.getElementById("gpu-canvas") as HTMLCanvasElement;
  if (!canvas) {
    console.error("Canvas not found");
    return;
  }

  const renderer = new Renderer(canvas);
  await renderer.init();

  const camera = new Camera(
    renderer.getDevice(),
    undefined,
    undefined,
    undefined,
    undefined,
    canvas.clientWidth / canvas.clientHeight,
  );
  camera.position.set(0, 1, 5);

  function resize() {
    const rect = canvas.getBoundingClientRect();
    renderer.resize(rect.width, rect.height);
    camera.resize(rect.width, rect.height);
  }

  window.addEventListener("resize", resize);
  resize();

  const controls = new FlyControls(canvas, camera);

  const world = new World();
  const scene = new Scene("MainScene");

  const light = new DirectionalLight("Sun");
  light.transform.setPosition(5, 10, 5);
  light.transform.lookAt(new Vec3(0, 0, 0));
  light.intensity = 1.0;
  scene.add(light);

  const time = new Time();

  function render() {
    time.update();
    controls.update(time.delta);
    renderer.render(world, camera, time);
    requestAnimationFrame(render);
  }

  requestAnimationFrame(render);
}

main().catch(console.error);

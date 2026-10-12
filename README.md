# Gaussian Studio

A desktop app for Linux and macOS that turns photos, videos, or both into a Gaussian
splat, using a GPU rented from [RunPod](https://www.runpod.io) for each job.

- Pick a folder on your computer or in Google Drive, and untick anything to leave out.
  Photos and videos together become one splat.
- The app finds a GPU within your price limit, uploads the input, trains the splat with
  [Spirula Studio](https://github.com/harry7557558/spirula-studio) or
  [Nerfstudio](https://docs.nerf.studio)'s Splatfacto, and downloads the `.ply` result.
- The GPU is shut down when the job finishes. If the app is closed mid-job, it reconnects
  when you reopen it.

**[Setup guide →](SETUP.md)** (about 15 minutes: install, RunPod account, optional Google Drive)

Taking good photos for a splat: open [capture_guide.html](capture_guide.html) in a browser.

## Quick start

```bash
git clone https://github.com/ipatedavid/gaussian-studio.git gaussian-studio
cd gaussian-studio
./install.sh
```

Then open **Gaussian Studio** from your desktop and paste your RunPod API key in
**Settings → Connection**.

## How it works

`desktop/` is the PySide6 app. It creates a pod through RunPod's API, connects over SSH, and
copies `worker/pipeline.py` and the input to the pod. Then it streams the training log and
preview, and copies the results back. Your API key, SSH key and Drive key stay on your
computer. The pod only gets a short-lived, read-only Drive token when a job reads from Drive.

Tests: `QT_QPA_PLATFORM=offscreen python3 -m unittest discover -s tests`

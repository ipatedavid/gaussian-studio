# Setting up Gaussian Studio

Gaussian Studio turns photos, videos, or both into a Gaussian splat. It rents a GPU on
[RunPod](https://www.runpod.io) for each job, sends the photos to it, trains the splat and
downloads the result to your computer. The GPU is shut down when the job ends, so you only pay
while a job runs (usually well under a dollar or two for a scene).

It runs on **Linux** (Ubuntu or similar) and **macOS**.

Setup takes about 15 minutes:

1. [Install the app](#1-install-the-app)
2. [Set up RunPod](#2-set-up-runpod)
3. [Optional: Google Drive](#3-optional-google-drive)

---

## 1. Install the app

### What you need first

**Linux (Ubuntu/Debian):** open a terminal and run

```bash
sudo apt install git python3 python3-venv ffmpeg libxcb-cursor0
```

**macOS:** install [Homebrew](https://brew.sh) if you don't have it, then run

```bash
brew install git python@3.12 ffmpeg
```

`ffmpeg` is only needed if you'll use videos. Photos work without it.

### Download and install

```bash
git clone https://github.com/ipatedavid/gaussian-studio.git gaussian-studio
cd gaussian-studio
./install.sh
```

The installer:

- installs the app's Python packages into a private folder (`desktop/.venv`), so nothing else
  on your computer changes;
- creates an SSH key at `~/.ssh/id_ed25519` if you don't already have one. The app uses it to
  talk to the GPU;
- adds a **Gaussian Studio** icon to your desktop and app menu. On a Mac, it's also in
  `~/Applications`.

If it made a new SSH key, it prints one line starting with `ssh-ed25519`. You'll paste that
into RunPod in the next step. On a Mac it's already on your clipboard.

> **Mac:** the first time you open the app, macOS may ask to allow it. If it says it can't be
> opened, right-click the icon → **Open** → **Open**.

### Updating

```bash
cd gaussian-studio
git pull
./install.sh
```

Your keys and settings are kept.

---

## 2. Set up RunPod

1. Create an account at [runpod.io](https://www.runpod.io) and add some credit under
   **Billing**. $10 is plenty to start; RunPod is prepaid, so you can't be charged more than
   you put in.
2. **Add your SSH key.** Go to **Settings → SSH Public Keys** and paste the `ssh-ed25519 …`
   line from the installer. If you already had a key, print it with:

   ```bash
   cat ~/.ssh/id_ed25519.pub
   ```

   If other keys are already in that box, put yours on a new line below them.
3. **Create an API key.** Still in **Settings**, go to **API Keys → Create API Key**. Give it
   a name like *Gaussian Studio*, choose **All** permissions, and copy the key. RunPod only
   shows it once.
4. **Paste it into the app.** Open Gaussian Studio → **Settings** → **Connection** → paste it
   in **RunPod API key** → **Save**.

The key is saved in `desktop/.env` inside the app folder, readable only by your user. Don't
share that file. Anyone with the key can spend your RunPod credit.

You're ready: on **New job**, choose **Local** and press **Choose folder**. The app lists the
folder's photos and videos with ticks; untick anything to leave out. Photos and videos together
become one splat. The app finds a GPU within your price limit, then press **Start**.

> **If the app closes unexpectedly** while a job runs, open it again: it reconnects to the
> running GPU. If you're ever unsure, check **Pods** on the RunPod website. A pod named
> `gaussian-…` that is still running is still being billed, and you can stop it there.

---

## 3. Optional: Google Drive

If your photos are in Google Drive, the app can read them from there instead of your
computer. The photos then go straight from Drive to the GPU, which is much faster than
uploading them from home.

For this, you create a **service account**: a separate Google identity that only you control.
You share your photo folders with it, and the app uses its key file to read them. It can only
see folders you share with it, and only read them.

It's free, and you only do this once.

### Create the service account

1. Go to [console.cloud.google.com](https://console.cloud.google.com) and sign in with your
   Google account. Accept the terms if it's your first time.
2. **Create a project:** click the project picker at the top → **New project** → name it, for
   example `gaussian-studio` → **Create**. Make sure the new project is selected at the top
   before you go on.
3. **Turn on the Drive API:** search the top bar for **Google Drive API**, open it and click
   **Enable**.
4. **Create the service account:** search for **Service Accounts** (under *IAM & Admin*) →
   **Create service account**. Give it a name like `drive-reader` → **Create and continue**.
   Skip the optional *Permissions* and *Principals* steps → **Done**.
5. **Download its key:** click the new service account → **Keys** tab → **Add key** →
   **Create new key** → **JSON** → **Create**. A `.json` file downloads.

   > If Google says key creation is disabled by an organization policy, your account belongs
   > to a company or school. Use a personal Gmail account instead.

Treat that `.json` file like a password: don't email it or put it in a shared folder.

### Add it to the app

Open Gaussian Studio → **Settings** → **Connection** → **Google Drive key** → **Choose file**
→ pick the `.json` file → **Save**.

The app copies it into its own folder (`desktop/service-account.json`). After that, you can
delete the downloaded copy.

Below the key, Settings shows an address like
`drive-reader@gaussian-studio-123456.iam.gserviceaccount.com`. That's the service account.
You'll share your folders with it.

### Share your photo folders

In [Google Drive](https://drive.google.com), right-click a folder of photos or videos (or a folder that
contains them) → **Share** → paste the service account's address → set it to **Viewer**. Turn
off **Notify people**, then click **Share**.

Sharing a parent folder shares everything inside it, so you can share one main folder once and
keep adding scenes to it.

### Use it

On **New job**, choose **Google Drive**, press **Choose folder** and pick the folder. Folders you
shared appear under **Shared with you**. Its photos and videos are listed with ticks, as for a
local folder.

---

## Troubleshooting

| Problem | Fix |
| --- | --- |
| The app doesn't open on Linux | Run `sudo apt install libxcb-cursor0`, or start it from a terminal to see the error: `cd gaussian-studio/desktop && .venv/bin/python main.py` |
| "Set up credentials" stays at the top | Check the RunPod key in **Settings**, and that both `~/.ssh/id_ed25519` and `~/.ssh/id_ed25519.pub` exist |
| No GPUs available | Raise the max price or lower the minimum VRAM, or try again later; stock changes all the time |
| The job fails while connecting | Make sure your public key is in RunPod's **SSH Public Keys** box. In **Settings → Pods & cost**, choose **Secure Cloud only** |
| Drive folder doesn't appear | Check it's shared with the service account's address, not your own |

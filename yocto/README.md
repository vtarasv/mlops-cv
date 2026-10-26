# Edge image (Yocto)

A headless Yocto 6.0 "wrynose" image for the Raspberry Pi 5, reachable as `root@raspberrypi5.local`
by ssh key over Ethernet and Wi-Fi. `kas/pi5.yml` pins the layers, `meta-mlops-cv/` holds the
image recipe.

## Prerequisites

- Docker (rootful) and [`kas-container`](https://github.com/siemens/kas/blob/5.5/kas-container)
  5.5 on `PATH`; it runs the build in `ghcr.io/siemens/kas/kas:5.5`, so the host distro doesn't matter.
- About 140 GB of disk and a few hours for the first build. Yocto asks for 32 GB of RAM;
  `kas/pi5.yml` caps `BB_NUMBER_THREADS`/`PARALLEL_MAKE` at 12 so a 30 GB host doesn't run out.

## Configure

```bash
cp kas/local.yml.example kas/local.yml   # gitignored
```

Fill in the Wi-Fi SSID/PSK and the root ssh public key.

## Build

Run from `yocto/`. Everything except the layer sources lives outside the repo:

```bash
export KAS_WORK_DIR=~/yocto/mlops-cv-pi5   # layer checkouts and build/ (tmp, buildhistory)
export DL_DIR=~/yocto/downloads             # shared across Yocto projects
export SSTATE_DIR=~/yocto/sstate-cache      # shared across Yocto projects
kas-container build kas/pi5.yml:kas/local.yml
```

The image lands in `$KAS_WORK_DIR/build/tmp/deploy/images/raspberrypi5/`. A change to
`local.yml` rebuilds in minutes from sstate.

## Flash

```bash
cd $KAS_WORK_DIR/build/tmp/deploy/images/raspberrypi5
sudo bmaptool copy mlops-cv-edge-image-raspberrypi5.rootfs.wic.bz2 /dev/sdX
# or: bzcat mlops-cv-edge-image-raspberrypi5.rootfs.wic.bz2 | sudo dd of=/dev/sdX bs=4M conv=fsync status=progress
```

## First boot

```bash
ssh root@raspberrypi5.local
```

Password logins are refused. Using the device (bench, live view) is in [docs/edge.md](../docs/edge.md).

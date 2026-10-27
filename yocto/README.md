# Edge image (Yocto)

A headless Yocto 6.0 "wrynose" image for the Raspberry Pi 5, reachable as `root@raspberrypi5.local`
by ssh key over Ethernet and Wi-Fi. It carries the camera stack (`rpicam-vid`) and the Python
runtime `mlops_cv.edge` needs; the code itself is rsynced (see
[docs/edge.md](../docs/edge.md#edge-image-yocto)). `kas/pi5.yml` pins the layers,
`meta-mlops-cv/` holds the image and Python recipes.

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

Find the card with `lsblk -d -o NAME,SIZE,TRAN,RM,MODEL`: it is the new removable (`RM 1`)
entry of the card's size, `sdX` behind a USB reader or `mmcblk0` in a built-in slot. Write to
the whole disk, not a partition. A desktop automounts the card's partitions: unmount them first
(`lsblk $CARD` shows no mountpoints after). `bmaptool` is in the `bmap-tools` package on most
distros; `dd` needs nothing.

```bash
CARD=/dev/sdX                # the whole card from lsblk, e.g. /dev/sda or /dev/mmcblk0
sudo umount $CARD?*          # every partition: sda1, sda2… (mmcblk0p1…); "not mounted" is harmless
cd $KAS_WORK_DIR/build/tmp/deploy/images/raspberrypi5
sudo bmaptool copy mlops-cv-edge-image-raspberrypi5.rootfs.wic.bz2 $CARD
# or: bzcat mlops-cv-edge-image-raspberrypi5.rootfs.wic.bz2 | sudo dd of=$CARD bs=4M conv=fsync status=progress

# Eject safely
sync
udisksctl power-off -b $CARD
```

## First boot

```bash
ssh root@raspberrypi5.local
```

Password logins are refused. Using the device (bench, live view) is in [docs/edge.md](../docs/edge.md).

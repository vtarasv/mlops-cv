SUMMARY = "Headless Raspberry Pi 5 edge image: root ssh by key over Ethernet and Wi-Fi"

require recipes-core/images/core-image-minimal.bb

IMAGE_FEATURES += "ssh-server-openssh"

# core-image-minimal skips MACHINE_EXTRA_RRECOMMENDS: add its firmware and udev rules, not all
# kernel-modules. brcmfmac is a module in bcm2712_defconfig.
IMAGE_INSTALL += " \
    rsync avahi-daemon wpa-supplicant \
    udev-rules-rpi \
    linux-firmware-rpidistro-bcm43455 linux-firmware-rpidistro-bcm43456 \
    bluez-firmware-rpidistro-bcm4345c0-hcd bluez-firmware-rpidistro-bcm4345c5-hcd \
    kernel-module-brcmfmac kernel-module-brcmfmac-wcc \
    kernel-module-pwm-fan \
"

# Camera: IMX219 on CAM0 through the downstream rp1-cfe driver and the PiSP back end.
IMAGE_INSTALL += " \
    libcamera libcamera-apps \
    kernel-module-rp1-cfe-downstream kernel-module-pisp-be \
    kernel-module-imx219 kernel-module-i2c-designware-platform \
"

# Set in kas/local.yml (see kas/local.yml.example).
python () {
    for var in ("MLOPS_CV_WIFI_SSID", "MLOPS_CV_WIFI_PSK", "MLOPS_CV_ROOT_PUBKEY"):
        if not d.getVar(var):
            bb.fatal(f"{var} is unset: copy yocto/kas/local.yml.example to yocto/kas/local.yml")
}

ROOTFS_POSTPROCESS_COMMAND += "mlops_cv_rootfs_config"

mlops_cv_rootfs_config () {
    install -d -m 0700 ${IMAGE_ROOTFS}/root/.ssh
    echo '${MLOPS_CV_ROOT_PUBKEY}' > ${IMAGE_ROOTFS}/root/.ssh/authorized_keys
    chmod 0600 ${IMAGE_ROOTFS}/root/.ssh/authorized_keys

    install -d ${IMAGE_ROOTFS}${sysconfdir}/ssh/sshd_config.d
    printf 'PermitRootLogin prohibit-password\nPasswordAuthentication no\n' \
        > ${IMAGE_ROOTFS}${sysconfdir}/ssh/sshd_config.d/10-root-key-only.conf

    # eth0 gets DHCP from systemd-conf's stock 80-wired.network.
    install -d ${IMAGE_ROOTFS}${sysconfdir}/systemd/network
    printf '[Match]\nName=wlan0\n\n[Network]\nDHCP=yes\n' \
        > ${IMAGE_ROOTFS}${sysconfdir}/systemd/network/80-wlan0.network

    install -d -m 0700 ${IMAGE_ROOTFS}${sysconfdir}/wpa_supplicant
    printf 'network={\n\tssid="%s"\n\tpsk="%s"\n}\n' '${MLOPS_CV_WIFI_SSID}' '${MLOPS_CV_WIFI_PSK}' \
        > ${IMAGE_ROOTFS}${sysconfdir}/wpa_supplicant/wpa_supplicant-wlan0.conf
    chmod 0600 ${IMAGE_ROOTFS}${sysconfdir}/wpa_supplicant/wpa_supplicant-wlan0.conf
    install -d ${IMAGE_ROOTFS}${sysconfdir}/systemd/system/multi-user.target.wants
    ln -sf ${systemd_system_unitdir}/wpa_supplicant@.service \
        ${IMAGE_ROOTFS}${sysconfdir}/systemd/system/multi-user.target.wants/wpa_supplicant@wlan0.service
}

---
title: "手抖删掉Python恢复手册"
date: 2017-09-17T23:59:17+08:00
draft: true
---

# 0x00
故事的开始，老张一不小心手抖，把Ubuntu系统自带的`python` 用apt-get给 Remove了，然后在remove的时候会自动带上所有依赖着python这个包的所有安装过的包，这次主要最麻烦的是ubuntu的好多图形化的Package都依赖着Python，这样一下就会把Ubuntu的Desktop搞挂了。为了祭奠这些被删掉的Package，下面提供一份不完整的修复手册。

# 0x01
先别慌，如果不小心在remove的时候按下了 **Y** ，把所有Package都删掉了，这个时候先不用慌，记住这个时候你的系统最好不要重启。然后你可以通过apt的日志里面把刚刚删除掉的包都找出来：
```
grep Remove /var/log/apt/history.log | tail -1 | sed -e 's|Remove: ||g' -e 's|([^)]*)||g' -e 's|:[^ ]* ||g' -e 's|,||g'
```
然后把这些包全部重新`apt-get install`一遍，如无意外因该都能安装回来。

# 0x02
好了，如果在上一步重新安装回来的时候遇到这些Package本身的依赖问题，或者在你remove完之后误操作导致的依赖问题，可以先把系统的基础包先安装回来。至于怎么找到这些系统的基础包，建议在虚拟机上用同样的系统镜像安装一个新的系统，然后`sudo apt-get remove python`后，运行下面找到被删除掉对应的系统基础包：

```
grep Remove /var/log/apt/history.log | tail -1 | sed -e 's|Remove: ||g' -e 's|([^)]*)||g' -e 's|:[^ ]* ||g' -e 's|,||g'
```

拿14.04来举个例子，如果你删掉了python，那么也会删除掉下面这些系统基础包：

```
unity-control-center-signon ibus-table python-httplib2 python-dbus python-distlib oneconf hplip rhythmbox-plugins gconf2 apt-xapian-index python-html5lib python-support python-pam python-crypto xul-ext-webaccounts python-pyasn1 apturl python-samba indicator-bluetooth python-dirspec python-twisted-bin python-pkg-resources python-twisted-mail python-qt4-dbus gnome-terminal python-colorama python-numpy python-serial python-urllib3 python-qt4 ubuntu-desktop deja-dup-backend-gvfs python-pil python-simplegeneric python-notify python-pip compiz-gnome unity python-xapian python-cups python-pexpect python-renderpm python-piston-mini-client python-tdb python-apt libreoffice-gnome gnome-terminal-data ibus-pinyin python-setuptools ubuntu-release-upgrader-gtk duplicity landscape-client-ui-install python3-oneconf rhythmbox-plugin-magnatune totem-plugins python-gobject compiz python-imaging python-libxml2 rhythmbox-mozilla python-oauthlib python-guppy python-ibus python-aptdaemon.gtk3widgets software-center-aptdaemon-plugins python-reportlab-accel rhythmbox-plugin-cdrecorder printer-driver-postscript-hp update-notifier-common checkbox-gui system-config-printer-common deja-dup python-reportlab python-twisted-names python-debian python-gi totem python-cairo python-dev samba-common-bin python-w3lib python-xdg sessioninstaller update-notifier python-guppy-lib python-commandnotfound python-defer ibus python-lxml python-django rhythmbox ubuntu-sso-client python-wheel samba-libs python-gnomekeyring python-sip python-debtagshw python-gi-cairo python-six oneconf-common gvfs-backends python-boto python-gconf python-aptdaemon python-gobject-2 python-smbc python python-openssl python-pygments python-decorator ubuntu-system-service python-twisted-core smbclient libpurple-bin totem-mozilla python-mysqldb rhythmbox-plugin-zeitgeist python-zeitgeist python-scrapy python-chardet python-twisted-conch zeitgeist python-zope.interface unity-control-center gedit unity-webapps-common webaccounts-extension-common python-requests python-lockfile software-center python-ubuntu-sso-client system-config-printer-gnome transmission-gtk python-talloc ubuntu-sso-client-qt hplip-data system-config-printer-udev python-cupshelpers plainbox-provider-checkbox python-twisted-web python-gtk2 python-oneconf update-manager python-gdbm python-ldb printer-driver-sag-gdi gnome-user-share aisleriot libsmbclient nautilus-share ipython xdiagnose 
```

# 0x03
OK， 剩下就是一些非系统包的安装了，通过事故机器的apt上的log找到被删掉的包以及和系统的基础包diff一下就出来了。或者你重新安装一遍你删除过的所有包：
```
rm restore
echo '#!/bin/bash' > restore
echo sudo apt-get install -f `grep Remove /var/log/apt/history.log | tail -1 | sed -e 's|Remove: ||g' -e 's|([^)]*)||g' -e 's|:[^ ]* ||g' -e 's|,||g'` >> restore
chmod +x restore 
./restore
```
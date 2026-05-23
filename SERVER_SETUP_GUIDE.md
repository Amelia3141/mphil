# Multi-User GPU Server Setup Guide

General best practices for setting up 2 GPU towers for ~3 remote users.

## Table of Contents
1. [Initial Server Configuration](#initial-server-configuration)
2. [User Account Setup](#user-account-setup)
3. [SSH Access & Security](#ssh-access--security)
4. [tmux for Persistent Sessions](#tmux-for-persistent-sessions)
5. [Shared File System](#shared-file-system)
6. [Monitoring & Resource Management](#monitoring--resource-management)
7. [Best Practices](#best-practices)
8. [Online Resources](#online-resources)

---

## Initial Server Configuration

### Set Static IP Addresses

**On each tower:**

```bash
# Find your network interface name
ip addr show

# Edit netplan config (Ubuntu/Debian)
sudo nano /etc/netplan/01-netcfg.yaml
```

```yaml
network:
  version: 2
  ethernets:
    eno1:  # Replace with your interface name
      dhcp4: no
      addresses: [192.168.1.101/24]  # Tower 1 = .101, Tower 2 = .102
      gateway4: 192.168.1.1
      nameservers:
        addresses: [8.8.8.8, 8.8.4.4]
```

```bash
sudo netplan apply
```

### Enable SSH Server

```bash
sudo apt update
sudo apt install openssh-server
sudo systemctl enable ssh
sudo systemctl start ssh

# Verify it's running
sudo systemctl status ssh
```

### Install Essential Tools

```bash
sudo apt update
sudo apt install -y \
    tmux \
    htop \
    ncdu \
    vim \
    git \
    rsync \
    screen
```

---

## User Account Setup

### Create User Accounts

```bash
# Create each user
sudo adduser alice
sudo adduser bob
sudo adduser charlie

# Set strong passwords
sudo passwd alice

# Optional: Give specific users sudo access
sudo usermod -aG sudo alice
```

### Set Up User Groups for Shared Resources

```bash
# Create a shared group for all GPU users
sudo groupadd gpuusers

# Add all users to this group
sudo usermod -aG gpuusers alice
sudo usermod -aG gpuusers bob
sudo usermod -aG gpuusers charlie
```

---

## SSH Access & Security

### SSH Key-Based Authentication (Recommended)

**On user's local laptop:**

```bash
# Generate SSH key (if not already done)
ssh-keygen -t ed25519 -C "alice@email.com"

# Copy public key to server
ssh-copy-id alice@192.168.1.101
```

**On server (optional - disable password login for security):**

```bash
sudo nano /etc/ssh/sshd_config
```

Change:
```
PasswordAuthentication no
PubkeyAuthentication yes
```

```bash
sudo systemctl restart ssh
```

### SSH Config for Easy Access

**Users create `~/.ssh/config` on their laptop:**

```
Host tower1
    HostName 192.168.1.101
    User alice
    IdentityFile ~/.ssh/id_ed25519
    ServerAliveInterval 60

Host tower2
    HostName 192.168.1.102
    User alice
    IdentityFile ~/.ssh/id_ed25519
    ServerAliveInterval 60
```

Now SSH with just: `ssh tower1`

---

## tmux for Persistent Sessions

### Why tmux?

- Sessions persist even if SSH disconnects
- Can detach and reattach from anywhere
- Multiple windows/panes
- Essential for long-running jobs

### Install and Configure

**Already installed via earlier apt command. Create config:**

```bash
nano ~/.tmux.conf
```

**Recommended tmux configuration:**

```bash
# ~/.tmux.conf

# Remap prefix from Ctrl+b to Ctrl+a (easier to type)
unbind C-b
set-prefix C-a
bind C-a send-prefix

# Enable mouse support
set -g mouse on

# Start window numbering at 1 (easier than 0)
set -g base-index 1
setw -g pane-base-index 1

# Split panes with | and -
bind | split-window -h
bind - split-window -v
unbind '"'
unbind %

# Reload config with r
bind r source-file ~/.tmux.conf \; display "Config reloaded!"

# Better colors
set -g default-terminal "screen-256color"

# Status bar
set -g status-bg black
set -g status-fg white
set -g status-left '#[fg=green]#H #[fg=yellow]#S '
set -g status-right '#[fg=cyan]%Y-%m-%d %H:%M'

# Highlight active window
setw -g window-status-current-style fg=black,bg=green

# Increase scrollback buffer
set -g history-limit 10000

# Vim-style pane navigation
bind h select-pane -L
bind j select-pane -D
bind k select-pane -U
bind l select-pane -R
```

### tmux Quick Reference

**Basic Commands:**

```bash
# Start new session
tmux

# Start named session
tmux new -s mywork

# List sessions
tmux ls

# Attach to session
tmux attach -t mywork

# Detach from session (while inside tmux)
Ctrl+a d

# Kill session
tmux kill-session -t mywork
```

**Inside tmux (Ctrl+a is prefix):**

| Command | Action |
|---------|--------|
| `Ctrl+a c` | Create new window |
| `Ctrl+a n` | Next window |
| `Ctrl+a p` | Previous window |
| `Ctrl+a 0-9` | Switch to window number |
| `Ctrl+a ,` | Rename window |
| `Ctrl+a \|` | Split pane vertically |
| `Ctrl+a -` | Split pane horizontally |
| `Ctrl+a arrow` | Navigate between panes |
| `Ctrl+a d` | Detach session |
| `Ctrl+a ?` | List all keybindings |

**Typical Workflow:**

```bash
# SSH to server
ssh tower1

# Create or attach to tmux session
tmux attach -t work || tmux new -s work

# Run your long job
python train_model.py  # This will keep running even if you disconnect

# Detach (Ctrl+a d)
# Close laptop, go home

# Later, from anywhere:
ssh tower1
tmux attach -t work  # Your job is still running!
```

---

## Shared File System

### Directory Structure

**Create shared directories:**

```bash
# Shared data storage (read-only for most users)
sudo mkdir -p /shared/datasets
sudo chown root:gpuusers /shared/datasets
sudo chmod 2775 /shared/datasets

# Shared results (everyone can write)
sudo mkdir -p /shared/results
sudo chown root:gpuusers /shared/results
sudo chmod 2775 /shared/results

# Shared scratch space (temporary files)
sudo mkdir -p /shared/scratch
sudo chown root:gpuusers /shared/scratch
sudo chmod 2777 /shared/scratch
```

**Recommended structure:**

```
/shared/
├── datasets/          # Common datasets (managed by admin)
│   ├── cohort1/
│   │   ├── prob_nl.npy
│   │   └── prob_score.npy
│   └── cohort2/
│
├── results/           # Everyone's results (organized by user)
│   ├── alice/
│   │   ├── experiment1/
│   │   └── experiment2/
│   ├── bob/
│   └── charlie/
│
└── scratch/           # Temporary working space
    ├── alice/
    ├── bob/
    └── charlie/

/home/alice/           # Private user space
├── code/              # User's personal code
└── private_data/      # User's private data
```

### Set Up User Subdirectories

**Run for each user:**

```bash
# Create user's results directory
sudo mkdir -p /shared/results/alice
sudo chown alice:gpuusers /shared/results/alice
sudo chmod 755 /shared/results/alice

# Create user's scratch space
sudo mkdir -p /shared/scratch/alice
sudo chown alice:gpuusers /shared/scratch/alice
sudo chmod 700 /shared/scratch/alice
```

### Disk Usage Monitoring

```bash
# See disk space
df -h

# See directory sizes
du -sh /shared/*

# Interactive disk usage explorer (better)
ncdu /shared
```

### Automatic Cleanup of Scratch (Optional)

```bash
# Add to root crontab to delete scratch files older than 30 days
sudo crontab -e
```

Add:
```
0 3 * * * find /shared/scratch -type f -mtime +30 -delete
```

---

## Monitoring & Resource Management

### Check GPU Usage

```bash
# Simple
nvidia-smi

# Continuous monitoring
watch -n 1 nvidia-smi

# Just show running processes
nvidia-smi pmon
```

### Check CPU/Memory

```bash
# Interactive process viewer (better than top)
htop

# See who's logged in
w

# See all running processes
ps aux | grep python
```

### Create Shared Status Script

Users can run to see current usage:

```bash
# Already created: gpu_status.sh
# Make it accessible
sudo cp gpu_status.sh /usr/local/bin/gpu_status
sudo chmod +x /usr/local/bin/gpu_status
```

---

## Best Practices

### For Server Admins

1. **Regular backups**: Back up `/home/` and `/shared/results/` weekly
   ```bash
   rsync -av /shared/results/ /backup/results-$(date +%Y%m%d)/
   ```

2. **Disk quotas**: Prevent users from filling disk
   ```bash
   sudo apt install quota
   # Configure per-user quotas
   ```

3. **Monitor security**: Check login attempts
   ```bash
   sudo journalctl -u ssh | grep "Failed password"
   ```

4. **Keep software updated**:
   ```bash
   sudo apt update && sudo apt upgrade
   ```

5. **Set up monitoring alerts**: Email on disk full, high temps, etc.

### For Users

1. **Always use tmux** for long jobs
2. **Clean up scratch space** regularly
3. **Document your experiments** in `/shared/results/yourname/README.md`
4. **Check GPU availability** before starting jobs
5. **Compress large results**: `tar -czf results.tar.gz results/`
6. **Use relative paths** in code for portability

### Communication

Set up a shared coordination system:
- **Slack/Discord channel** for quick questions
- **Shared Google Sheet** for GPU scheduling
- **Wiki/Notion page** for documentation

---

## Online Resources

### General Linux Server Management

- **DigitalOcean Community Tutorials**  
  https://www.digitalocean.com/community/tutorials  
  Excellent beginner-friendly guides for Linux server setup

- **Ubuntu Server Guide (Official)**  
  https://ubuntu.com/server/docs  
  Comprehensive official documentation

- **The Linux Command Line (free book)**  
  https://linuxcommand.org/tlcl.php  
  Great intro to command line

### SSH & Security

- **SSH Academy**  
  https://www.ssh.com/academy/ssh  
  Everything about SSH

- **Hardening SSH (Linode Guide)**  
  https://www.linode.com/docs/guides/use-public-key-authentication-with-ssh/  
  Security best practices

### tmux

- **tmux Cheat Sheet**  
  https://tmuxcheatsheet.com/  
  Quick reference

- **A tmux Crash Course**  
  https://thoughtbot.com/blog/a-tmux-crash-course  
  Excellent tutorial

- **The Tao of tmux (free book)**  
  https://leanpub.com/the-tao-of-tmux/read  
  Deep dive into tmux

### GPU Server Management

- **NVIDIA GPU Admin Guide**  
  https://docs.nvidia.com/deploy/gpu-admin-guide/  
  Official NVIDIA documentation

- **Managing Multiple Users on GPU Servers**  
  https://towardsdatascience.com/how-to-setup-a-multi-user-deep-learning-server-8f5494cfa9d0  
  Good practical guide

- **SLURM Documentation (if you want a proper scheduler)**  
  https://slurm.schedmd.com/quickstart.html  
  Industry-standard job scheduler

### File System & Storage

- **Linux File Permissions Guide**  
  https://wiki.archlinux.org/title/File_permissions_and_attributes  
  Understanding chmod, chown, etc.

- **Rsync Tutorial**  
  https://www.digitalocean.com/community/tutorials/how-to-use-rsync-to-sync-local-and-remote-directories  
  For backups and file syncing

### System Monitoring

- **htop Explained**  
  https://www.deonsworld.co.za/2012/12/20/understanding-and-using-htop-monitor-system-resources/  
  Understanding system resources

- **Prometheus + Grafana for Server Monitoring**  
  https://prometheus.io/docs/visualization/grafana/  
  Advanced monitoring dashboards

### Books

- **"UNIX and Linux System Administration Handbook"**  
  The bible of sysadmin - comprehensive

- **"The Practice of System and Network Administration"**  
  Best practices and patterns

### Community

- **r/linuxadmin** (Reddit)  
  Active community for questions

- **Server Fault** (StackExchange)  
  https://serverfault.com/  
  Q&A for sysadmins

- **Ask Ubuntu**  
  https://askubuntu.com/  
  Ubuntu-specific help

---

## Quick Start Checklist

- [ ] Set static IPs on both towers
- [ ] Enable SSH server
- [ ] Create user accounts (alice, bob, charlie)
- [ ] Set up SSH key authentication
- [ ] Install tmux and copy config to each user's home
- [ ] Create `/shared/` directory structure
- [ ] Set up GPU management scripts (claim/release/status)
- [ ] Test SSH + tmux from each user's laptop
- [ ] Document tower IPs and login info
- [ ] Create shared Slack/Discord for coordination

## Getting Help

**Server logs:**
```bash
# SSH login attempts
sudo journalctl -u ssh

# System messages
dmesg

# GPU driver issues
nvidia-smi -q
```

**Test connectivity:**
```bash
# From local machine
ping 192.168.1.101
ssh -v tower1  # Verbose SSH for debugging
```

---

**Further questions?** Check the online resources above or the community forums!

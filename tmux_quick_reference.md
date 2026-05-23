# tmux Quick Reference Card

## What is tmux?

**tmux** = terminal multiplexer. Keeps your sessions alive even when you disconnect from SSH.

**Why you need it for remote GPU work:**
- Start a training job, disconnect, come back tomorrow - still running
- Multiple windows/panes without multiple SSH connections
- Screen sharing / collaboration

---

## Installation & Setup

```bash
# Install
sudo apt install tmux

# Create config file
nano ~/.tmux.conf
```

**Copy this basic config:**

```bash
# Change prefix to Ctrl+a (easier than Ctrl+b)
set -g prefix C-a
unbind C-b
bind C-a send-prefix

# Enable mouse
set -g mouse on

# Better colors
set -g default-terminal "screen-256color"

# Split panes with | and -
bind | split-window -h
bind - split-window -v

# Reload config
bind r source-file ~/.tmux.conf
```

---

## Essential Commands

### Session Management

```bash
# Create new session
tmux

# Create named session (recommended!)
tmux new -s mywork

# List all sessions
tmux ls

# Attach to existing session
tmux attach -t mywork
tmux a -t mywork        # Short version

# Kill session
tmux kill-session -t mywork

# Detach from current session (inside tmux)
Ctrl+a d
```

---

## Keyboard Shortcuts

**Prefix key:** `Ctrl+a` (press this before any command below)

### Windows (like tabs)

| Shortcut | Action |
|----------|--------|
| `Ctrl+a c` | **C**reate new window |
| `Ctrl+a ,` | Rename current window |
| `Ctrl+a n` | **N**ext window |
| `Ctrl+a p` | **P**revious window |
| `Ctrl+a 0-9` | Switch to window 0-9 |
| `Ctrl+a w` | List all windows |
| `Ctrl+a &` | Kill current window |

### Panes (split screen)

| Shortcut | Action |
|----------|--------|
| `Ctrl+a \|` | Split vertically (left/right) |
| `Ctrl+a -` | Split horizontally (top/bottom) |
| `Ctrl+a arrow` | Move between panes |
| `Ctrl+a o` | Cycle through panes |
| `Ctrl+a x` | Kill current pane |
| `Ctrl+a z` | Toggle pane zoom (fullscreen) |
| `Ctrl+a {` or `}` | Swap panes |

### Sessions

| Shortcut | Action |
|----------|--------|
| `Ctrl+a d` | **D**etach from session |
| `Ctrl+a s` | List all **s**essions |
| `Ctrl+a $` | Rename current session |
| `Ctrl+a (` | Previous session |
| `Ctrl+a )` | Next session |

### Other

| Shortcut | Action |
|----------|--------|
| `Ctrl+a ?` | Show all keybindings (VERY useful!) |
| `Ctrl+a :` | Enter command mode |
| `Ctrl+a [` | Enter scroll mode (use arrow keys, q to quit) |

---

## Common Workflows

### Long-Running Job

```bash
# SSH to server
ssh tower1

# Start tmux session
tmux new -s training

# Start your job
python long_training.py

# Detach (Ctrl+a d)
# Close laptop, go home

# Next day: SSH back and reattach
ssh tower1
tmux attach -t training

# Your job is still running!
```

### Multiple Windows for Different Tasks

```bash
tmux new -s work

# Window 0: Training job
python train.py

# Create new window (Ctrl+a c)
# Window 1: Monitor GPUs
watch -n 1 nvidia-smi

# Create new window (Ctrl+a c)
# Window 2: Edit code
vim model.py

# Switch between windows: Ctrl+a 0, Ctrl+a 1, Ctrl+a 2
```

### Split Screen Monitoring

```bash
tmux new -s monitor

# Split vertically (Ctrl+a |)
# Left pane: training log
tail -f training.log

# Right pane: GPU usage
watch -n 1 nvidia-smi

# Split right pane horizontally (Ctrl+a -)
# Top-right: GPU
# Bottom-right: htop
htop
```

---

## Tips & Tricks

### Named Windows

```bash
# Rename window (Ctrl+a ,) to describe what it's doing
# Example names: "training", "monitoring", "editing"
```

### Copy Mode (Scrolling)

```bash
# Enter copy mode: Ctrl+a [
# Navigate with arrow keys or Page Up/Down
# Press q to exit
```

### Synchronize Panes

```bash
# Type same command in all panes simultaneously
# Ctrl+a : setw synchronize-panes on
# (Type again with 'off' to disable)
```

### Mouse Support

With `set -g mouse on` in config:
- Click to switch panes
- Click and drag to resize panes
- Scroll to navigate history
- Click window names to switch

---

## Troubleshooting

**tmux session still exists after server reboot?**
No, tmux sessions are lost on reboot. For truly persistent jobs, use systemd services or `nohup`.

**Can't see colors properly?**
Add to `~/.bashrc`:
```bash
export TERM=screen-256color
```

**Nested tmux sessions (tmux inside tmux)?**
Press prefix twice: `Ctrl+a Ctrl+a` to send command to inner session.

**Lost which session you're in?**
Look at bottom status bar - shows session name.

---

## Alternatives

- **screen**: Older alternative to tmux (simpler but less features)
- **byobu**: tmux wrapper with more user-friendly defaults
- **zellij**: Modern Rust-based alternative

---

## Cheat Sheet Summary

**Start work:**
```bash
ssh tower1
tmux new -s mywork
# do stuff
Ctrl+a d  # detach
exit  # close SSH
```

**Continue work:**
```bash
ssh tower1
tmux attach -t mywork
# keep working
```

**Check what's running:**
```bash
tmux ls
```

**That's 90% of what you need to know!**

---

## Learn More

- Interactive tutorial: `tmux` then `Ctrl+a ?`
- https://tmuxcheatsheet.com/
- https://github.com/tmux/tmux/wiki

import os

# Bind to the port provided by Render
port = int(os.environ.get('PORT', 10000))
bind = f"0.0.0.0:{port}"

# Number of worker processes
workers = 1

# Timeout seconds
timeout = 120

# Log level
loglevel = 'info'
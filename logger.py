# -------------------------------
# Debug LOGGING
# -------------------------------

def log(msg):
    """Send debug log to Max API or print if unavailable."""
    # if debug:
    try:
        print(msg)
    except Exception:
        print(msg)
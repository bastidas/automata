"""
Matplotlib configuration for backend/headless environments.
This must be imported before any other matplotlib imports.
"""
import os
import matplotlib

# Set non-GUI backend for headless/backend environments
# This prevents "Starting a Matplotlib GUI outside of the main thread" warnings
if 'DISPLAY' not in os.environ or os.environ.get('MPLBACKEND') == 'Agg':
    matplotlib.use('Agg', force=True)
else:
    # Try to use Agg backend, fall back to current if it fails  
    try:
        matplotlib.use('Agg', force=True)
    except:
        pass

def configure_matplotlib_for_backend():
    """Ensure matplotlib is configured for backend/headless use"""
    current_backend = matplotlib.get_backend()
    if current_backend not in ['Agg', 'svg', 'pdf', 'ps']:
        print(f"Warning: Using {current_backend} backend, may cause GUI issues in backend")
    return current_backend
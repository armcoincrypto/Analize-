#!/usr/bin/env python3
"""
SSL Bypass - Disable SSL verification for networks with SSL interception.

USE WITH CAUTION: Only use this on trusted networks where SSL interception
is causing certificate errors (corporate VPNs, proxies, etc.)

Import this module at the start of your script to disable SSL verification:
    import ssl_bypass  # Must be first import!
"""

import ssl
import urllib3
import warnings

# Disable SSL verification warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
warnings.filterwarnings('ignore', message='Unverified HTTPS request')

# Create unverified SSL context
ssl._create_default_https_context = ssl._create_unverified_context

# Monkey-patch requests to disable SSL verification by default
try:
    import requests
    from requests.adapters import HTTPAdapter
    from urllib3.util.ssl_ import create_urllib3_context

    # Store original
    _original_init = HTTPAdapter.__init__

    def _patched_init(self, *args, **kwargs):
        _original_init(self, *args, **kwargs)
        self.config = {'verify': False}

    # Patch requests.Session to default verify=False
    _original_request = requests.Session.request

    def _patched_request(self, method, url, **kwargs):
        kwargs.setdefault('verify', False)
        return _original_request(self, method, url, **kwargs)

    requests.Session.request = _patched_request

    # Also patch the module-level functions
    _original_get = requests.get
    _original_post = requests.post

    def patched_get(url, **kwargs):
        kwargs.setdefault('verify', False)
        return _original_get(url, **kwargs)

    def patched_post(url, data=None, json=None, **kwargs):
        kwargs.setdefault('verify', False)
        return _original_post(url, data=data, json=json, **kwargs)

    requests.get = patched_get
    requests.post = patched_post

    print("[SSL] SSL verification disabled for this session")

except ImportError:
    pass

# Test function
def test_ssl():
    """Test if SSL bypass is working."""
    import requests
    try:
        response = requests.get('https://api.binance.com/api/v3/time', timeout=10)
        print(f"[SSL] Test successful: {response.json()}")
        return True
    except Exception as e:
        print(f"[SSL] Test failed: {e}")
        return False


if __name__ == "__main__":
    test_ssl()

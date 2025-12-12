#!/usr/bin/env python3
"""
Network Diagnostic - Check what's actually being returned from APIs.
Helps identify if VPN is blocking, returning captcha, or other issues.
"""

import ssl_bypass  # Must be first!
import requests

def diagnose_api(name: str, url: str):
    """Check what an API actually returns."""
    print(f"\n{'='*60}")
    print(f"Testing: {name}")
    print(f"URL: {url}")
    print('='*60)

    try:
        response = requests.get(url, timeout=15)
        print(f"Status Code: {response.status_code}")
        print(f"Content-Type: {response.headers.get('content-type', 'N/A')}")
        print(f"Content Length: {len(response.text)} bytes")

        # Show first 500 chars of response
        content = response.text[:500]
        print(f"\nResponse Preview:")
        print("-" * 40)
        print(content)
        if len(response.text) > 500:
            print(f"\n... ({len(response.text) - 500} more bytes)")
        print("-" * 40)

        # Check for common block indicators
        lower_content = response.text.lower()
        if '<html' in lower_content:
            print("\n⚠️  HTML DETECTED - Likely a block page or captcha!")
        if 'captcha' in lower_content:
            print("\n⚠️  CAPTCHA DETECTED - Network requires verification!")
        if 'blocked' in lower_content or 'forbidden' in lower_content:
            print("\n⚠️  BLOCK MESSAGE DETECTED!")
        if 'vpn' in lower_content or 'proxy' in lower_content:
            print("\n⚠️  VPN/PROXY MESSAGE DETECTED!")
        if response.status_code == 403:
            print("\n⚠️  403 FORBIDDEN - Access denied by server or network!")
        if response.text == '':
            print("\n⚠️  EMPTY RESPONSE - Network may be silently blocking!")

        # Try to parse as JSON
        try:
            data = response.json()
            print(f"\n✓ Valid JSON response")
            if isinstance(data, dict):
                print(f"  Keys: {list(data.keys())[:5]}")
        except:
            print(f"\n✗ Not valid JSON")

    except requests.exceptions.SSLError as e:
        print(f"\n✗ SSL ERROR: {e}")
        print("  -> SSL bypass may not be working correctly")
    except requests.exceptions.ConnectionError as e:
        print(f"\n✗ CONNECTION ERROR: {e}")
        print("  -> Network blocking or DNS issue")
    except requests.exceptions.Timeout:
        print(f"\n✗ TIMEOUT - Request took too long")
        print("  -> Network may be throttling or blocking")
    except Exception as e:
        print(f"\n✗ ERROR: {type(e).__name__}: {e}")


def main():
    print("=" * 60)
    print("NETWORK DIAGNOSTIC FOR CRYPTO APIs")
    print("=" * 60)
    print("This will test various APIs to identify what your network returns.")

    # Test APIs in order of preference
    apis = [
        ("Binance Time (simple)", "https://api.binance.com/api/v3/time"),
        ("Binance US Time", "https://api.binance.us/api/v3/time"),
        ("CoinGecko Ping", "https://api.coingecko.com/api/v3/ping"),
        ("Bybit Time", "https://api.bybit.com/v5/market/time"),
        ("Kraken Time", "https://api.kraken.com/0/public/Time"),
        ("KuCoin Time", "https://api.kucoin.com/api/v1/timestamp"),
        ("Non-crypto test (httpbin)", "https://httpbin.org/json"),
        ("Google (should work)", "https://www.google.com"),
    ]

    for name, url in apis:
        diagnose_api(name, url)

    print("\n" + "=" * 60)
    print("DIAGNOSIS COMPLETE")
    print("=" * 60)
    print("""
POSSIBLE ISSUES:

1. If crypto APIs return HTML/block pages but httpbin/google work:
   -> Your network specifically blocks crypto traffic
   -> Try disconnecting from VPN

2. If all APIs timeout or connection error:
   -> Firewall blocking outbound HTTPS
   -> Check firewall settings

3. If 403 Forbidden on all crypto APIs:
   -> Geographic restriction or corporate policy
   -> Try a different network

4. If SSL errors persist:
   -> VPN has deep packet inspection
   -> May need to disconnect VPN entirely

5. If everything works:
   -> Run analysis again, APIs should work!
""")


if __name__ == "__main__":
    main()

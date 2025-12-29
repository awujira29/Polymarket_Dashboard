import requests

print("Testing Polymarket API...")

# Test 1: Gamma API Events
try:
    url = "https://gamma-api.polymarket.com/events"
    params = {"limit": 5, "active": True}
    response = requests.get(url, params=params, timeout=10)
    print(f"\n✅ Events API Status: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        print(f"   Events found: {len(data)}")
        if data:
            print(f"   First event: {data[0].get('title', 'No title')}")
    else:
        print(f"   Response: {response.text[:200]}")
except Exception as e:
    print(f"❌ Events API Error: {e}")

# Test 2: Markets API
try:
    url = "https://gamma-api.polymarket.com/markets"
    params = {"limit": 5}
    response = requests.get(url, params=params, timeout=10)
    print(f"\n✅ Markets API Status: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        print(f"   Markets found: {len(data)}")
        if data:
            print(f"   First market: {data[0].get('question', 'No question')[:60]}")
    else:
        print(f"   Response: {response.text[:200]}")
except Exception as e:
    print(f"❌ Markets API Error: {e}")

print("\n" + "="*60)
import os, requests, json
API_KEY = os.environ.get('RUNPOD_API_KEY', '')
URL = 'https://api.runpod.ai/v2/54td14oe86jexh/runsync'
HEADERS = {'Authorization': f'Bearer {API_KEY}', 'Content-Type': 'application/json'}
TEXT = """Siri's AI overhaul may have been the headline announcement at Apple's Worldwide Developers Conference earlier this month, but Apple's broader AI strategy is taking shape through a series of smaller features embedded across its software. Rather than asking consumers to adopt the new AI-powered version of Siri to get all the benefits that AI brings, the company is weaving AI into the apps and services people already use, with a focus on solving real-world problems. The result is that your iPhone will be able to split restaurant bills among friends, secure your passwords after data breaches, automate tasks, and organize information with less manual effort, among other things. Individually, these features may not be as dramatic as a Siri that finally understands your personal context and can take action on your behalf. But combined, they showcase a vision for AI that's less about chatting with a bot and more about making Apple's software itself feel smarter and more capable. Beyond Siri AI, here are the smaller AI features in iOS twenty-seven that we're most looking forward to using. The features are live now in the developer beta and will soon arrive in the public beta, before iOS twenty-seven's general public release later this fall."""
payload = {
    'input': {
        'text': TEXT, 
        'voice': 'af_heart', 
        'lang_code': 'a', 
        'speed': 1.0, 
        'timestamps': True, 
        'micro_pause_ms': 50, 
        'pause_after': [2, 8, 13, 16, 21, 30, 34, 46, 54, 62, 67, 74, 87, 95, 100, 107, 115, 124, 131, 139, 146, 153, 158, 167, 177, 186, 194, 203]
    }
}
r = requests.post(URL, headers=HEADERS, json=payload)
data = r.json()
if "output" in data and "audio_base64" in data["output"]:
    print("SUCCESS")
    print(json.dumps(data["output"].get("word_boundaries", [])[:2], indent=2))
else:
    print("FAILED")
    print(json.dumps(data, indent=2))

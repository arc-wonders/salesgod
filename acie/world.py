"""World knowledge for questions that have nothing to do with the product.

Callers ask random things ("aaj mausam kaisa hai?", "who won the last world cup?", "what's the time?")
and an agent that only says "I can't help with that" sounds like a script. So before the LLM
answers an off-topic question, this module looks things up and gives the result to the LLM as
reference text (TurnPlan.context):

  clock      date / time / day, local (Asia/Kolkata)                       no network
  weather    Open-Meteo geocoding + current weather (free, no key)          ~300 ms
  wikipedia  search + intro extract, Hindi wiki for Devanagari, else English ~400 ms

The LLM (Qwen3-14B) is the world-knowledge model: it answers from its own knowledge, and the
fetched text grounds it (fresh facts, names, numbers it would otherwise guess). Everything has
a hard time budget; on timeout the LLM still answers from what it knows. Only the question
text is sent out (no phone number, no name). Results are cached per process.
"""

from __future__ import annotations

import asyncio
import re
import time
from datetime import datetime
from zoneinfo import ZoneInfo

import httpx

from .text import normalize

# Wikimedia rejects user agents without a contact URL (403); a generic one, no personal details
UA = {"User-Agent": "VoiceAgentDemo/0.1 (https://www.mediawiki.org/wiki/API:Etiquette)"}
# the geocoder only knows Latin names: common Indian cities as Whisper writes them in Devanagari
_CITY_HI = {"दिल्ली": "Delhi", "मुंबई": "Mumbai", "बंबई": "Mumbai", "कोलकाता": "Kolkata", "चेन्नई": "Chennai",
            "बेंगलुरु": "Bengaluru", "बैंगलोर": "Bengaluru", "बेंगलोर": "Bengaluru", "हैदराबाद": "Hyderabad",
            "पुणे": "Pune", "अहमदाबाद": "Ahmedabad", "जयपुर": "Jaipur", "लखनऊ": "Lucknow", "कानपुर": "Kanpur",
            "पटना": "Patna", "भोपाल": "Bhopal", "इंदौर": "Indore", "नोएडा": "Noida", "गुड़गांव": "Gurugram",
            "गुरुग्राम": "Gurugram", "गुडगांव": "Gurugram", "चंडीगढ़": "Chandigarh", "वाराणसी": "Varanasi",
            "बनारस": "Varanasi", "आगरा": "Agra", "सूरत": "Surat", "नागपुर": "Nagpur", "गोवा": "Panaji",
            "देहरादून": "Dehradun", "शिमला": "Shimla", "अमृतसर": "Amritsar", "लुधियाना": "Ludhiana",
            "रांची": "Ranchi", "गुवाहाटी": "Guwahati", "भुवनेश्वर": "Bhubaneswar", "कोच्चि": "Kochi",
            "गाजियाबाद": "Ghaziabad", "मेरठ": "Meerut", "प्रयागराज": "Prayagraj", "इलाहाबाद": "Prayagraj"}
TZ = ZoneInfo("Asia/Kolkata")

_CLOCK = re.compile(r"\btime\b|what day|which day|\bdate\b|today|samay|kitne baje|tarikh|tareekh|"
                    r"समय|टाइम|कितने बजे|तारीख|आज कौन सा दिन|कौन सा दिन|आज क्या दिन", re.I)
_WEATHER = re.compile(r"weather|temperature|rain|raining|hot|cold|mausam|baarish|barish|garmi|thand|"
                      r"मौसम|बारिश|गर्मी|ठंड|तापमान|टेम्परेचर|वेदर", re.I)
_STOP = set("""what who whom whose which when where why how is are was were the a an of in on at to for do does did
can could would should will tell me about please you your i my we it this that there know kya kaun kab kahan kaise
kyun hai hain tha the ki ka ke ko se me mein batao bataiye bolo pata aap tum sir ji hello hi yaar bhai
right now today tomorrow currently like going there here outside
क्या कौन कब कहाँ कहां कैसे क्यों है हैं था थे की का के को से में पर बताओ बताइए बोलो पता आप तुम सर जी हेलो यार भाई
मुझे मेरा मेरी कोई कुछ आज अभी वो ये यह वह और भी तो""".split())

_CITY_EN = {"bangalore": "Bengaluru", "bombay": "Mumbai", "calcutta": "Kolkata", "madras": "Chennai",
            "gurgaon": "Gurugram", "banaras": "Varanasi", "benares": "Varanasi", "allahabad": "Prayagraj",
            "poona": "Pune", "baroda": "Vadodara", "trivandrum": "Thiruvananthapuram", "cochin": "Kochi"}

_cache: dict[str, tuple[float, str]] = {}
CACHE_S = 600


def _topic_words(text: str) -> list[str]:
    return [w for w in re.findall(r"[\wऀ-ॿ]+", text.lower()) if w not in _STOP and len(w) > 1]


class WorldKnowledge:
    def __init__(self, timeout_s: float = 2.0):
        self.timeout_s = timeout_s
        self.client = httpx.AsyncClient(timeout=timeout_s, headers=UA)

    async def lookup(self, question: str) -> str:
        """Reference text for the LLM ('' if nothing useful). Never raises, never exceeds the budget."""
        key = normalize(question)[:120]
        hit = _cache.get(key)
        if hit and time.time() - hit[0] < CACHE_S:
            return hit[1]
        parts = [self.clock()] if _CLOCK.search(question) else []
        try:
            # a weather question gets the weather only (the encyclopedia returned a 19th-century famine)
            if _WEATHER.search(question):
                tasks = [self.weather(question)]
            elif parts:  # a clock question needs nothing else
                tasks = []
            else:
                tasks = [self.wikipedia(question)]
            found = await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), self.timeout_s + 0.3)
            parts += [f for f in found if isinstance(f, str) and f]
        except (asyncio.TimeoutError, Exception):
            pass
        out = "\n".join(parts)
        _cache[key] = (time.time(), out)
        return out

    @staticmethod
    def clock() -> str:
        now = datetime.now(TZ)
        return f"Current date and time in India: {now.strftime('%A, %d %B %Y, %I:%M %p')} IST."

    async def weather(self, question: str) -> str:
        words = [w for w in _topic_words(question) if not _WEATHER.search(w)][:4]
        words = [_CITY_HI.get(w, _CITY_EN.get(w, w)) for w in words]
        words = [w for w in words if len(w) >= 4 and not re.search(r"[\u0900-\u097F]", w)]

        async def geo(w: str):
            r = await self.client.get("https://geocoding-api.open-meteo.com/v1/search",
                                      params={"name": w, "count": 1, "language": "en"})
            res = (r.json().get("results") or [None])[0]
            # the geocoder matches prefixes ("now" -> Nowrangapur): the name must be the word itself
            if res and (res.get("population") or 0) >= 50000 and res.get("name", "").lower().startswith(w.lower()[:5]):
                return res
            return None

        found = await asyncio.gather(*(geo(w) for w in words), return_exceptions=True)  # all at once
        for res in found:
            if not res or isinstance(res, Exception):
                continue
            f = await self.client.get("https://api.open-meteo.com/v1/forecast", params={
                "latitude": res["latitude"], "longitude": res["longitude"],
                "current": "temperature_2m,relative_humidity_2m,precipitation,weather_code",
                "daily": "temperature_2m_max,temperature_2m_min,precipitation_probability_max",
                "timezone": "auto", "forecast_days": 1})
            j = f.json()
            c, d = j.get("current", {}), j.get("daily", {})
            return (f"Weather now in {res['name']}: {c.get('temperature_2m')}°C, humidity {c.get('relative_humidity_2m')}%, "
                    f"{_WMO.get(c.get('weather_code'), 'conditions unknown')}; today {d.get('temperature_2m_min', ['?'])[0]}"
                    f"-{d.get('temperature_2m_max', ['?'])[0]}°C, chance of rain {d.get('precipitation_probability_max', ['?'])[0]}%.")
        return "Weather: no city was named; ask which city."

    async def wikipedia(self, question: str) -> str:
        words = _topic_words(question)
        if not words:
            return ""
        if re.search(r"[ऀ-ॿ]", question):
            # Hindi Wikipedia search returned off-target pages that the LLM then repeated ("who won the most
            # IPL trophies?" -> a player's article); for Hindi questions the LLM's own knowledge is better
            return ""
        for wiki in ["en"]:
            api = f"https://{wiki}.wikipedia.org/w/api.php"
            r = await self.client.get(api, params={"action": "query", "list": "search", "srsearch": " ".join(words),
                                                   "srlimit": 1, "format": "json"})
            hits = r.json().get("query", {}).get("search", [])
            if not hits:
                continue
            title = hits[0]["title"]
            e = await self.client.get(api, params={"action": "query", "prop": "extracts", "exintro": 1, "explaintext": 1,
                                                   "exsentences": 3, "titles": title, "format": "json", "redirects": 1})
            pages = e.json().get("query", {}).get("pages", {})
            text = next(iter(pages.values()), {}).get("extract", "").strip()
            if text:
                return f"Encyclopedia ({title}): {text[:600]}"
        return ""


_WMO = {0: "clear sky", 1: "mainly clear", 2: "partly cloudy", 3: "overcast", 45: "fog", 48: "fog",
        51: "light drizzle", 53: "drizzle", 55: "heavy drizzle", 61: "light rain", 63: "rain", 65: "heavy rain",
        71: "light snow", 73: "snow", 75: "heavy snow", 80: "rain showers", 81: "rain showers", 82: "violent rain showers",
        95: "thunderstorm", 96: "thunderstorm with hail", 99: "thunderstorm with hail"}

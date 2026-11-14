from common import T
from b3 import OF_EN, OF_HI, OF_HG, OL_EN, OL_HI
from b2 import AA_EN, AA_HI

BAL_EN = "your available balance is eighty two thousand five hundred rupees"
BAL_HI = "आपके खाते में बयासी हज़ार पाँच सौ रुपये उपलब्ध हैं"
LONG_HI = "आपकी दैनिक अंतरण सीमा चालीस हज़ार रुपये है जिसमें से पंद्रह हज़ार का उपयोग हो चुका है अतः शेष सीमा पच्चीस हज़ार है"

# ---------- explicit preferences ----------
# language -> hi
T("en", "can you speak in hindi", "unknown", ex={"language": "hi"}, agent=OL_EN, notes="pure language request; no task intent")
T("en", "hindi please my english is not so good", "unknown", ex={"language": "hi"}, c=1, agent=OF_EN, notes="non-native; implies comprehension trouble")
T("hi", "हिंदी में बोलो", "unknown", ex={"language": "hi"}, c=1, agent=OL_EN, notes="caller answers in Hindi to English agent")
T("hi", "हिंदी में बताइए ना बेटा अंग्रेजी नहीं आती", "unknown", ex={"language": "hi"}, c=1, agent=BAL_EN, notes="elderly; नहीं not deny")
T("hinglish", "hindi mein bolo yaar", "unknown", ex={"language": "hi"}, f=1, agent=OF_EN, notes="'yaar' with request; mild impatience")
T("hinglish", "aap hindi mein baat kar sakte ho kya", "unknown", ex={"language": "hi"}, agent=OL_EN, notes="")
T("hi", "हिंदी में बोलिए ना कितनी बार बोलूं", "unknown", ex={"language": "hi"}, f=2, agent=OL_EN, notes="repeated language request -> clear frustration")
T("hinglish", "woh balance wala phir se bolo hindi mein", "check_balance", ex={"language": "hi"}, agent=BAL_EN, notes="code switch mid request; task + language")
# language -> en
T("hi", "इंग्लिश में बोलो", "unknown", ex={"language": "en"}, agent=LONG_HI, notes="Hindi speaker asks for English (urban caller)")
T("hinglish", "can you speak english please hindi thoda mushkil hai", "unknown", ex={"language": "en"}, c=1, agent=LONG_HI, notes="South Indian / non-Hindi caller")
T("en", "english please", "unknown", ex={"language": "en"}, e=0, agent=OL_HI, notes="")
T("en", "sorry sorry i dont know hindi english please", "unknown", ex={"language": "en"}, c=1, agent=OF_HI, notes="")
T("hi", "आप इंग्लिश में बात करो मुझे वो ज्यादा सही लगता है", "unknown", ex={"language": "en"}, agent=OL_HI, notes="")
T("en", "english english", "unknown", ex={"language": "en"}, f=1, b=True, agent=OF_HI, notes="barge-in repeated keyword")
# language -> hinglish
T("hi", "इतनी शुद्ध हिंदी मत बोलो नॉर्मल बोलो जैसे हम बोलते हैं", "clarify", ex={"language": "hinglish", "complexity": "low"}, c=1, agent=LONG_HI, notes="asks for colloquial mixed Hindi; mapped to hinglish + low complexity")
T("hinglish", "thoda mix karke bolo na hindi english dono", "unknown", ex={"language": "hinglish"}, agent=OL_EN, notes="")
T("en", "you can talk in hinglish also its fine", "unknown", ex={"language": "hinglish"}, agent=OL_HI, notes="")
# verbosity low
T("hinglish", "short mein batao", "unknown", ex={"verbosity": "low"}, f=1, agent=LONG_HI, b=True, notes="")
T("en", "just tell me the number", "check_balance", ex={"verbosity": "low"}, f=1, agent="sure let me first explain the different balances available in your account", notes="impatience")
T("hi", "सीधा बताओ कितना है बस", "check_balance", ex={"verbosity": "low"}, f=1, u=1, agent="जी, आपके सेविंग्स अकाउंट की जानकारी देने से पहले...", b=True, notes="")
T("en", "keep it short i'm in a meeting", "unknown", ex={"verbosity": "low"}, u=2, b=True, notes="")
T("hi", "ज्यादा लंबा मत बोलो बस हां या ना बताओ", "unknown", ex={"verbosity": "low"}, f=1, agent=LONG_HI, notes="'हां या ना' is not affirm/deny")
T("hinglish", "bas amount bolo baaki chhodo", "check_balance", ex={"verbosity": "low"}, agent=BAL_EN, notes="")
# verbosity high
T("en", "can you explain it properly how this limit thing works", "transfer_limit", ex={"verbosity": "high"}, e=2, c=1, agent=OL_EN, notes="")
T("hi", "थोड़ा डिटेल में बताओ कार्ड के बारे में क्या क्या मिलेगा", "card_offer", ex={"verbosity": "high"}, e=2, pend="card_offer", agent=OF_HI, notes="")
T("hinglish", "poora samjhao na fees vagera sab", "card_offer", ex={"verbosity": "high"}, e=2, pend="card_offer", agent=OF_HG, notes="")
# complexity low
T("hi", "थोड़ा धीरे और आसान भाषा में बताइए", "clarify", ex={"complexity": "low"}, c=1, agent=LONG_HI, notes="slow + simple; 'dheere' not mapped to verbosity")
T("hi", "हम ज्यादा पढ़े लिखे नहीं हैं सीधी भाषा में बोलो", "clarify", ex={"complexity": "low"}, c=1, agent=OF_HI, notes="rural; नहीं not deny")
T("en", "uh can you say it in simple words", "clarify", ex={"complexity": "low"}, c=1, agent=OF_EN, notes="")
T("en", "sorry im not good with these banking terms make it simple", "clarify", ex={"complexity": "low"}, c=2, agent=OF_EN, notes="")
T("hinglish", "aasan bhasha mein bolo na ye pre approved kya hota hai", "clarify", ex={"complexity": "low"}, c=2, pend="card_offer", agent=OF_HG, notes="")
T("hi", "आसान शब्दों में समझाओ लिमिट वाला", "clarify", ex={"complexity": "low"}, c=1, agent=OL_HI, notes="")
# hard negatives: mention language/length but not a preference request
T("en", "my mother only speaks hindi she will call later for her account", "unknown", notes="mentions hindi but no preference for this call; explicit empty")
T("hi", "हां बोलो छोटा सा काम है बैलेंस देखना था", "check_balance", notes="'chhota sa kaam' is not a verbosity request")
T("en", "explain later first send the money", "transfer", u=1, notes="'explain' present but this is prioritisation, not verbosity preference")

from common import T

OL_EN = "sorry that amount is above your daily transfer limit you can send up to twenty five thousand today"
OL_HI = "माफ़ कीजिए, यह राशि आपकी डेली लिमिट से ज़्यादा है, आज आप पच्चीस हज़ार तक भेज सकते हैं"
OF_EN = "you also have a pre approved credit card with a limit of one point five lakh would you like to know more"
OF_HI = "आपके लिए एक प्री-अप्रूव्ड क्रेडिट कार्ड है, डेढ़ लाख की लिमिट के साथ, क्या आप जानना चाहेंगे?"
OF_HG = "aapke liye ek pre approved credit card offer hai ek lakh pachas hazaar limit ke saath interested hain"

# ---------- transfer_limit ----------
T("en", "whats my daily transfer limited", "transfer_limit", notes="ASR artefact limit->limited")
T("en", "how much can i send to day", "transfer_limit", notes="no word 'limit'")
T("en", "why can't i send more is there some cap", "transfer_limit", c=1, f=1, agent=OL_EN, notes="'cap' synonym; mild frustration after over-limit")
T("en", "how much is left for today then", "transfer_limit", agent=OL_EN, notes="follow-up on remaining limit")
T("en", "can i increase the limit just for today", "transfer_limit", u=1, e=2, agent=OL_EN, notes="limit increase request")
T("en", "wait what limit i never set any limit", "transfer_limit", c=2, f=1, agent=OL_EN, notes="confusion + mild irritation; repeating agent word")
T("en", "is forty thousand within my limit", "transfer_limit", amt=40000, notes="amount inside a limit question; intent is limit query")
T("en", "the limit i mean the transfer one not the card", "transfer_limit", c=0, notes="disambiguating which limit")
T("en", "so how much more can i do", "transfer_limit", e=1, notes="")
T("hi", "मेरी लिमिट कितनी है ट्रांसफर की", "transfer_limit", notes="")
T("hi", "एक दिन में कितना भेज सकते हैं", "transfer_limit", notes="no limit word")
T("hi", "लिमिट क्यों है मेरा अपना पैसा है", "transfer_limit", f=2, agent=OL_HI, notes="frustration expressed as indignation, no swear words")
T("hi", "तो आज कितना और भेज सकता हूं", "transfer_limit", agent=OL_HI, notes="")
T("hi", "लिमिट बढ़ा दो ना थोड़ी", "transfer_limit", e=2, agent=OL_HI, notes="")
T("hi", "पच्चीस हज़ार तक ही क्यों", "transfer_limit", c=1, amt=None, agent=OL_HI, notes="number echoes agent's limit, not an amount to send")
T("hi", "डेली लिमिट कितनी है मेरी", "transfer_limit", notes="")
T("hinglish", "limit kitni hai daily ki", "transfer_limit", notes="")
T("hinglish", "aaj ka kitna bacha hai limit mein", "transfer_limit", notes="'bacha' also appears in balance queries; here limit")
T("hinglish", "yaar limit kab reset hogi kal subah", "transfer_limit", e=2, notes="")
T("hinglish", "kitna tak bhej sakta hoon ek baar mein", "transfer_limit", notes="")

# ---------- card_offer ----------
T("en", "i got some sms about a credit card offer something", "card_offer", e=2, notes="customer-initiated")
T("en", "whats the annual fee on that", "card_offer", e=2, pend="card_offer", agent=OF_EN, notes="engaged question about offer")
T("en", "okay okay tell me", "card_offer", e=2, pend="card_offer", agent=OF_EN, notes="also affirm-like; interest in details -> card_offer")
T("en", "is it lifetime free or", "card_offer", e=2, pend="card_offer", agent=OF_EN, notes="trailing off")
T("en", "so how do i apply for the uh credit card", "card_offer", notes="")
T("en", "pre approved means i don't need to submit salary slip", "card_offer", c=1, e=2, pend="card_offer", agent=OF_EN, notes="checking understanding; between clarify and card_offer")
T("en", "what's the limit on the card one point five you said", "card_offer", e=2, pend="card_offer", agent=OF_EN, notes="'limit' here is card limit, not transfer")
T("en", "am i a legible for credit card", "card_offer", notes="")
T("en", "fine sign me up", "card_offer", e=1, pend="card_offer", agent=OF_EN, notes="acceptance; could be affirm -- labelled card_offer since it's apply intent. Arguably affirm")
T("hi", "क्रेडिट कार्ड वाला ऑफर क्या है", "card_offer", e=2, notes="")
T("hi", "इसका कोई चार्ज तो नहीं लगेगा", "card_offer", e=2, pend="card_offer", agent=OF_HI, notes="नहीं inside a question, not deny")
T("hi", "कार्ड के लिए क्या करना पड़ेगा", "card_offer", pend="card_offer", agent=OF_HI, notes="")
T("hi", "मुझे भी क्रेडिट कार्ड चाहिए था", "card_offer", notes="")
T("hi", "हां बताओ कार्ड के बारे में", "card_offer", e=2, pend="card_offer", agent=OF_HI, notes="haan + request for details")
T("hi", "लिमिट कितनी मिलेगी कार्ड पे", "card_offer", pend="card_offer", agent=OF_HI, notes="card limit")
T("hinglish", "credit card ka offer aaya tha message pe", "card_offer", notes="")
T("hinglish", "acha card ka kya fees hai", "card_offer", e=2, pend="card_offer", agent=OF_HG, notes="")
T("hinglish", "theek hai apply kar do card", "card_offer", pend="card_offer", agent=OF_HG, notes="acceptance phrased as apply; arguably affirm")

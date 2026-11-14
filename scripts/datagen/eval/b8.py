from common import T
from b3 import OF_EN, OF_HI, OF_HG, OL_EN, OL_HI
from b2 import AA_EN, AA_HI, AA_HG
from b4 import CF_HI, CF_EN, CF_HG
from b6 import BAL_EN, BAL_HI, LONG_HI

# ----- urgency-heavy, barge-ins -----
T("en", "listen i don't have time just tell me if the transfer can go through or not", "transfer_limit", u=2, f=1, b=True, agent=LONG_HI, notes="rushed businessperson")
T("en", "quick question what's my balance", "check_balance", u=1, notes="'quick' mild urgency")
T("en", "hey sorry i'm at the counter right now how much is in the account", "check_balance", u=2, e=1, notes="situational urgency")
T("en", "no no stop i don't want the card just do the transfer", "deny", u=1, f=1, b=True, pend="card_offer", agent=OF_EN, notes="interrupts offer")
T("en", "yes confirm confirm", "affirm", u=2, b=True, pend="transfer", agent=CF_EN, notes="")
T("en", "my rent is due today can i send forty thousand", "transfer", u=1, amt=40000, notes="'can i' phrasing but transfer intent")
T("en", "the shop guy is waiting send 2500", "transfer", u=2, amt=2500, notes="")
T("hi", "जल्दी बताओ बैलेंस गाड़ी चला रहा हूं", "check_balance", u=2, notes="")
T("hi", "हां हां करो करो जल्दी", "affirm", u=2, b=True, pend="transfer", agent=CF_HI, notes="")
T("hi", "आज ही चाहिए पैसा उसको शाम तक", "transfer", u=1, notes="")
T("hi", "रुको रुको मत भेजो", "deny", u=2, b=True, pend="transfer", agent=CF_HI, notes="")
T("hi", "बस बस समझ गई कार्ड नहीं चाहिए", "deny", f=1, b=True, pend="card_offer", agent=OF_HI, notes="'samajh gayi' + deny; female speaker")
T("hi", "फटाफट पच्चीस हज़ार डालो", "transfer", u=2, amt=25000, notes="'phataphat' urgency word")
T("hi", "प्लीज़ जल्दी कीजिए मेरी माँ का ऑपरेशन है", "unknown", u=2, e=2, agent="कृपया प्रतीक्षा करें", notes="urgency, no new task intent")
T("hi", "नमस्ते जी मुझे पैसा भेजना है अभी तुरंत", "transfer", u=2, notes="greeting + task; label task")
T("hinglish", "jaldi bolo kitna hai balance", "check_balance", u=2, notes="")
T("hinglish", "haan haan bhejo bhejo", "affirm", u=1, b=True, pend="transfer", agent=CF_HG, notes="")
T("hinglish", "abhi ke abhi chahiye bhai please", "transfer", u=2, pend="transfer", agent=AA_HG, slot="amount", notes="amount slot not answered")
# ----- more Hindi greet/goodbye/card_offer/limit coverage -----
T("hi", "हेलो जी गुड मॉर्निंग", "greet", notes="English greeting in Devanagari")
T("hi", "हां जी कहिए किसका फोन है", "greet", c=1, notes="elderly thinks bank called them")
T("hi", "अच्छा ठीक है धन्यवाद जी", "goodbye", agent=BAL_HI, notes="")
T("hi", "सालाना फीस कितनी है इसकी", "card_offer", e=2, pend="card_offer", agent=OF_HI, notes="")
T("hi", "हां बनवा दो कार्ड", "card_offer", pend="card_offer", agent=OF_HI, notes="acceptance as card request; arguably affirm")
T("hi", "क्रेडिट कार्ड पर कितना ब्याज लगता है", "card_offer", e=2, pend="card_offer", agent=OF_HI, notes="")
T("en", "the card offer the one with one fifty limit is that still on", "card_offer", e=2, amt=None, notes="'one fifty' = card limit, not an amount to send")
T("hi", "कल फिर से भेज पाऊंगा ना पूरा चालीस", "transfer_limit", amt=40000, e=2, agent=OL_HI, notes="asks about tomorrow's limit")

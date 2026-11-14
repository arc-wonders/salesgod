"""Fixed wording for the Indian negotiation ladder: every bank has three tones (professional / mid / casual) and three languages
(en, hi, hinglish). Each row is (en, hi, hinglish). Rows are picked deterministically by NegState.used, so a call never repeats a line.

Tone rules (checked by tests/test_tone_*.py):
  professional  "Sir/सर" + formal "aap", no slang, never bhai/yaar/boss.
  mid           friendly "aap" + natural Hinglish, no slang addresses.
  casual        warm street-friendly "Bhai/Boss/Yaar", short; still "aap" verbs, never तू/तेरी/तुम.
Placeholders: {p} a price, {n} the caller's number, {v} value text, {d} per-day value.
"""

from __future__ import annotations

TONES = ("professional", "mid", "casual")
LANGS = ("en", "hi", "hinglish")


def _b(**by_tone: list[tuple[str, str, str]]) -> dict[str, list[tuple[str, str, str]]]:
    return by_tone


# ---- a firm, warm "no" on the first push (no price cut yet)
NO_FIRST = _b(
    professional=[
        ("Sir, I understand your concern, but I am not able to reduce the price at this stage.",
         "सर, मैं आपकी बात समझता हूँ, लेकिन इस स्तर पर मैं कीमत में कमी नहीं कर पाऊँगा।",
         "Sir, main aapki baat samajhta hoon, lekin is stage par main price mein kami nahi kar paunga."),
        ("Sir, thank you for being candid; the price is already carefully set, so I cannot change it right away.",
         "सर, स्पष्ट बताने के लिए धन्यवाद; कीमत पहले ही सोच-समझकर तय की गई है, इसलिए अभी बदल नहीं सकता।",
         "Sir, saaf bataane ke liye dhanyavaad; price pehle hi soch-samajhkar tay hui hai, isliye abhi badal nahi sakta."),
    ],
    mid=[
        ("Look, I get it, but cutting the price right now is difficult.",
         "देखिए, मैं समझता हूँ, पर अभी कीमत कम करना मुश्किल है।",
         "Dekhiye, main samajhta hoon, par abhi price kam karna mushkil hai."),
        ("I hear you, but I can't just drop the price on the first ask.",
         "मैं आपकी बात सुन रहा हूँ, पर पहली बार में ही कीमत नहीं गिरा सकता।",
         "Main aapki baat sun raha hoon, par pehli baar mein hi price nahi gira sakta."),
    ],
    casual=[
        ("Bhai, I get it, but cutting the price right now is tough.",
         "भाई, समझ रहा हूँ, पर अभी कीमत कम करना मुश्किल है।",
         "Bhai, samajh raha hoon, par abhi price kam karna mushkil hai."),
        ("Boss, I hear you, but I can't just drop it on the first ask.",
         "बॉस, आपकी बात सुन रहा हूँ, पर पहली बार में ही नहीं गिरा सकता।",
         "Boss, aapki baat sun raha hoon, par pehli baar mein hi nahi gira sakta."),
    ],
)
# the caller named a number below the offer on the first push: say no to THAT number, once
NO_NUM = _b(
    professional=[("Sir, {n} is difficult for me on this plan, I am sorry.", "सर, इस प्लान पर {n} करना मेरे लिए कठिन है, क्षमा कीजिए।",
                   "Sir, is plan par {n} karna mere liye kathin hai, kshama kijiye.")],
    mid=[("{n} is too low for me to do, honestly.", "{n} में करना तो मेरे लिए मुश्किल है, सच में।", "{n} mein karna mere liye mushkil hai, sach mein.")],
    casual=[("Bhai, {n} is tough, honestly.", "भाई, {n} में बनना मुश्किल है, सच में।", "Bhai, {n} mein banna mushkil hai, sach mein.")],
)
# when they named a number and we still can't meet it: say no to it, then move one step
NO_NUM_THEN = _b(
    professional=[("Sir, {n} is below what I am able to do, but let me see what I can offer.",
                   "सर, {n} मेरी सीमा से नीचे है, पर देखता हूँ मैं क्या कर सकता हूँ।",
                   "Sir, {n} meri seema se neeche hai, par dekhta hoon main kya kar sakta hoon."),
              ("Sir, {n} is a little too low for us; allow me to come closer to you instead.",
               "सर, {n} हमारे लिए थोड़ा कम है; मैं ही आपके थोड़ा नज़दीक आता हूँ।",
               "Sir, {n} hamare liye thoda kam hai; main hi aapke thoda nazdeek aata hoon.")],
    mid=[("{n} is below what I can do, but let me come down a bit.", "{n} मेरे बस से नीचे है, पर मैं थोड़ा नीचे आता हूँ।",
          "{n} mere bas se neeche hai, par main thoda neeche aata hoon."),
         ("{n} is too low, but I'll meet you part of the way.", "{n} बहुत कम है, पर मैं रास्ते में थोड़ा मिलता हूँ।", "{n} bahut kam hai, par main raaste mein thoda milta hoon.")],
    casual=[("Bhai, {n} is too low, but let me come down a bit.", "भाई, {n} बहुत कम है, पर मैं थोड़ा नीचे आता हूँ।", "Bhai, {n} bahut kam hai, par main thoda neeche aata hoon."),
            ("Boss, {n} is too low, but I'll move a little for you.", "बॉस, {n} बहुत कम है, पर आपके लिए थोड़ा हिलता हूँ।", "Boss, {n} bahut kam hai, par aapke liye thoda hilta hoon.")],
)
# value: what is included (benefits) and the per-day figure
VALUE = _b(
    professional=[("Sir, what you receive: {v}.", "सर, आपको मिलता है: {v}।", "Sir, aapko milta hai: {v}.")],
    mid=[("Here is what you get: {v}.", "इसमें आपको मिलता है: {v}।", "Isme aapko milta hai: {v}.")],
    casual=[("You get {v}, that is the deal.", "इसमें मिलता है: {v}, बस यही डील है।", "Isme milta hai: {v}, bas yehi deal hai.")],
)
VALUE_NEED = _b(
    professional=[("Sir, you mentioned \"{need}\"; the part that helps is {v}.", "सर, आपने \"{need}\" बताया था; इसमें काम का हिस्सा है: {v}।",
                   "Sir, aapne \"{need}\" bataya tha; isme kaam ka hissa hai: {v}.")],
    mid=[("You mentioned \"{need}\", so the useful part is {v}.", "आपने \"{need}\" बताया था, तो काम का हिस्सा है: {v}।", "Aapne \"{need}\" bataya tha, to kaam ka hissa hai: {v}.")],
    casual=[("You said \"{need}\", so the part for you is {v}.", "आपने \"{need}\" बोला था, तो आपके काम का है: {v}।", "Aapne \"{need}\" bola tha, to aapke kaam ka hai: {v}.")],
)
PER_DAY = _b(
    professional=[("It comes to only {d} a day.", "यह प्रतिदिन केवल {d} पड़ता है।", "Yeh roz ka sirf {d} padta hai.")],
    mid=[("That works out to about {d} a day.", "यानी रोज़ का करीब {d}।", "Yaani roz ka kareeb {d}.")],
    casual=[("That is just {d} a day.", "रोज़ का बस {d}।", "Roz ka bas {d}.")],
)
# the first no also asks what they had in mind
ASK_BUDGET = _b(
    professional=[("Sir, may I know what budget you had in mind?", "सर, क्या मैं जान सकता हूँ कि आपका बजट कितना है?", "Sir, kya main jaan sakta hoon aapka budget kitna hai?")],
    mid=[("What budget did you have in mind?", "आपका बजट कितना है?", "Aapka budget kitna hai?")],
    casual=[("Bhai, what budget did you have in mind?", "भाई, आपका बजट कितना है?", "Bhai, aapka budget kitna hai?")],
)
ASK_COME_UP = _b(
    professional=[("Sir, could you perhaps tell me a figure closer to the plan?", "सर, क्या आप प्लान के थोड़ा नज़दीक की कोई रकम बता सकते हैं?",
                   "Sir, kya aap plan ke thoda nazdeek ki koi rakam bata sakte hain?")],
    mid=[("Can you come up a little from there?", "क्या आप वहाँ से थोड़ा ऊपर आ सकते हैं?", "Kya aap wahan se thoda upar aa sakte hain?")],
    casual=[("Bhai, can you come up a little?", "भाई, थोड़ा ऊपर आ सकते हैं क्या?", "Bhai, thoda upar aa sakte hain kya?")],
)
ASK_KEEP = _b(  # first push, committed buyer who just wants a number
    professional=[("Sir, what figure would suit you?", "सर, आपको कितनी रकम ठीक लगेगी?", "Sir, aapko kitni rakam theek lagegi?")],
    mid=[("What number would work for you?", "आपके हिसाब से कितना ठीक रहेगा?", "Aapke hisaab se kitna theek rahega?")],
    casual=[("Bhai, what number works for you?", "भाई, आपके हिसाब से कितना ठीक रहेगा?", "Bhai, aapke hisaab se kitna theek rahega?")],
)
COMPARE_LEAD = _b(
    professional=[("Sir, comparing is wise; I can only speak for ours.", "सर, तुलना करना समझदारी है; मैं केवल अपने बारे में बता सकता हूँ।",
                   "Sir, compare karna samajhdaari hai; main sirf apne baare mein bata sakta hoon.")],
    mid=[("Good to compare; I can only speak for ours.", "तुलना करना अच्छा है; मैं बस अपने बारे में बता सकता हूँ।", "Compare karna achha hai; main bas apne baare mein bata sakta hoon.")],
    casual=[("Bhai, compare karo, good; I can only tell you about ours.", "भाई, तुलना करना अच्छा है; मैं बस अपने बारे में बता सकता हूँ।",
             "Bhai, compare karna achha hai; main bas apne baare mein bata sakta hoon.")],
)

# ---- a concession (one per push), then a commitment ask
STEP = _b(
    professional=[
        ("Sir, from my side, I can offer you {p}.", "सर, अपनी ओर से मैं आपको {p} में दे सकता हूँ।", "Sir, apni taraf se main aapko {p} mein de sakta hoon."),
        ("Sir, specially for you, I can bring it to {p}.", "सर, विशेष रूप से आपके लिए मैं इसे {p} कर सकता हूँ।", "Sir, vishesh roop se aapke liye main ise {p} kar sakta hoon."),
        ("Sir, I can stretch a little more, to {p}.", "सर, मैं थोड़ा और बढ़ सकता हूँ, {p} तक।", "Sir, main thoda aur kheench sakta hoon, {p} tak."),
        ("Sir, one more small step from me: {p}.", "सर, मेरी ओर से एक और छोटा कदम: {p}।", "Sir, meri taraf se ek aur chhota kadam: {p}."),
    ],
    mid=[
        ("Look, from my side I can do {p} for you.", "देखिए, मैं अपनी तरफ़ से आपके लिए {p} कर सकता हूँ।", "Dekhiye, main apni taraf se aapke liye {p} kar sakta hoon."),
        ("Okay, only for you, I'll make it {p}.", "ठीक है, सिर्फ़ आपके लिए {p} कर देता हूँ।", "Theek hai, sirf aapke liye {p} kar deta hoon."),
        ("Let me stretch a bit more: {p}.", "थोड़ा और खींचता हूँ: {p}।", "Thoda aur kheenchta hoon: {p}."),
        ("One more small step down, to {p}.", "एक छोटा क़दम और, {p}।", "Ek chhota kadam aur, {p}."),
    ],
    casual=[
        ("Bhai, I'll do {p} for you.", "भाई, आपके लिए {p} कर देता हूँ।", "Bhai, aapke liye {p} kar deta hoon."),
        ("Boss, only for you, {p}.", "बॉस, सिर्फ़ आपके लिए, {p}।", "Boss, sirf aapke liye, {p}."),
        ("Okay yaar, a bit more, {p}.", "ठीक है यार, थोड़ा और, {p}।", "Theek hai yaar, thoda aur, {p}."),
        ("Bhai, one more step, {p}.", "भाई, एक क़दम और, {p}।", "Bhai, ek kadam aur, {p}."),
    ],
)
# said on a step that is not the last: honest, no false "last offer"
NOT_LAST = _b(
    professional=[("Going below that would be difficult for me.", "इससे नीचे जाना मेरे लिए कठिन होगा।", "Isse neeche jaana mere liye mushkil hoga."),
                  ("I would find it hard to go much lower.", "इससे बहुत कम करना कठिन होगा।", "Isse bahut kam karna mushkil hoga.")],
    mid=[("Going below that is difficult.", "इससे नीचे मुश्किल है।", "Isse neeche mushkil hai."),
         ("Much lower than this is hard for me.", "इससे बहुत नीचे मेरे लिए मुश्किल है।", "Isse bahut neeche mere liye mushkil hai.")],
    casual=[("Lower than that is tough, bhai.", "इससे नीचे मुश्किल है, भाई।", "Isse neeche mushkil hai, bhai."),
            ("Much lower is hard, boss.", "इससे बहुत नीचे मुश्किल है, बॉस।", "Isse bahut neeche mushkil hai, boss.")],
)
MID = _b(
    professional=[("Sir, let us meet in the middle: {p}.", "सर, चलिए बीच का रास्ता निकालते हैं: {p}।", "Sir, chaliye beech ka raasta nikaalte hain: {p}.")],
    mid=[("Let's meet in the middle: {p}.", "चलिए बीच का कर लेते हैं: {p}।", "Chaliye, beech ka kar lete hain: {p}.")],
    casual=[("Okay bhai, let's meet in the middle: {p}.", "ठीक है भाई, बीच का कर लेते हैं: {p}।", "Theek hai bhai, beech ka kar lete hain: {p}.")],
)
FINAL = _b(
    professional=[("Sir, {p} is our final price; it is not possible to go below it.", "सर, {p} हमारी अंतिम कीमत है; इससे नीचे जाना संभव नहीं है।",
                   "Sir, {p} hamari antim keemat hai; isse neeche jaana sambhav nahi hai.")],
    mid=[("{p} is final; I can't go below that.", "{p} फ़ाइनल है, इससे नीचे नहीं हो पाएगा।", "{p} final hai, isse neeche nahi ho payega.")],
    casual=[("Bhai, {p} is final, can't go below that.", "भाई, {p} फ़ाइनल है, इससे नीचे नहीं होगा।", "Bhai, {p} final hai, isse neeche nahi hoga.")],
)
ACCEPT = _b(
    professional=[
        ("Sir, agreed, {p} it is; thank you.", "सर, ठीक है, {p} पक्का; धन्यवाद।", "Sir, theek hai, {p} pakka; dhanyavaad."),
        ("Sir, that works for me: {p}.", "सर, यह मेरे लिए ठीक है: {p}।", "Sir, yeh mere liye theek hai: {p}."),
    ],
    mid=[
        ("Done, {p} it is.", "ठीक है, {p} पक्का।", "Done, {p} pakka."),
        ("Alright, {p} works for me.", "चलिए, {p} में बात पक्की।", "Chaliye, {p} mein baat pakki."),
    ],
    casual=[
        ("Done bhai, {p} pakka!", "डन भाई, {p} पक्का!", "Done bhai, {p} pakka!"),
        ("Deal boss, {p}!", "डील बॉस, {p}!", "Deal boss, {p}!"),
    ],
)
ASK = _b(
    professional=[
        ("Shall I send the payment link at {p}, Sir?", "सर, क्या मैं {p} पर पेमेंट लिंक भेज दूँ?", "Sir, kya main {p} par payment link bhej doon?"),
        ("Sir, would you join today at {p}?", "सर, क्या आप {p} पर आज जॉइन करेंगे?", "Sir, kya aap {p} par aaj join karenge?"),
        ("May I confirm {p} for you, Sir?", "सर, क्या मैं आपके लिए {p} पक्का कर दूँ?", "Sir, kya main aapke liye {p} pakka kar doon?"),
        ("Sir, shall I SMS the link for {p}?", "सर, क्या मैं {p} का लिंक SMS कर दूँ?", "Sir, kya main {p} ka link SMS kar doon?"),
    ],
    mid=[
        ("If I make it {p}, will you join today?", "{p} कर दूँ तो आज जॉइन कर लेंगे?", "{p} kar doon to aaj join kar lenge?"),
        ("Shall I lock {p} and send the link?", "क्या मैं {p} पक्का कर दूँ और लिंक भेज दूँ?", "Kya main {p} pakka kar doon aur link bhej doon?"),
        ("Is {p} a deal, then?", "तो {p} में बात पक्की?", "To {p} mein baat pakki?"),
        ("Want me to SMS the link for {p} now?", "{p} का लिंक अभी SMS कर दूँ?", "{p} ka link abhi SMS kar doon?"),
    ],
    casual=[
        ("{p}, then you join today?", "{p} में आज ही ले लो?", "{p} mein aaj hi le lo?"),
        ("Bhai, {p} locked? I send the link?", "भाई, {p} पक्का? लिंक भेज दूँ?", "Bhai, {p} pakka? Link bhej doon?"),
        ("Boss, {p} deal?", "बॉस, {p} में डील?", "Boss, {p} mein deal?"),
        ("Shall I SMS the {p} link, bhai?", "भाई, {p} का लिंक SMS कर दूँ?", "Bhai, {p} ka link SMS kar doon?"),
    ],
)
PERK_LEAD = _b(
    professional=[("Sir, instead of a cash reduction, allow me to add something for you.", "सर, कीमत घटाने के बजाय मैं आपके लिए कुछ जोड़ देता हूँ।",
                   "Sir, price ghatane ke bajaye main aapke liye kuch jod deta hoon.")],
    mid=[("Instead of cutting more, let me add one thing for you.", "और घटाने की जगह मैं एक चीज़ जोड़ देता हूँ।", "Aur ghatane ki jagah main ek cheez jod deta hoon.")],
    casual=[("Bhai, no more cut, but I add one thing free.", "भाई, और कम नहीं, पर एक चीज़ फ्री जोड़ता हूँ।", "Bhai, aur kam nahi, par ek cheez free jodta hoon.")],
)
SWEET_LEAD = _b(
    professional=[("Sir, the price cannot move further, but this may help.", "सर, कीमत इससे आगे नहीं हिल सकती, पर यह काम आ सकता है।",
                   "Sir, price isse aage nahi hil sakta, par yeh kaam aa sakta hai.")],
    mid=[("The price can't move more, but this may help.", "कीमत इससे आगे नहीं हिल सकती, पर एक चीज़ शायद काम आए।", "Keemat isse aage nahi hil sakti, par ek cheez shayad kaam aaye.")],
    casual=[("Bhai, price can't move more, but this may help.", "भाई, कीमत और नहीं हिलेगी, पर यह काम आ सकता है।", "Bhai, keemat aur nahi hilegi, par yeh kaam aa sakta hai.")],
)
LOWEST = _b(
    professional=[("Sir, {p} is the lowest I am able to offer.", "सर, {p} मेरी सबसे कम कीमत है।", "Sir, {p} meri sabse kam keemat hai.")],
    mid=[("{p} is already the lowest I can go.", "{p} मेरी सबसे कम कीमत है।", "{p} meri sabse kam keemat hai.")],
    casual=[("Bhai, {p} is already my lowest.", "भाई, {p} पहले से मेरी सबसे कम है।", "Bhai, {p} pehle se meri sabse kam hai.")],
)
CLOSE_WARM = _b(
    professional=[
        ("Sir, I fully understand. Shall I call you back at a convenient time, or send the details? There is no pressure.",
         "सर, मैं पूरी तरह समझता हूँ। क्या मैं सुविधाजनक समय पर कॉल करूँ, या विवरण भेज दूँ? कोई दबाव नहीं है।",
         "Sir, main poori tarah samajhta hoon. Kya main suvidha ke time par call karoon, ya details bhej doon? Koi pressure nahi hai."),
    ],
    mid=[
        ("Of course, that's fine. Want me to call you back another time, or send the details? No pressure at all.",
         "बिल्कुल, कोई बात नहीं। क्या मैं किसी और समय कॉल करूँ, या डिटेल भेज दूँ? कोई दबाव नहीं है।",
         "Bilkul, koi baat nahi. Kya main kisi aur time call karoon, ya details bhej doon? Koi pressure nahi hai."),
    ],
    casual=[
        ("Bhai, no problem at all. Want me to call you back later or send the details? No pressure.",
         "भाई, कोई बात नहीं। बाद में कॉल करूँ या डिटेल भेज दूँ? कोई दबाव नहीं।",
         "Bhai, koi baat nahi. Baad mein call karoon ya details bhej doon? Koi pressure nahi."),
    ],
)
CLOSED_AGAIN = _b(
    professional=[
        ("Sir, I understand. The price stays as agreed, and the link is ready whenever you wish.", "सर, मैं समझता हूँ। कीमत वही रहेगी जो तय हुई, और लिंक जब चाहें तैयार है।",
         "Sir, main samajhta hoon. Price wahi rahegi jo tay hui, aur link jab chahein tayyar hai."),
        ("Sir, there is no hurry; I can also call you at a time that suits you.", "सर, कोई जल्दी नहीं है; मैं आपके सुविधाजनक समय पर कॉल भी कर सकता हूँ।",
         "Sir, koi jaldi nahi hai; main aapke suvidhajanak samay par call bhi kar sakta hoon."),
        ("Sir, thank you for hearing me out; whatever you decide, it was a pleasure.", "सर, मेरी बात सुनने के लिए धन्यवाद; आप जो भी तय करें, आपसे बात करके अच्छा लगा।",
         "Sir, meri baat sunne ke liye dhanyavaad; aap jo bhi tay karein, aapse baat karke achha laga."),
    ],
    mid=[
        ("I hear you. The price stays where we agreed, and the link is ready whenever you want.", "मैं समझता हूँ। कीमत वही रहेगी जो तय हुई, और लिंक जब चाहें तैयार है।",
         "Main samajhta hoon. Keemat wahi rahegi jo tay hui, aur link jab chahein ready hai."),
        ("No hurry at all; I can also call you back when it suits you.", "कोई जल्दी नहीं है; चाहें तो मैं आपके सुविधा वाले समय पर वापस कॉल कर सकता हूँ।",
         "Koi jaldi nahi hai; chahein to main aapke suvidha wale time par wapas call kar sakta hoon."),
        ("Thanks for hearing me out; whatever you decide, it was good talking to you.", "मेरी बात सुनने के लिए शुक्रिया; आप जो भी तय करें, आपसे बात करके अच्छा लगा।",
         "Meri baat sunne ke liye shukriya; aap jo bhi tay karein, aapse baat karke achha laga."),
    ],
    casual=[
        ("Bhai, the price stays as we agreed, and the link is ready when you want.", "भाई, कीमत वही रहेगी जो तय हुई, और लिंक जब चाहो तैयार है।",
         "Bhai, keemat wahi rahegi jo tay hui, aur link jab chahein ready hai."),
        ("No hurry, boss; I can call you back whenever suits you.", "कोई जल्दी नहीं, बॉस; जब आपको सही लगे मैं वापस कॉल कर लूँगा।",
         "Koi jaldi nahi, boss; jab aapko sahi lage main wapas call kar loonga."),
        ("Thanks for the chat, bhai; whatever you decide, it was good talking.", "बात करने के लिए शुक्रिया भाई; जो भी तय करें, अच्छा लगा बात करके।",
         "Baat karne ke liye shukriya bhai; jo bhi tay karein, achha laga baat karke."),
    ],
)

# "can I just pay 33 a day?": one payment covers the whole plan, the per-day figure is only the maths
PER_DAY_Q = _b(
    professional=[("Sir, it is one payment of {p} for {u}; the {d} a day is only the arithmetic, not a daily payment.",
                   "सर, यह {u} के लिए एक ही भुगतान {p} है; {d} रोज़ का केवल हिसाब है, रोज़ का भुगतान नहीं।",
                   "Sir, yeh {u} ke liye ek hi payment {p} hai; {d} roz ka sirf hisaab hai, roz ka payment nahi.")],
    mid=[("It's one payment of {p} for {u}; the {d} a day is just the maths, not a daily payment.",
          "यह {u} के लिए एक ही पेमेंट {p} है; {d} रोज़ का बस हिसाब है, रोज़ का पेमेंट नहीं।",
          "Yeh {u} ke liye ek hi payment {p} hai; {d} roz ka bas hisaab hai, roz ka payment nahi.")],
    casual=[("Bhai, it's one payment of {p} for {u}; {d} a day is just the maths, not a daily payment.",
             "भाई, यह {u} के लिए एक ही पेमेंट {p} है; {d} रोज़ का बस हिसाब है, रोज़ का पेमेंट नहीं।",
             "Bhai, yeh {u} ke liye ek hi payment {p} hai; {d} roz ka bas hisaab hai, roz ka payment nahi.")],
)

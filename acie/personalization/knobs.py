"""Explicit-request patterns for the newer expression knobs (regional variant, explanation style,
information density, reassurance). Same conventions as policy._EXPLICIT: patterns run on normalized
text padded with spaces (" tok tok "), Devanagari without nukta. (key, value) entries in BLOCK cancel
that value when the customer negates it ("not worried", "delhi wali nahi")."""

from __future__ import annotations

import re

_NEG = r"(?:nahi|nahin|nhi|नहीं|नही|mat|मत)"
_IN = r"(?:me|mein|mai|main|mei|men|में|मे|मैं)"
_WALI = r"(?:wali|waali|wala|vali|vala|style|andaz|andaaz|andaaj|ki|ka|jaisi|jaisa|tarah|tarike|accent|slang|boli|bhasha|वाली|वाला|स्टाइल|अंदाज|की|का|जैसी|जैसा)"


def _rx(*alts: str) -> re.Pattern[str]:
    return re.compile("|".join(alts))


# value -> city/community words
_PLACES: dict[str, str] = {
    "delhi": r"(?:delhi|dilli|दिल्ली|दिल्लीवाली|dilliwali)",
    "mumbai": r"(?:mumbai|mumbaiya|mumbaikar|bombay|bambaiya|bambai|tapori|मुंबई|मुंबईया|बंबईया|टपोरी)",
    "lucknow": r"(?:lucknow|lucknowi|lakhnavi|nawabi|लखनऊ|लखनवी|नवाबी)",
    "bihar": r"(?:bihar|bihari|patna|bhojpuri|बिहार|बिहारी|पटना|भोजपुरी)",
    "south_indian_english": (r"(?:(?:tamil|telugu|kannada|malayali|malayalam|kerala|chennai|madrasi|south indian|south indian)"
                             r" (?:accent|style|english|wali|wala|jaisi|jaisa|tone))"),
}

_ALTS_PLACE: dict[str, list[str]] = {}
for _v, _p in _PLACES.items():
    if _v == "south_indian_english":
        _ALTS_PLACE[_v] = [rf" {_p} ", r" south indian english ", r" (?:tamil|telugu|kannada|malayali|madrasi) english ",
                           r" english (?:with|in) (?:a )?(?:tamil|telugu|kannada|south indian|malayali) (?:accent|style) ",
                           r" english (?:in|with) (?:tamil|south indian) "]
        continue
    _ALTS_PLACE[_v] = [rf" {_p} {_WALI} ", rf" {_p} (?:hindi|bhasha|boli|slang|style|accent) ",
                       rf" {_p} {_IN} (?:baat|bol|bolo|bolna|samjha|samjhao|बात|बोल|समझा)\S* ",
                       rf" (?:speak|talk|reply|answer) (?:like|in) (?:a )?{_p} "]
    if _v == "bihar":  # "bihari mein" is a complete request on its own
        _ALTS_PLACE[_v].append(rf" {_p} (?:mein|me|में) ")

EXPLICIT: list[tuple[re.Pattern[str], str, str]] = [
    (_rx(*_ALTS_PLACE[v]), "regional_variant", v) for v in _PLACES
] + [
    (_rx(r" (?:normal|standard|neutral|saaf|plain|shuddh|shudh|saadi|sadi) (?:hindi|english|bhasha|accent|boli) ",
         r" (?:koi )?(?:accent|slang) (?:nahi|nhi|mat) ", r" no (?:accent|slang) ",
         r" (?:सामान्य|साफ|शुद्ध|सादी) (?:हिंदी|भाषा) "), "regional_variant", "standard"),

    (_rx(r" example (?:de|dekar|dekar|deke|dekar|ke saath|ke sath|ke sang|se|do|dijiye|dena|batao|chahiye)\S* ",
         r" (?:ek |koi )?(?:example|udaharan|misaal|मिसाल|उदाहरण|एक्जाम्पल|एग्जाम्पल|एक्साम्पल) ",
         r" for (?:example|instance) ", r" (?:give|show|tell) (?:me )?(?:an |some )?examples? ",
         r" (?:example|उदाहरण|मिसाल) (?:दे|के साथ|से|देकर|दीजिए|दो)\S* ",
         r" example ke saath ", r" like what ", r" jaise ki ", r" kaise ek example "), "explanation_style", "example"),
    (_rx(r" step (?:by|wise|bi) step ", r" stepwise ", r" step wise ", r" one (?:step )?at a time ", r" one by one ",
         r" ek ek (?:step|karke|kar ke|cheez|point) ", r" ek (?:step|kadam) (?:ek )?(?:karke|kar ke|at a time) ",
         r" (?:point|steps?) (?:by point|wise) ", r" (?:ek|एक) (?:ek|एक) (?:step|स्टेप|करके) ",
         r" स्टेप बाय स्टेप ", r" स्टेप वाइज ", r" एक एक करके ", r" एक एक स्टेप ", r" kadam (?:ba|by) kadam ",
         r" क़दम ", r" कदम (?:बा|बाय) कदम ", r" pehle (?:kya|kaise) phir (?:kya|kaise) ", r" guide me through "),
     "explanation_style", "step_by_step"),
    (_rx(r" seedh[ae] (?:bata|bol|bol|keh|kah|samjha|bata)\S* ", r" seedha seedha ", r" sidha sidha ", r" sidha (?:bata|bol)\S* ",
         r" just tell me ", r" straight(?:forward)? (?:answer|answers) ", r" give me (?:the )?(?:answer|bottom line) ",
         r" no (?:story|lecture|explanation|explanations|background) ", r" without (?:explanation|explaining|background) ",
         r" bina (?:explain|samjhaye|samjhaaye|bhumika|kahani)\S* ", r" (?:kahani|bhumika) (?:mat|nahi) ",
         r" सीधा सीधा ", r" सीधा बता\S* ", r" सीधे बता\S* ", r" बिना (?:समझाए|भूमिका|कहानी) ", r" bottom line "),
     "explanation_style", "direct"),

    (_rx(r" sirf (?:zaroori|zaruri|jaruri|jaroori|important|imp|main|mukhya|kaam ki|kaam ka)\S* ",
         r" bas (?:zaroori|zaruri|jaruri|jaroori|important|main|mukhya|kaam ki)\S* ",
         r" only (?:the )?(?:important|essential|key|main|necessary|relevant) ",
         r" just (?:the )?(?:basics|essentials|gist|main points?|key points?|important (?:stuff|things|parts?)|highlights) ",
         r" (?:zyada|jyada|ज्यादा) (?:info|jankari|jaankari|detail|details|जानकारी)\S* (?:nahi|nhi|mat|नहीं|मत) ",
         r" (?:too|so) much (?:info|information|detail|details) ", r" keep it minimal ", r" bare minimum ",
         r" सिर्फ (?:जरूरी|जरुरी|ज़रूरी|मुख्य|काम की|काम का)\S* ", r" बस (?:जरूरी|जरुरी|मुख्य)\S* ",
         r" mukhya baat ", r" मुख्य बात ", r" kaam ki baat ", r" काम की बात ", r" gist "), "info_density", "low"),
    (_rx(r" (?:poori|puri|pura|saari|sari|sabhi|sab|पूरी|सारी|सभी|सब) (?:jaankari|jankari|info|information|details?|जानकारी|डिटेल)\S* ",
         r" (?:sab|sabkuch|सब) ?(?:kuch|कुछ) (?:bata|batao|bolo|samjha|samjhao|बता|बोल|समझा)\S* ",
         r" everything (?:about|you know|there is) ", r" all (?:the )?(?:details|information|info|specifics) ",
         r" (?:complete|full|entire|detailed|comprehensive) (?:information|info|details|picture|rundown|breakdown) ",
         r" ek ek detail ", r" हर (?:एक )?डिटेल ", r" in depth ", r" in-depth ", r" deep dive ", r" every detail ",
         r" poora (?:breakdown|byora|vivaran)\S* ", r" पूरा (?:ब्योरा|विवरण)\S* ", r" vistar se sab "),
     "info_density", "high"),

    (_rx(r" (?:mujhe |main |hume |humein |mai )?(?:dar|darr|dr|डर) (?:lag|lagta|lagti|raha|rahi|lg|लग)\S* ",
         r" (?:ghabra|ghabrahat|घबरा|घबराहट)\S* ", r" (?:chinta|chintit|fikar|fikr|चिंता|चिंतित|फिक्र)\S* ",
         r" (?:tension|tenshan|टेंशन) (?:ho|hai|me|mein|lag|है|हो)\S* ", r" (?:bharosa|bharosha|भरोसा) ",
         r" (?:worried|nervous|scared|afraid|anxious|apprehensive|terrified|uneasy) ",
         r" (?:is|was) (?:it|this|that) (?:really )?(?:safe|secure|legit|genuine|trustworthy) ",
         r" (?:safe|secure|surakshit|सुरक्षित) (?:hai|h|है)\S* (?:na|kya|ya|ना|क्या)? ",
         r" (?:fraud|scam|dhokha|धोखा)\S* (?:toh|to|तो)? (?:nahi|nhi|नहीं) ", r" (?:please |pls )?(?:assure|reassure|guarantee) ",
         r" (?:pakka|pakka|पक्का) (?:na|hai|h|ना|है)\S* ", r" kuch (?:galat|ulta) (?:toh|to|तो) (?:nahi|nhi|नहीं) "),
     "reassurance", "high"),
    (_rx(rf" (?:koi )?(?:tension|dar|darr|chinta|fikar|fikr|ghabrahat|टेंशन|डर|चिंता|फिक्र) {_NEG} ",
         r" (?:not|never|no) (?:worried|nervous|scared|afraid|anxious|worry|worries|fear) ", r" (?:im|i am) (?:not|never) (?:worried|nervous|scared|afraid|anxious) ",
         r" i know what (?:im|i am) doing ", r" no (?:need|reason) (?:to|for) (?:reassur|assur|worr)\S* ",
         r" (?:koi )?(?:chinta|tension|dar|डर|चिंता) (?:ki )?(?:baat )?(?:nahi|nhi|नहीं) "), "reassurance", "low"),
]

# A match cancels that (key, value): "not worried" is not anxiety, "delhi wali nahi" is not delhi.
BLOCK: list[tuple[re.Pattern[str], str, str]] = [
    (_rx(rf" {_NEG} (?:koi )?(?:dar|darr|chinta|tension|fikar|fikr|ghabra\S*|डर|चिंता|टेंशन|फिक्र)\S* ",
         rf" (?:dar|darr|chinta|fikar|fikr|tension|ghabrahat|टेंशन|डर|चिंता|फिक्र)\S* {_NEG} ",
         r" (?:not|never|no|dont|do not|isnt|arent|wasnt) (?:\w+ )?(?:worried|nervous|scared|afraid|anxious|apprehensive|uneasy|worry|safe|secure|legit|genuine) ",
         r" (?:im|i am|am) (?:not|never) (?:worried|nervous|scared|afraid|anxious) ",
         r" (?:safe|secure|surakshit)\S* (?:nahi|nhi|नहीं) ",
         r" (?:not|never|no|dont|do not) (?:assure|reassure|guarantee) "), "reassurance", "high"),
] + [
    (_rx(rf" (?:not|no|{_NEG}|dont want|do not want|nahi chahiye) {_p} ",
         rf" {_p} (?:{_WALI} )?(?:{_IN} )?{_NEG} "), "regional_variant", v)
    for v, _p in _PLACES.items()
] + [
    (_rx(r" (?:no|not|dont want|do not want|nahi chahiye|mat) (?:any )?(?:example|examples|udaharan|उदाहरण) ",
         r" (?:example|examples|udaharan|उदाहरण) (?:nahi|nhi|mat|नहीं|मत) ", r" no need (?:of|for) (?:an? )?examples? "), "explanation_style", "example"),
    (_rx(r" (?:no|not|dont want|do not want|nahi chahiye|mat) (?:step by step|stepwise) ", r" (?:step by step|stepwise) (?:nahi|nhi|mat|नहीं|मत) "),
     "explanation_style", "step_by_step"),
    (_rx(r" (?:not|no|nahi|mat) (?:poori|puri|saari|sari|all|full|complete) (?:jaankari|jankari|info|details?) ",
         r" (?:poori|puri|saari|sari) (?:jaankari|jankari|info|details?) (?:nahi|nhi|mat) ",
         r" (?:not|dont want|do not want) (?:all|every|full|complete) (?:the )?(?:details?|info|information) "), "info_density", "high"),
    (_rx(r" (?:not|no|nahi|mat) (?:sirf|only|just) (?:zaroori|important|essential|the basics) ",
         r" (?:sirf|only) (?:zaroori|important) (?:nahi|nhi|mat) "), "info_density", "low"),
]

"""Distress detection on garbled ASR: live call 2ccf3f1387 (gym sales call, Hindi).

POSITIVES must be a crisis (score >= DISTRESS_T); CONCERN cases must land in the middle band;
NEGATIVES (gym / slang / everyday Hindi-Hinglish-English) must stay below the concern band."""
import pytest

from acie.safety import CONCERN_T, DISTRESS_T, crisis_from_nlu, distress_score, is_concern, is_distress, skeletons
from acie.understanding import Understanding

POSITIVES = [
    # the live call, verbatim ASR
    "मेरा गोल है छट से कूदना",
    "मैं जिन्दगी ते खारा हुआ हूँ मुझे नहीं दीना",
    # the same, correctly spelled and other garbles
    "मेरा गोल है छत से कूदना",
    "मैं ज़िंदगी से हार गया हूँ मुझे नहीं जीना",
    "मैं जिंदगी से हारा हुआ हूं मुझे नहीं जीना",
    "मुझे नहीं जिना",
    "मुझे जीना नहीं है",
    "मेरा जीने का मन नहीं है",
    "जीने का मन नहीं करता",
    "मुझे अब जीना ही नहीं है",
    "मैं मर जाऊं तो अच्छा है",
    "मैं मर जाऊँगा",
    "मैं मर जाऊंगी",
    "मुझे मरना है",
    "मैं मरना चाहता हूं",
    "मैं मरना चाहती हूँ",
    "मरने का मन करता है",
    "मर जाने का मन है",
    "मैं खुद को खत्म कर लूंगा",
    "मैं खुद को खत्म कर दूंगी",
    "मुझे आत्महत्या करनी है",
    "मैं आत्महत्या कर लूंगा",
    "मैं खुदकुशी करना चाहता हूँ",
    "मुझे सुसाइड करना है",
    "मैं बिल्डिंग से कूद जाऊंगा",
    "मैं पुल से कूद जाऊंगा",
    "मैं छत से कूद जाऊंगा",
    "मैं ट्रेन के आगे कूद जाऊंगा",
    "मैं ट्रेन के आगे आ जाऊंगा मुझे मरना है",
    "मैं नस काट लूंगा",
    "मैं अपनी नस काट लूंगी",
    "मैं ज़हर खा लूंगा",
    "मुझे ज़हर पीना है",
    "मैं फांसी लगा लूंगा",
    "मैं पंखे से लटक जाऊंगा",
    "मैं नींद की गोलियां सारी खा लूंगा",
    "जिंदगी खत्म कर दूंगा मैं",
    "मेरी जिंदगी खत्म हो गई है",
    "जी कर क्या करूंगा मैं",
    "मैं जिन्दगी से थक गया हूं मुझे जीना नहीं",
    # Hinglish
    "mujhe jeena nahi hai",
    "main jeena nahi chahta",
    "mera jeene ka mann nahi hai",
    "jeene ka man nahi karta",
    "mujhe marna hai",
    "main marna chahta hoon",
    "mar jaunga main ab",
    "main mar jaun to accha hai",
    "marne ka man karta hai",
    "main khud ko khatam kar lunga",
    "mujhe suicide karna hai",
    "main atmahatya kar lunga",
    "main khudkushi karunga",
    "main chhat se kud jaunga",
    "mera goal hai chhat se koodna",
    "main building se kud jaunga",
    "main bridge se kud jaunga",
    "main train ke aage kud jaunga",
    "main nas kaat lunga",
    "main zeher kha lunga",
    "mujhe poison pina hai",
    "main fansi laga lunga",
    "main pankhe se latak jaunga",
    "main neend ki goliyan saari kha lunga",
    "zindagi khatam kar dunga",
    "main zindagi se haar gaya hoon mujhe jeena nahi",
    "main jindagi se thak gaya hu jeena nahi chahta",
    # English
    "i want to die",
    "I wanna die",
    "i am going to kill myself",
    "I want to end my life",
    "i just want to end it all",
    "i dont want to live anymore",
    "there is no point in living",
    "no reason to live",
    "i'm going to jump off the roof",
    "I will jump from the bridge",
    "I feel like hanging myself",
    "I am going to cut my wrists",
    "I think about suicide every day",
    "everyone would be better off dead without me",
    "I can't go on and I wish I was dead",
]

NEGATIVES = [
    "gym me jump squats karne hain",
    "kood ke exercise karna hai",
    "box jump bahut mushkil hai",
    "jump rope kitne ki hai",
    "roof top cafe me milte hain",
    "roof par yoga karte hain",
    "chhat par gym banaya hai",
    "terrace pe workout karte hain",
    "building ke neeche gym hai",
    "muscle building ke liye protein chahiye",
    "job se haar gaya match mein",
    "match me haar gaya hum",
    "cricket mein team haar gayi",
    "election mein haar gaye woh",
    "price se mar gaya yaar",
    "itna mehnga price sunke mar gaya",
    "mar jaunga hans hans ke",
    "main hans hans ke mar jaunga",
    "deadlift se jaan nikal gayi",
    "squats ke baad jaan nikal gayi",
    "workout me thak gaya hoon",
    "zindagi me pehli baar itna thak gaya workout karke",
    "gym me bahut mehnat karni padti hai",
    "mujhe jeena sikhna hai gym se",
    "mujhe paise nahi dena",
    "main fees nahi dunga",
    "mujhe abhi nahi chahiye",
    "mujhe marna nahi hai",
    "main marna nahi chahta",
    "marne se darta hoon",
    "ghar jaana hai mujhe",
    "mujhe nahi jaana",
    "mar gaya traffic mein pura din",
    "fees sunke mar gaya",
    "mera naam raju hai",
    "price kitna hai trial ka",
    "link bhej do whatsapp pe",
    "zindagi me pehli baar gym join kar raha hoon",
    "mujhe weight loss karna hai",
    "mera goal hai chhe mahine me chhe kilo kam karna",
    "mera goal hai pet kam karna",
    "meri zindagi badal gayi gym se",
    "zindagi se bahut khush hoon",
    "life me kuch karna hai",
    "i dont want to die without trying this gym",
    "killing it at the gym today",
    "this workout will kill me lol",
    "i want to join the gym",
    "my goal is to lose weight",
    "what is the price",
    "I want to jump start my fitness",
    "I want to end the call",
    "yes please send the link",
    "mujhe ghar jaldi jaana hai",
    "baad me baat karte hain thak gaya hoon",
    "garmi se mar gaya yaar",
    "neend ki goli leta hoon doctor ne di hai kal tak",
    "dost ne fansi ki sajaa wali movie dekhi",
    "bridge course karna hai mujhe",
    "pul ke paas gym hai",
    "train ke aage wali seat chahiye",
    "main kal chhat par gym dekhne aaunga",
    "haan haan theek hai",
    "मुझे जिम जॉइन करना है",
    "मेरा गोल है वजन कम करना",
    "छत पर योगा करते हैं",
    "मुझे पैसे नहीं देना",
    "मैं मर जाऊंगा हंस हंस के",
    "प्राइस सुनकर मर गया",
    "मैं मैच में हार गया",
    "जिम में कूद कर एक्सरसाइज करनी है",
    "डेडलिफ्ट से जान निकल गई",
    "मुझे मरना नहीं है",
    "ज़िंदगी में पहली बार जिम आया हूँ",
    "नहीं अभी नहीं चाहिए",
    "मुझे घर जाना है",
]

CONCERN = [
    "छत से कूदना",                      # method, no self-reference
    "ज़िंदगी से हारा",                    # hopelessness only
    "mujhe zindagi se haar mil gayi",
]


@pytest.mark.parametrize("text", POSITIVES)
def test_positive_is_crisis(text):
    score, ev = distress_score(text)
    assert score >= DISTRESS_T, (text, score, ev, skeletons(text))
    assert is_distress(text)


@pytest.mark.parametrize("text", NEGATIVES)
def test_negative_stays_clean(text):
    score, ev = distress_score(text)
    assert score < CONCERN_T, (text, score, ev, skeletons(text))
    assert not is_distress(text) and not is_concern(text)


@pytest.mark.parametrize("text", CONCERN[:2])
def test_concern_band(text):
    assert is_concern(text) and not is_distress(text), distress_score(text)


def test_live_call_exact_transcripts():
    assert is_distress("मेरा गोल है छट से कूदना")
    assert is_distress("मैं जिन्दगी ते खारा हुआ हूँ मुझे नहीं दीना")
    assert not is_distress("मुझे पैसे नहीं देना")


def test_skeleton_spellings_agree():
    assert skeletons("छत") == skeletons("छट") == skeletons("chhat")
    assert skeletons("जीना") == skeletons("दीना") == skeletons("जिना") == skeletons("jeena")
    assert skeletons("हारा") == skeletons("खारा")
    assert skeletons("ज़िंदगी") == skeletons("जिन्दगी") == skeletons("जिंदगी") == skeletons("zindagi")
    assert skeletons("देना") != skeletons("जीना")  # "to give" is not "to live"
    assert skeletons("मरना") == skeletons("marna")


def test_two_turn_history_adds_up():
    # each turn alone is only worrying; together they are a crisis
    assert is_concern("छत से कूदना") and is_concern("ज़िंदगी से हारा")
    assert is_distress("ज़िंदगी से हारा", ["छत से कूदना"])
    assert is_distress("जिंदगी से हारा", ["मेरा गोल है छत से कूदना"])
    assert is_distress("छत से कूदना", ["hmm", "मेरा गोल है"])  # self-reference in the previous turn
    # history alone never re-fires on an unrelated current turn


def test_old_turn_does_not_refire_on_unrelated_turn():
    assert distress_score("haan bhej do link", ["छत से कूदना"])[0] < CONCERN_T


def test_history_formats():
    h = [("agent", "hello"), ("customer", "छत से कूदना"), ("agent", "kya?")]
    assert is_distress("ज़िंदगी से हारा", h)


def test_backward_compatible_single_arg():
    assert is_distress("I want to die") and not is_distress("hello")


def test_nlu_second_net():
    assert crisis_from_nlu(Understanding(crisis=True, crisis_reason="garbled suicidal"))
    assert not crisis_from_nlu(Understanding())
    assert not crisis_from_nlu(None)


def test_precision_recall_report():
    tp = sum(is_distress(t) for t in POSITIVES)
    fp = sum(is_distress(t) or is_concern(t) for t in NEGATIVES)
    assert tp == len(POSITIVES) and fp == 0

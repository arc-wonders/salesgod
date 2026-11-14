import json, random

CONVS = []
rnd = random.Random(42)


def t(text, barge=False, lat=None, **expect):
    d = {"text": text}
    if lat is not None or barge or rnd.random() < 0.6:
        d["timing"] = {"barge_in": barge,
                       "response_latency_ms": lat or int(rnd.gauss(950, 220)) // 10 * 10,
                       "speech_rate_wps": round(rnd.gauss(2.6, 0.3), 1)}
    if expect:
        d["expect"] = expect
    return d


def C(cust, desc, *turns):
    CONVS.append({"id": f"c{len(CONVS)+1:03d}", "customer_id": cust, "description": desc, "turns": list(turns)})


# ===================== mainly Hindi =====================
C("c1", "Hindi caller asks over-limit transfer, gets confused by 'limit', accepts the remaining 25k",
  t("हेलो", intent="greet"),
  t("हां वो पचास हज़ार भेजना है भाई को", intent="transfer_over_limit"),
  t("मतलब लिमिट क्या होती है मेरा पैसा है तो", lat=1700, complexity="low"),
  t("अच्छा तो पच्चीस हज़ार ही भेज दो", intent="transfer_confirm"),
  t("हां", intent="transfer_done"),
  t("बस धन्यवाद", intent="goodbye"))

C("c2", "Hindi caller tries 15k with only 12k balance, settles for 10k",
  t("बैलेंस बताओ", intent="check_balance"),
  t("पंद्रह हज़ार ट्रांसफर करना है", intent="transfer_insufficient"),
  t("अच्छा दस हज़ार कर दो फिर", intent="transfer_confirm"),
  t("हां कर दो", intent="transfer_done"),
  t("ओके बाय", intent="goodbye"))

C("c1", "Hindi caller asks about the card offer after balance and applies",
  t("बैलेंस कितना है मेरा", intent="check_balance"),
  t("अच्छा"),
  t("वो क्रेडिट कार्ड का मैसेज आया था वो क्या है", intent="card_offer"),
  t("सालाना फीस कितनी है इसकी"),
  t("ठीक है बनवा दो", intent="card_apply_started"),
  t("धन्यवाद", intent="goodbye"))

C("c1", "Hindi caller declines card, then declines again angrily; emotion should be acknowledged once",
  t("बैलेंस", intent="check_balance"),
  t("कार्ड वाला क्या ऑफर है", intent="card_offer"),
  t("नहीं चाहिए", intent="card_declined"),
  t("अरे कितनी बार बोलूं नहीं चाहिए कार्ड", barge=True, intent="card_declined_again", acknowledge_emotion=True, verbosity="low"),
  t("बस रखता हूं", intent="goodbye", acknowledge_emotion=False))

C("c2", "Hindi caller (not eligible) asks for a credit card, then checks balance",
  t("हेलो जी", intent="greet"),
  t("मुझे क्रेडिट कार्ड चाहिए था", intent="card_ineligible"),
  t("क्यों नहीं मिलेगा"),
  t("ठीक है बैलेंस बता दो", intent="check_balance"),
  t("ठीक है धन्यवाद", intent="goodbye"))

C("c1", "Starts in Hindi, explicitly asks for English, agent must switch immediately and stay",
  t("हेलो", intent="greet"),
  t("मेरी ट्रांसफर लिमिट कितनी है", intent="transfer_limit"),
  t("इंग्लिश में बोलो प्लीज़", language="en"),
  t("ok send twenty thousand to my brother", intent="transfer_confirm", language="en"),
  t("yes", intent="transfer_done", language="en"),
  t("bye", intent="goodbye"))

C("c1", "Hindi caller hits 30k over remaining limit (under 40k daily cap), clear frustration, accepts 25k",
  t("पैसे भेजने हैं", intent="transfer_ask_amount"),
  t("तीस हज़ार", intent="transfer_over_limit"),
  t("अरे यार ये क्या है अपना ही पैसा है कितनी बार यही सुनूं", barge=True, acknowledge_emotion=True, verbosity="low"),
  t("ठीक है पच्चीस कर दो", intent="transfer_confirm", acknowledge_emotion=False),
  t("हां", intent="transfer_done"))

C("c1", "Hindi caller with a blocked debit card (out of scope) asks for a human",
  t("हेलो जी नमस्ते", intent="greet"),
  t("बैलेंस बताओ", intent="check_balance"),
  t("और मेरा एटीएम कार्ड ब्लॉक हो गया है", intent="unknown"),
  t("किसी आदमी से बात कराओ", intent="human_agent"))

C("c2", "Elderly Hindi caller checking pension, confused by the answer, then sends 5k to grandson",
  t("हां जी नमस्ते", lat=1800, intent="greet"),
  t("पेंशन आई कि नहीं देखो जरा", intent="check_balance"),
  t("बारह हज़ार हां पर पेंशन आई या नहीं समझ नहीं आया", lat=2100, complexity="low"),
  t("अच्छा बेटा पांच हज़ार पोते को भेजना है", intent="transfer_confirm"),
  t("हां जी भेज दो", intent="transfer_done"),
  t("धन्यवाद बेटा", intent="goodbye"))

C("c1", "Hindi caller asks exactly the remaining 25k (boundary), then cancels",
  t("पच्चीस हज़ार ट्रांसफर करना है", intent="transfer_confirm"),
  t("नहीं नहीं रुको मत भेजो", barge=True, intent="cancelled"),
  t("बैलेंस कितना है", intent="check_balance"),
  t("ठीक है बाय", intent="goodbye"))

C("c2", "Hindi caller: 20k is within limit but above balance -> insufficient, then 10k",
  t("आज कितना भेज सकती हूं", intent="transfer_limit"),
  t("बीस हज़ार भेजना है", intent="transfer_insufficient"),
  t("अच्छा तो दस हज़ार", intent="transfer_confirm"),
  t("हां", intent="transfer_done"))

# ===================== mainly English =====================
C("c1", "English caller: balance, over-limit 50k, confused about which limit, accepts 25k",
  t("hello", intent="greet"),
  t("whats my balance", intent="check_balance"),
  t("ok send fifty thousand to my landlord", intent="transfer_over_limit"),
  t("which limit", lat=1600, complexity="low"),
  t("fine twenty five then", intent="transfer_confirm"),
  t("yes", intent="transfer_done"),
  t("thanks bye", intent="goodbye"))

C("c2", "English caller over balance, checks balance, starts 10k transfer then cancels",
  t("i want to send fifteen thousand", intent="transfer_insufficient"),
  t("how much do i have", intent="check_balance"),
  t("okay send ten thousand", intent="transfer_confirm"),
  t("no wait dont", barge=True, intent="cancelled"),
  t("bye", intent="goodbye"))

C("c1", "English caller enquires about SMS card offer and applies",
  t("hi", intent="greet"),
  t("i got some sms about a credit card offer", intent="card_offer"),
  t("whats the annual fee"),
  t("okay sign me up", intent="card_apply_started"),
  t("thanks", intent="goodbye"))

C("c1", "English caller: 40k is the daily cap but only 25k left; confusion then frustration and handoff",
  t("transfer money", intent="transfer_ask_amount"),
  t("forty thousand", intent="transfer_over_limit"),
  t("what i never set any limit", lat=1500, complexity="low"),
  t("this is ridiculous just connect me to a person", barge=True, intent="human_agent", acknowledge_emotion=True, verbosity="low"))

C("c2", "English caller not eligible for card, mildly annoyed, checks balance",
  t("am i eligible for a credit card", intent="card_ineligible"),
  t("why not ive been with you for like ten years"),
  t("forget it whats my balance", intent="check_balance"),
  t("bye", intent="goodbye"))

C("c1", "English caller explicitly switches to Hindi mid-call",
  t("hello", intent="greet"),
  t("transfer limit please", intent="transfer_limit"),
  t("हिंदी में बोलो", language="hi"),
  t("बीस हज़ार भेज दो", intent="transfer_confirm", language="hi"),
  t("हां", intent="transfer_done", language="hi"))

C("c1", "English caller declines card twice (second time curt)",
  t("check my balance", intent="check_balance"),
  t("tell me about the credit card thing", intent="card_offer"),
  t("no thanks", intent="card_declined"),
  t("i said no", intent="card_declined_again"),
  t("bye", intent="goodbye"))

C("c2", "Rushed English caller, small transfer within balance",
  t("balance quick im driving", intent="check_balance"),
  t("send five thousand to ramesh", intent="transfer_confirm"),
  t("yes yes", barge=True, intent="transfer_done"),
  t("bye", intent="goodbye"))

C("c1", "English caller: slot-filling, boundary 25k, confused about status before confirming",
  t("i need to make a transfer", intent="transfer_ask_amount"),
  t("um twenty five thousand", intent="transfer_confirm"),
  t("wait so the money went or not", lat=1400, complexity="low"),
  t("yes go ahead", intent="transfer_done"),
  t("thank you", intent="goodbye"))

C("c1", "Non-native English caller with out-of-scope debit card block, then balance, then human",
  t("hello madam", intent="greet"),
  t("my debit card got blocked", intent="unknown"),
  t("ok then balance please", intent="check_balance"),
  t("i want person to talk for card block", intent="human_agent"))

C("c2", "English caller asks limit, then asks more than both limit and balance (no intent asserted), settles for 10k",
  t("uh how much can i send to day", intent="transfer_limit"),
  t("okay send thirty thousand to my sister"),
  t("hmm ok ten thousand then", intent="transfer_confirm"),
  t("yeah", intent="transfer_done"),
  t("thats all thanks", intent="goodbye"))

# ===================== mixed / switching =====================
C("c1", "Hinglish caller over-limit, confused ('limit wimit'), accepts 25k",
  t("hello ji", intent="greet"),
  t("balance batao", intent="check_balance"),
  t("pachas hazaar bhejna hai", intent="transfer_over_limit"),
  t("ye limit wimit kya hai mujhe nahi pata", lat=1600, complexity="low"),
  t("acha pachees hazaar kar do", intent="transfer_confirm"),
  t("haan", intent="transfer_done"))

C("c2", "English to Hinglish explicit switch, not eligible for card",
  t("can i get a credit card", intent="card_ineligible"),
  t("thoda mix karke bolo na hindi english dono", language="hinglish"),
  t("acha balance kitna hai", intent="check_balance", language="hinglish"),
  t("theek hai bye", intent="goodbye"))

C("c1", "Hindi start, switches to English explicitly, applies for card",
  t("हेलो जी", intent="greet"),
  t("कार्ड वाला ऑफर बताओ", intent="card_offer"),
  t("sorry english please", language="en"),
  t("ok whats the limit on the card", language="en"),
  t("yes apply", intent="card_apply_started", language="en"))

C("c1", "Hinglish transfer within limit, then declines card offer",
  t("paise bhejne hai", intent="transfer_ask_amount"),
  t("das hazaar", intent="transfer_confirm"),
  t("haan kar do", intent="transfer_done"),
  t("credit card ka kya offer tha", intent="card_offer"),
  t("nahi abhi nahi", intent="card_declined"),
  t("bas bye", intent="goodbye"))

C("c2", "Hinglish/Hindi caller over balance, confused about why",
  t("mera balance kitna hai", intent="check_balance"),
  t("bees hazaar transfer karna hai", intent="transfer_insufficient"),
  t("matlab mera paisa kam hai kya samjha nahi", lat=1700, complexity="low"),
  t("acha theek hai das hazaar bhejo", intent="transfer_confirm"),
  t("haan", intent="transfer_done"),
  t("bye", intent="goodbye"))

C("c1", "Elderly caller starts in English, asks for Hindi, sends 10k, then wants a human",
  t("hello hello", intent="greet"),
  t("my pension came or not", intent="check_balance"),
  t("बेटा हिंदी में बोलो", language="hi"),
  t("पोते को दस हज़ार भेजना है", intent="transfer_confirm", language="hi"),
  t("हां", intent="transfer_done", language="hi"),
  t("किसी आदमी से बात हो सकती है क्या", intent="human_agent", language="hi"))

with open("/home/arkin/salesgod/data/eval/conversations.jsonl", "w", encoding="utf-8") as fh:
    for c in CONVS:
        fh.write(json.dumps(c, ensure_ascii=False) + "\n")
print(len(CONVS))

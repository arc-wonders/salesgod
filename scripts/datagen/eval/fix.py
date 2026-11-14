import glob,re
subs = [
 ('T("en", "transfer 40 to rahul", "transfer", amt=40000, c=0, notes="ASR dropped \'thousand\'? likely 40000 given context; ambiguous, best guess 40000",)',
  'T("en", "transfer 40 to rahul", "transfer", amt=None, notes="ASR likely dropped \'thousand\' (or means 40 rupees); genuinely ambiguous -> null, system should ask")'),
 ('c=1, ex={"verbosity": "low"}, agent=OL_EN, notes="slowly -> not exactly verbosity; arguably complexity. Keep explicit verbosity low? judged: asks slower, treated as verbosity low"),',
  'c=1, agent=OL_EN, notes="\'slowly\' is a pace request, not verbosity/complexity; explicit left empty")'),
 ('T("en", "the balance part again but hindi mein", "check_balance"', 'T("hinglish", "woh balance wala phir se bolo hindi mein", "check_balance"'),
 ('"हां मुझे ये बताओ क्रेडिट कार्ड की लिमिट और मेरी ट्रांसफर लिमिट अलग है"', '"रुको ये कार्ड की लिमिट और वो ट्रांसफर वाली लिमिट अलग अलग है क्या"'),
 ('"sorry i don\'t understand hindi can you continue in english"', '"sorry sorry i dont know hindi english please"'),
 ('"what is the interest rate on fixed deposit"', '"what is fd interest rate now"'),
 ('"how much can i send today"', '"how much can i send to day"'),
 ('"what\'s my daily transfer limit", "transfer_limit", notes="baseline"', '"whats my daily transfer limited", "transfer_limit", notes="ASR artefact limit->limited"'),
 ('"am i eligible for a credit card"', '"am i a legible for credit card"'),
 ('"my debit card is blocked can you unblock it"', '"my debit card got blocked can you unlock it"'),
]
barge = ['"this is ridiculous"', '"हद है"', '"arre yaar kitni baar bolna padega"', '"i\'m not interested i told you already"',
 '"अरे मना किया ना कार्ड नहीं चाहिए"', '"card nahi chahiye kitni baar bolu"', '"just connect me to customer care"', '"short mein batao"',
 '"सीधा बताओ कितना है बस"', '"keep it short i\'m in a meeting"', '"hello hello can you hear me hello"', '"हां हां बिल्कुल"']
for fn in sorted(glob.glob("b*.py")):
    s = open(fn).read(); o = s
    for a,b in subs:
        s = s.replace(a,b)
    for t in barge:
        i = s.find('T("'); 
        for m in re.finditer(r'^T\(.*$', s, re.M):
            line = m.group(0)
            if t in line and "b=True" not in line:
                s = s.replace(line, line.replace('notes=', 'b=True, notes=',1))
    if s != o: open(fn,"w").write(s); print("fixed", fn)

import glob
subs = [
 ('"i need to transfer fifty thousand to my brother"', '"umm i need to transfer like fifty thousand to my brother"'),
 ('"send twenty five thousand rupees to ramesh"', '"send twenty five thousand rupees to ramesh to ramesh kumar"'),
 ('"i want to check my balance"', '"yeah i want to check my balance"'),
 ('"how much money is there in my account"', '"uh how much money is there in my in my account"'),
 ('"can i talk to a person please"', '"can i talk to a a person please"'),
 ('"how do i apply for the credit card"', '"so how do i apply for the uh credit card"'),
 ('"can you put me through to the branch manager"', '"can you put me to the branch manager"'),
 ('"what\'s the annual fee on that card"', '"whats the annual fee on that"'),
 ('"okay tell me more about it"', '"okay okay tell me"'),
 ('"i got an sms about some credit card offer"', '"i got some sms about a credit card offer something"'),
 ('"खाते में कितने पैसे हैं"', '"खाते में कितने पैसे हैं अभी"'),
 ('"मुझे कस्टमर केयर से बात करनी है"', '"मुझे वो कस्टमर केयर से बात करनी है"'),
 ('"what\'s my balance"', '"whats my balance"'),
 ('"can you say it in simple words"', '"uh can you say it in simple words"'),
 ('"is there someone i can speak to about this"', '"is there someone i can speak to about this like a person"'),
 ('"alright i\'ll call back later"', '"alright ill call back later"'),
 ('"i don\'t want to talk to a machine get me someone real"', '"i dont want to talk to a machine get me someone"'),
]
for fn in sorted(glob.glob("b*.py")):
    s=open(fn).read(); o=s
    for a,b in subs: s=s.replace(a,b)
    if s!=o: open(fn,"w").write(s); print("fixed",fn)

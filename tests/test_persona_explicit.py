import pytest

from acie.personalization.policy import detect_explicit_requests as d


@pytest.mark.parametrize("text,key,val", [
    ("Delhi wali Hindi mein baat karo", "regional_variant", "delhi"),
    ("dilli style me bolo", "regional_variant", "delhi"),
    ("Mumbai style please", "regional_variant", "mumbai"),
    ("mumbaiya hindi chalegi", "regional_variant", "mumbai"),
    ("bihari mein samjhao", "regional_variant", "bihar"),
    ("lucknowi andaaz mein", "regional_variant", "lucknow"),
    ("Tamil accent English", "regional_variant", "south_indian_english"),
    ("speak in south indian english", "regional_variant", "south_indian_english"),
    ("दिल्ली वाली हिंदी बोलो", "regional_variant", "delhi"),
    ("normal hindi mein bolo", "regional_variant", "standard"),
    ("example dekar samjhao", "explanation_style", "example"),
    ("can you give me an example", "explanation_style", "example"),
    ("एक उदाहरण दीजिए", "explanation_style", "example"),
    ("step by step batao", "explanation_style", "step_by_step"),
    ("ek ek karke batao", "explanation_style", "step_by_step"),
    ("स्टेप बाय स्टेप समझाओ", "explanation_style", "step_by_step"),
    ("seedha batao", "explanation_style", "direct"),
    ("just tell me the answer", "explanation_style", "direct"),
    ("sirf zaroori baat batao", "info_density", "low"),
    ("only the important things please", "info_density", "low"),
    ("poori jaankari chahiye", "info_density", "high"),
    ("tell me all the details", "info_density", "high"),
    ("पूरी जानकारी दीजिए", "info_density", "high"),
    ("mujhe dar lag raha hai", "reassurance", "high"),
    ("I am worried about this", "reassurance", "high"),
    ("is it safe?", "reassurance", "high"),
    ("मुझे घबराहट हो रही है", "reassurance", "high"),
    ("koi tension nahi hai", "reassurance", "low"),
])
def test_detects(text, key, val):
    assert d(text).get(key) == val


@pytest.mark.parametrize("text,key", [
    ("I am not worried", "reassurance"),
    ("tension mat lo, dar nahi lag raha", "reassurance"),
    ("delhi wali nahi chahiye", "regional_variant"),
    ("no examples please", "explanation_style"),
    ("step by step nahi, seedha batao", "explanation_style"),
    ("hello how are you", "regional_variant"),
    ("price kya hai", "info_density"),
    ("the dar of the bond", "reassurance"),
])
def test_no_false_positive_or_blocked(text, key):
    out = d(text)
    if text.startswith("step by step nahi"):
        assert out.get(key) == "direct"
    else:
        assert key not in out or (key == "reassurance" and out[key] == "low")


def test_existing_keys_unchanged():
    assert d("hindi mein baat karo") == {"language": "hi"}
    assert d("short mein batao")["verbosity"] == "low"
    assert d("please speak english") == {"language": "en"}


def test_multiple_requests_one_turn():
    out = d("Mumbai style mein example dekar samjhao, sirf zaroori baat")
    assert out["regional_variant"] == "mumbai" and out["explanation_style"] == "example" and out["info_density"] == "low"

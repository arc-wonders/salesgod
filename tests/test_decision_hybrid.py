from acie.decision.domains.product_sales import Product, ProductSalesDomain
from acie.decision.engine import KeywordIntentClassifier
from acie.decision.hybrid import HybridIntentClassifier, load_turn_model
from acie.schemas import SessionState


class FakeModel:
    def __init__(self, label="price", conf=0.9, boom=False):
        self.label, self.conf, self.boom, self.calls = label, conf, boom, 0

    def classify_with_confidence(self, text, session):
        self.calls += 1
        if self.boom:
            raise RuntimeError("model down")
        return self.label, self.conf


def sess(**dialog):
    s = SessionState(session_id="s", customer_id="c")
    s.dialog.update(dialog)
    return s


def dom(model):
    return ProductSalesDomain(Product(), turn_model=model)


def test_confident_model_wins_over_keywords():
    d = dom(FakeModel("benefits", 0.9))
    assert isinstance(d.classifier, HybridIntentClassifier)
    assert d.classifier.classify("hmm interesting stuff", sess()) == "benefits"


def test_low_confidence_falls_back_to_keywords():
    d = dom(FakeModel("benefits", 0.3))
    assert d.classifier.classify("how much is it", sess()) == "price"


def test_gating_affirm_deny_need_pending_question():
    d = dom(FakeModel("affirm", 0.99))
    assert d.classifier.classify("hmm interesting stuff", sess()) != "affirm"
    assert d.classifier.classify("hmm interesting stuff", sess(pending="close")) == "affirm"
    d = dom(FakeModel("deny", 0.99))
    assert d.classifier.classify("hmm interesting stuff", sess()) != "deny"


def test_rule_priority_beats_model():
    d = dom(FakeModel("price", 0.99))
    assert d.classifier.classify("abhi busy hoon baad mein baat karo", sess()) == "callback"
    assert d.classifier.classify("can you repeat that", sess()) == "clarify"


def test_unknown_label_or_error_falls_back_silently():
    assert dom(FakeModel("not_a_domain_intent", 0.99)).classifier.classify("how much", sess()) == "price"
    assert dom(FakeModel("unknown", 0.99)).classifier.classify("how much", sess()) == "price"
    assert dom(FakeModel(boom=True)).classifier.classify("how much", sess()) == "price"


def test_bare_classify_model_is_trusted():
    class Bare:
        def classify(self, text, session):
            return "benefits"
    assert dom(Bare()).classifier.classify("mm okay", sess()) == "benefits"


def test_env_unset_or_bad_path_means_keywords(monkeypatch):
    monkeypatch.delenv("ACIE_TURN_MODEL", raising=False)
    assert isinstance(ProductSalesDomain(Product()).classifier, KeywordIntentClassifier)
    monkeypatch.setenv("ACIE_TURN_MODEL", "/nonexistent/model-dir")
    assert load_turn_model() is None
    assert isinstance(ProductSalesDomain(Product()).classifier, KeywordIntentClassifier)

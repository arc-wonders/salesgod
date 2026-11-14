from __future__ import annotations

import re

from ..schemas import ExpressionParams, TurnPlan

_SECRET = re.compile(r"\b(otp|one[- ]time (password|code)|pin|cvv|cvc|password|passcode|mpin|upi pin)\b|ओटीपी|पिन|सीवीवी|पासवर्ड",
                     re.I)
# "never share your OTP with anyone" is a warning, not a request
_WARNING = re.compile(r"\b(never|don'?t|do not|not|won'?t|nobody|no one|kabhi nahi|mat|nahi|nahin)\b|कभी नहीं|मत|नहीं|किसी को नहीं|न ही",
                      re.I)
_ACCOUNT_ASK = re.compile(
    r"(tell|share|give|read|send|confirm|provide|enter|bata|batao|bataiye|bhejiye|bhejo|dijiye|do)\w*.{0,30}"
    r"(account|a/c|card) (number|no\b)|(account|a/c|card) (number|no\b).{0,30}"
    r"(tell|share|give|read|send|confirm|provide|enter|bata|batao|bataiye|bhejiye|bhejo|dijiye)|"
    r"(अकाउंट|खाता|कार्ड) (नंबर|संख्या)", re.I)
_LONG_DIGITS = re.compile(r"(?<!\d)(?:\d[ -]?){9,}\d(?!\d)")


class BankingPack:
    name = "banking"

    def check(self, sentence: str, plan: TurnPlan, expr: ExpressionParams) -> str | None:
        if _SECRET.search(sentence) and not _WARNING.search(sentence):
            return "asks for OTP/PIN/CVV/password"
        if _ACCOUNT_ASK.search(sentence) and not _WARNING.search(sentence):
            return "asks for an account or card number"
        if _LONG_DIGITS.search(sentence):
            return "account-number-like digits"
        return None
